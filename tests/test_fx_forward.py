from __future__ import annotations

import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from fxbot.config import BrokerSettings, FxBotSettings, RiskSettings, RuntimeSettings, StrategySettings
from fxbot.forward import ForwardTestWorker
from fxbot.instruments import FxInstrument, PriceSnapshot
from fxbot.journal import StructuredJournal
from fxbot.models import BotRunState, FxSignalIntent, Side
from fxbot.risk import FxExitPlan, FxRiskDecision


FIXED_NOW = datetime(2026, 1, 6, 14, 0, tzinfo=timezone.utc)


class FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return FIXED_NOW if tz is None else FIXED_NOW.astimezone(tz)


def trending_frame(start: float, step: float, rows: int = 120) -> pd.DataFrame:
    index = pd.date_range("2026-01-06T07:00:00Z", periods=rows, freq="15min")
    closes = [start + i * step for i in range(rows)]
    opens = [close - step * 0.5 for close in closes]
    highs = [max(open_, close) + abs(step) * 2 for open_, close in zip(opens, closes)]
    lows = [min(open_, close) - abs(step) * 2 for open_, close in zip(opens, closes)]
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [100 + i for i in range(rows)],
        },
        index=index,
    )


class FakeMt5Client:
    def __init__(self, *, entry_frame: pd.DataFrame | None = None, htf_frame: pd.DataFrame | None = None) -> None:
        self.created: list[str] = []
        self.created_orders: list[dict] = []
        self.entry_frame = entry_frame
        self.htf_frame = htf_frame if htf_frame is not None else entry_frame
        last_close = float(entry_frame.iloc[-1]["close"]) if entry_frame is not None else 1.1
        self.price = PriceSnapshot("EUR_USD", bid=last_close - 0.00005, ask=last_close + 0.00005, time=FIXED_NOW)

    def order_by_client_id(self, client_order_id: str):
        return None

    def create_market_order(self, **kwargs):
        self.created.append(kwargs["client_order_id"])
        self.created_orders.append(kwargs)
        broker_order_id = str(100 + len(self.created_orders))
        broker_trade_id = str(200 + len(self.created_orders))
        return {
            "orderCreateTransaction": {"id": broker_order_id},
            "orderFillTransaction": {
                "orderID": broker_order_id,
                "instrument": kwargs["instrument"].name,
                "units": kwargs["signed_units"],
                "price": self.price.mid,
                "time": FIXED_NOW.isoformat(),
                "tradeOpened": {"tradeID": broker_trade_id, "units": kwargs["signed_units"]},
            },
        }

    def instruments(self, names: list[str]):
        return {
            name: FxInstrument(
                name,
                pip_location=-4,
                display_precision=5,
                margin_rate=1 / 30,
                minimum_trade_size=1,
                maximum_order_units=1_000_000,
            )
            for name in names
        }

    def pricing(self, instruments: list[str]):
        return SimpleNamespace(prices={"EUR_USD": self.price}, conversion_rates={}, raw={})

    def account_summary(self):
        return {
            "NAV": 10_000.0,
            "balance": 10_000.0,
            "marginUsed": 0.0,
            "openPositionCount": 0,
            "currency": "USD",
            "positionValue": 0.0,
        }

    def open_positions(self):
        return []

    def open_trades(self):
        return []

    def closed_trades_since(self, since, until=None):
        return []

    def candles(self, instrument: str, timeframe: str, count: int):
        if timeframe == "1h":
            return self.htf_frame.copy()
        return self.entry_frame.copy()


class ForwardWorkerTests(unittest.TestCase):
    def test_scan_once_places_market_order_from_confirmed_indicator_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            entry = trending_frame(1.08, 0.00025)
            htf = trending_frame(1.06, 0.0005)
            settings = FxBotSettings(
                instruments=["EUR_USD"],
                broker=BrokerSettings(),
                strategy=StrategySettings(
                    partial_tp_enabled=False,
                    trade_sessions_utc=(),
                    avoid_rollover_minutes=0,
                    require_volume_confirmation=False,
                    min_atr_pips=0.1,
                    max_atr_pips=30,
                    adx_min=10,
                    htf_adx_min=10,
                ),
                risk=RiskSettings(
                    risk_per_trade_pct=0.01,
                    max_units_per_trade=1_000_000,
                    max_pair_exposure_pct=10.0,
                    max_gross_exposure_pct=10.0,
                    max_currency_exposure_pct=10.0,
                ),
                runtime=RuntimeSettings(
                    database_url=f"sqlite:///{Path(tmp) / 'journal.db'}",
                    log_jsonl_path=str(Path(tmp) / "j.jsonl"),
                ),
            )
            with closing(StructuredJournal(settings.runtime.database_url, settings.runtime.log_jsonl_path)) as journal:
                client = FakeMt5Client(entry_frame=entry, htf_frame=htf)
                worker = ForwardTestWorker(settings, client=client, journal=journal)
                journal.set_state(BotRunState.RUNNING, "test scan")

                with patch("fxbot.forward.datetime", FixedDatetime):
                    worker.scan_once()

                self.assertEqual(len(client.created_orders), 1)
                order_request = client.created_orders[0]
                self.assertEqual(order_request["instrument"].name, "EUR_USD")
                self.assertGreater(order_request["signed_units"], 0)
                self.assertLess(order_request["stop_loss"], client.price.mid)
                self.assertGreater(order_request["take_profit"], client.price.mid)

                signal = journal.recent_signals(limit=1)[0]
                self.assertEqual(signal.status, "accepted")
                self.assertEqual(signal.reason, "signal_and_risk_accepted")
                self.assertEqual(signal.side, Side.LONG.value)

                order = journal.recent_orders(limit=1)[0]
                self.assertEqual(order.status, "filled")
                self.assertEqual(order.instrument, "EUR_USD")
                self.assertEqual(order.side, Side.LONG.value)
                self.assertIsNotNone(order.broker_order_id)
                self.assertIsNotNone(order.broker_trade_id)

                trade = journal.recent_trades(limit=1)[0]
                self.assertEqual(trade.state, "open")
                self.assertEqual(trade.instrument, "EUR_USD")
                self.assertEqual(trade.side, Side.LONG.value)

    def test_idempotent_submit_does_not_duplicate_reserved_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = FxBotSettings(
                broker=BrokerSettings(),
                strategy=StrategySettings(partial_tp_enabled=False),
                risk=RiskSettings(),
                runtime=RuntimeSettings(database_url=f"sqlite:///{Path(tmp) / 'journal.db'}", log_jsonl_path=str(Path(tmp) / "j.jsonl")),
            )
            with closing(StructuredJournal(settings.runtime.database_url, settings.runtime.log_jsonl_path)) as journal:
                client = FakeMt5Client()
                worker = ForwardTestWorker(settings, client=client, journal=journal)
                intent = FxSignalIntent(
                    instrument="EUR_USD",
                    side=Side.LONG,
                    timestamp=datetime(2026, 1, 6, 14, 0, tzinfo=timezone.utc),
                    entry_price=1.1,
                    signal_row={"atr": 0.001},
                )
                decision = FxRiskDecision(
                    allowed=True,
                    reason="accepted",
                    units=1000,
                    signed_units=1000,
                    risk_amount=10,
                    exit_plan=FxExitPlan(1.09, 1.12, 0.01, 0.02, 2.0, 100, 200),
                )

                worker._submit_idempotent(intent, FxInstrument("EUR_USD"), decision)
                worker._submit_idempotent(intent, FxInstrument("EUR_USD"), decision)

                self.assertEqual(len(client.created), 1)


if __name__ == "__main__":
    unittest.main()
