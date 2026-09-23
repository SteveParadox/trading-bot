from __future__ import annotations

import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fxbot.journal import StructuredJournal, row_to_dict
from fxbot.models import BotRunState
from fxbot.config import BrokerSettings, FxBotSettings, RuntimeSettings
from fxbot.forward import ForwardTestWorker

class StructuredJournalTests(unittest.TestCase):
    def test_equity_history_is_chronological_and_keeps_date_range_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with closing(StructuredJournal(f"sqlite:///{Path(tmp) / 'journal.db'}")) as journal:
                start = datetime(2026, 1, 1, tzinfo=timezone.utc)
                for index in range(20):
                    timestamp = start + timedelta(hours=index)
                    journal.record_equity(timestamp=timestamp, payload={"equity": 1000 + index, "balance": 1000})
                rows = journal.equity_history(start=start + timedelta(hours=4), end=start + timedelta(hours=15), limit=4)
                actual = [row.timestamp.replace(tzinfo=timezone.utc) for row in rows]
                self.assertEqual(actual[0], start + timedelta(hours=4))
                self.assertEqual(actual[-1], start + timedelta(hours=15))
                self.assertEqual(actual, sorted(actual))
                self.assertEqual(len(actual), 4)
                serialized = row_to_dict(rows[0])
                self.assertTrue(serialized["timestamp"].endswith("+00:00"))

    def test_forward_sync_records_closed_history_and_updates_existing_trade(self) -> None:
        class HistoryClient:
            def __init__(self):
                self.since = None

            def closed_trades_since(self, since, until):
                self.since = since
                return [{
                    "broker_trade_id": "mt5-closed-1", "instrument": "EUR_USD", "side": "LONG",
                    "units": 1000, "exit_time": datetime(2026, 1, 6, 14, 5, tzinfo=timezone.utc),
                    "exit_price": 1.103, "realized_pl": 2.75, "financing": -0.1,
                    "exit_reason": "mt5_history_deal",
                }]

        with tempfile.TemporaryDirectory() as tmp:
            settings = FxBotSettings(
                broker=BrokerSettings(),
                runtime=RuntimeSettings(database_url=f"sqlite:///{Path(tmp) / 'journal.db'}", log_jsonl_path=None),
            )
            with closing(StructuredJournal(settings.runtime.database_url)) as journal:
                journal.upsert_trade(
                    broker_trade_id="mt5-closed-1", instrument="EUR_USD", side="LONG", units=1000,
                    state="open", entry_time=datetime(2026, 1, 6, 14, tzinfo=timezone.utc),
                    entry_price=1.1, payload={"strategy_context": {"signal_score": 71}},
                )
                client = HistoryClient()
                worker = ForwardTestWorker(settings, client=client, journal=journal)
                worker._sync_trade_history(datetime(2026, 1, 6, 14, 10, tzinfo=timezone.utc))
                closed = journal.find_trade("mt5-closed-1")
                self.assertIsNotNone(closed)
                self.assertEqual(closed.state, "closed")
                self.assertAlmostEqual(closed.realized_pl, 2.75)
                self.assertAlmostEqual(closed.financing, -0.1)
                self.assertEqual(closed.payload["strategy_context"]["signal_score"], 71)
                self.assertEqual(client.since, datetime(2025, 12, 7, 14, 10, tzinfo=timezone.utc))
                worker.close()
    def test_order_reservation_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with closing(StructuredJournal(f"sqlite:///{Path(tmp) / 'journal.db'}")) as journal:
                first, created_first = journal.reserve_order(
                    client_order_id="client-1",
                    timestamp=datetime(2026, 1, 6, 14, 0, tzinfo=timezone.utc),
                    instrument="EUR_USD",
                    side="LONG",
                    units=1000,
                    order_type="MARKET",
                    risk_amount=2.0,
                    payload={"leg": "full"},
                )
                second, created_second = journal.reserve_order(
                    client_order_id="client-1",
                    timestamp=datetime(2026, 1, 6, 14, 0, tzinfo=timezone.utc),
                    instrument="EUR_USD",
                    side="LONG",
                    units=1000,
                    order_type="MARKET",
                    risk_amount=2.0,
                    payload={"leg": "full"},
                )

                self.assertTrue(created_first)
                self.assertFalse(created_second)
                self.assertEqual(first.id, second.id)

    def test_order_status_update_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with closing(StructuredJournal(f"sqlite:///{Path(tmp) / 'journal.db'}")) as journal:
                journal.reserve_order(
                    client_order_id="client-2",
                    timestamp=datetime(2026, 1, 6, 14, 0, tzinfo=timezone.utc),
                    instrument="EUR_USD",
                    side="LONG",
                    units=1000,
                    order_type="MARKET",
                    risk_amount=2.0,
                    payload={},
                )

                journal.update_order("client-2", status="filled", broker_order_id="10", broker_trade_id="11")
                row = journal.find_order("client-2")

                self.assertIsNotNone(row)
                self.assertEqual(row.status, "filled")
                self.assertEqual(row.broker_trade_id, "11")

    def test_current_positions_are_marked_closed_when_absent_from_latest_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with closing(StructuredJournal(f"sqlite:///{Path(tmp) / 'journal.db'}")) as journal:
                timestamp = datetime(2026, 1, 6, 14, 0, tzinfo=timezone.utc)
                journal.record_position_snapshot(
                    timestamp=timestamp,
                    instrument="EUR_USD",
                    side="LONG",
                    units=1000,
                    avg_price=1.1,
                    unrealized_pl=3.0,
                    margin_used=40.0,
                    price=1.103,
                    estimated_daily_financing=-0.02,
                    payload={},
                )

                self.assertEqual(len(journal.current_positions()), 1)

                journal.mark_current_positions_closed(set(), timestamp)

                self.assertEqual(journal.current_positions(), [])

    def test_daily_loss_halt_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with closing(StructuredJournal(f"sqlite:///{Path(tmp) / 'journal.db'}")) as journal:
                timestamp = datetime(2026, 1, 6, 14, 0, tzinfo=timezone.utc)
                self.assertIsNone(
                    journal.update_protection_state(
                        timestamp=timestamp,
                        equity=10_000,
                        max_daily_loss_pct=0.02,
                        max_drawdown_pct=0.10,
                    )
                )

                reason = journal.update_protection_state(
                    timestamp=timestamp,
                    equity=9_750,
                    max_daily_loss_pct=0.02,
                    max_drawdown_pct=0.10,
                )

                self.assertEqual(reason, "daily_loss_halt")
                self.assertEqual(journal.get_state().state, BotRunState.HALTED.value)

    def test_filtered_trade_outcome_uses_realized_pnl_and_financing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with closing(StructuredJournal(f"sqlite:///{Path(tmp) / 'journal.db'}")) as journal:
                timestamp = datetime(2026, 1, 6, 14, 0, tzinfo=timezone.utc)
                journal.upsert_trade(
                    broker_trade_id="winner-after-financing",
                    instrument="EUR_USD",
                    side="LONG",
                    units=1000,
                    state="closed",
                    entry_time=timestamp,
                    realized_pl=2.0,
                    financing=-1.0,
                )
                journal.upsert_trade(
                    broker_trade_id="loser-after-financing",
                    instrument="EUR_USD",
                    side="LONG",
                    units=1000,
                    state="closed",
                    entry_time=timestamp,
                    realized_pl=-2.0,
                    financing=1.0,
                )

                wins = journal.filtered_trades(outcome="win")
                losses = journal.filtered_trades(outcome="loss")

                self.assertEqual([row.broker_trade_id for row in wins], ["winner-after-financing"])
                self.assertEqual([row.broker_trade_id for row in losses], ["loser-after-financing"])

    def test_trade_close_preserves_initial_strategy_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with closing(StructuredJournal(f"sqlite:///{Path(tmp) / 'journal.db'}")) as journal:
                timestamp = datetime(2026, 1, 6, 14, 0, tzinfo=timezone.utc)
                journal.upsert_trade(
                    broker_trade_id="context-preserved",
                    instrument="EUR_USD",
                    side="LONG",
                    units=1000,
                    state="open",
                    entry_time=timestamp,
                    payload={"strategy_context": {"signal_score": 73.5}},
                )
                closed = journal.upsert_trade(
                    broker_trade_id="context-preserved",
                    instrument="EUR_USD",
                    side="LONG",
                    units=1000,
                    state="closed",
                    exit_time=timestamp,
                    realized_pl=5.0,
                    payload={"exit_source": "mt5_history"},
                )

                self.assertEqual(closed.payload["strategy_context"]["signal_score"], 73.5)
                self.assertEqual(closed.payload["exit_source"], "mt5_history")


if __name__ == "__main__":
    unittest.main()
