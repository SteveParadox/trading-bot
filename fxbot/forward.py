"""Unattended MetaTrader 5 forward-test worker."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from fxbot.config import FxBotSettings, ensure_runtime_dirs, settings_from_env
from fxbot.instruments import (
    FxInstrument,
    PriceSnapshot,
    estimated_daily_financing_home,
    position_value_home,
)
from fxbot.journal import StructuredJournal
from fxbot.market_hours import trading_allowed_now
from fxbot.models import BotRunState, FxPortfolioState, FxSignalIntent, Side
from fxbot.mt5 import Mt5Client, Mt5CredentialsMissing, Mt5Error, extract_order_ids
from fxbot.risk import FxRiskDecision, FxRiskManager
from fxbot.strategy import build_signal_intent, evaluate_signal_frame, prepare_indicators

log = logging.getLogger(__name__)

Publisher = Callable[[dict[str, Any]], Awaitable[None] | None]


class ForwardTestWorker:
    def __init__(
        self,
        settings: FxBotSettings | None = None,
        *,
        client: Mt5Client | None = None,
        journal: StructuredJournal | None = None,
        publisher: Publisher | None = None,
    ) -> None:
        self.settings = settings or settings_from_env()
        ensure_runtime_dirs(self.settings)
        self.client = client or Mt5Client(self.settings.broker)
        self.journal = journal or StructuredJournal(
            self.settings.runtime.database_url,
            self.settings.runtime.log_jsonl_path,
        )
        self.publisher = publisher
        self.risk = FxRiskManager(self.settings.risk, self.settings.strategy)
        self._stop = asyncio.Event()
        self._instrument_cache: dict[str, FxInstrument] = {}

    async def run_forever(self) -> None:
        await self._publish({"type": "worker_started"})
        while not self._stop.is_set():
            state = self.journal.get_state().state
            if state == BotRunState.STOPPED.value:
                await asyncio.sleep(1.0)
                continue
            if state == BotRunState.PAUSED.value:
                await asyncio.sleep(2.0)
                continue
            if state == BotRunState.HALTED.value:
                await asyncio.sleep(5.0)
                continue
            try:
                await asyncio.to_thread(self.scan_once)
            except Mt5CredentialsMissing as exc:
                self.journal.set_state(BotRunState.PAUSED, "missing_mt5_connection")
                self.journal.log_event("credentials_missing", str(exc), level="error")
                await self._publish({"type": "worker_paused", "reason": "missing_mt5_connection"})
            except Exception as exc:
                log.exception("forward-test scan failed")
                self.journal.log_event("scan_error", str(exc), level="error")
                await self._publish({"type": "scan_error", "error": str(exc)})
            await asyncio.sleep(self.settings.runtime.loop_interval_seconds)

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        shutdown = getattr(self.client, "shutdown", None)
        if shutdown is not None:
            shutdown()
        self.journal.close()

    def scan_once(self) -> None:
        now = datetime.now(timezone.utc)
        if self.journal.get_state().state != BotRunState.RUNNING.value:
            return
        instruments = self._load_instruments()
        prices = self.client.pricing(self.settings.instruments)
        account = self.client.account_summary()
        positions = self.client.open_positions()
        portfolio = self._portfolio_from_broker(
            now,
            instruments,
            prices.prices,
            prices.conversion_rates,
            account=account,
            positions=positions,
        )
        self.journal.record_equity(
            timestamp=now,
            payload={
                "equity": portfolio.equity,
                "balance": portfolio.balance,
                "margin_used": portfolio.margin_used,
                "open_positions": portfolio.open_positions,
                "gross_exposure": portfolio.gross_exposure,
                "portfolio_risk": portfolio.portfolio_risk,
                "account_currency": portfolio.account_currency,
            },
        )
        self._sync_trade_history(now)
        self._sync_open_trades(now, instruments, prices.prices, prices.conversion_rates)
        halt_reason = self.journal.update_protection_state(
            timestamp=now,
            equity=portfolio.equity,
            max_daily_loss_pct=self.settings.risk.max_daily_loss_pct,
            max_drawdown_pct=self.settings.risk.max_drawdown_pct,
        )
        if halt_reason:
            self.journal.log_event("risk_halt", halt_reason, level="warning")
            return

        for name in self.settings.instruments:
            instrument = instruments.get(name)
            price = prices.prices.get(name)
            if instrument is None or price is None:
                self._skip(now, name, "instrument_or_price_unavailable")
                continue
            if portfolio.pair_exposures.get(name, 0.0) > 0:
                self._skip(now, name, "pair_position_open")
                continue
            allowed, reason = trading_allowed_now(name, self.settings.strategy, self.settings.news_events, now)
            if not allowed:
                self._skip(now, name, reason)
                continue
            if price.spread_pips(instrument) > self.settings.strategy.max_spread_pips:
                self._skip(now, name, "spread_filter", {"spread_pips": price.spread_pips(instrument)})
                continue
            self._scan_instrument(now, instrument, price, portfolio, prices.conversion_rates)

    def _scan_instrument(
        self,
        now: datetime,
        instrument: FxInstrument,
        price: PriceSnapshot,
        portfolio: FxPortfolioState,
        conversion_rates: dict[str, float],
    ) -> None:
        entry_frame = prepare_indicators(
            self.client.candles(instrument.name, self.settings.strategy.entry_timeframe, self.settings.strategy.candle_limit)
        )
        htf_frame = prepare_indicators(
            self.client.candles(instrument.name, self.settings.strategy.htf_timeframe, self.settings.strategy.candle_limit)
        )
        decision = evaluate_signal_frame(
            entry_frame,
            htf_frame,
            instrument=instrument,
            settings=self.settings.strategy,
            timestamp=now,
        )
        if decision.signal is None:
            self._skip(now, instrument.name, decision.reason, decision.details)
            return

        last_close = float(entry_frame.iloc[-1]["close"])
        deviation_pips = abs(price.mid - last_close) / instrument.pip_size
        if deviation_pips > self.settings.strategy.max_entry_deviation_pips:
            self._skip(now, instrument.name, "entry_deviation_filter", {"deviation_pips": deviation_pips})
            return

        intent = build_signal_intent(
            entry_frame,
            htf_frame,
            instrument=instrument,
            settings=self.settings.strategy,
            entry_price=price.mid,
            timestamp=now,
        )
        if intent is None:
            self._skip(now, instrument.name, "intent_unavailable")
            return

        risk = self.risk.evaluate_intent(
            intent,
            instrument,
            portfolio,
            conversion_rates=conversion_rates,
            snapshot_quote_factor=price.quote_to_home_factor,
            now=now,
        )
        if not risk.allowed or risk.exit_plan is None:
            self.journal.record_signal(
                timestamp=now,
                instrument=instrument.name,
                status="rejected",
                reason=risk.reason,
                side=intent.side.value,
                score=intent.score,
                entry_price=price.mid,
                payload={"risk": asdict(risk), "intent": asdict(intent)},
            )
            return

        self.journal.record_signal(
            timestamp=now,
            instrument=instrument.name,
            status="accepted",
            reason="signal_and_risk_accepted",
            side=intent.side.value,
            score=intent.score,
            entry_price=price.mid,
            stop_loss=risk.exit_plan.stop_loss,
            take_profit=risk.exit_plan.take_profit,
            risk_amount=risk.risk_amount,
            payload={"risk": asdict(risk), "intent": asdict(intent)},
        )
        self._submit_idempotent(intent, instrument, risk)

    def _submit_idempotent(self, intent: FxSignalIntent, instrument: FxInstrument, risk: FxRiskDecision) -> None:
        if risk.exit_plan is None:
            return
        legs = self._order_legs(intent, risk, instrument)
        for leg_name, units, take_profit in legs:
            client_id = client_order_id(intent, leg_name)
            payload = {
                "instrument": instrument.name,
                "side": intent.side.value,
                "units": units,
                "signed_units": units * intent.side.broker_units_sign,
                "stop_loss": risk.exit_plan.stop_loss,
                "take_profit": take_profit,
                "leg": leg_name,
            }
            row, created = self.journal.reserve_order(
                client_order_id=client_id,
                timestamp=intent.timestamp,
                instrument=instrument.name,
                side=intent.side.value,
                units=units,
                order_type="MARKET",
                risk_amount=risk.risk_amount * (units / risk.units) if risk.units else risk.risk_amount,
                payload=payload,
            )
            if not created and row.status in {"pending", "submitted", "filled", "unknown"}:
                self.journal.log_event(
                    "idempotent_order_skip",
                    f"{client_id} already recorded as {row.status}",
                    payload={"client_order_id": client_id},
                )
                continue
            broker_order = self._broker_order_if_exists(client_id)
            if broker_order:
                self.journal.update_order(client_id, status="submitted", broker_order_id=str(broker_order.get("id")), response=broker_order)
                continue
            try:
                response = self.client.create_market_order(
                    instrument=instrument,
                    signed_units=units * intent.side.broker_units_sign,
                    stop_loss=risk.exit_plan.stop_loss,
                    take_profit=take_profit,
                    client_order_id=client_id,
                    comment=f"{intent.side.value} {instrument.name} {leg_name}",
                )
                broker_order_id, broker_trade_id = extract_order_ids(response)
                self.journal.update_order(
                    client_id,
                    status="filled" if broker_trade_id else "submitted",
                    broker_order_id=broker_order_id,
                    broker_trade_id=broker_trade_id,
                    response=response,
                )
                self._record_fill_trade(response, intent, instrument, units, broker_trade_id)
            except Mt5Error as exc:
                self.journal.update_order(client_id, status="unknown", error=str(exc))
                raise

    def _order_legs(
        self,
        intent: FxSignalIntent,
        risk: FxRiskDecision,
        instrument: FxInstrument,
    ) -> list[tuple[str, float, float]]:
        if risk.exit_plan is None or not self.settings.strategy.partial_tp_enabled:
            return [("full", risk.units, risk.exit_plan.take_profit if risk.exit_plan else intent.entry_price)]
        tp1_units = instrument.round_units(risk.units * self.settings.strategy.tp1_units_pct)
        tp2_units = instrument.round_units(risk.units - tp1_units)
        if tp1_units <= 0 or tp2_units <= 0:
            return [("full", risk.units, risk.exit_plan.take_profit)]
        tp2_distance = risk.exit_plan.reward_distance * self.settings.strategy.tp2_multiplier
        tp2 = (
            intent.entry_price + tp2_distance
            if intent.side is Side.LONG
            else intent.entry_price - tp2_distance
        )
        tp2 = instrument_price_round(intent, tp2)
        return [("tp1", tp1_units, risk.exit_plan.take_profit), ("tp2", tp2_units, tp2)]

    def _broker_order_if_exists(self, client_id: str) -> dict[str, Any] | None:
        try:
            return self.client.order_by_client_id(client_id)
        except Mt5Error:
            return None

    def _load_instruments(self) -> dict[str, FxInstrument]:
        missing = [name for name in self.settings.instruments if name not in self._instrument_cache]
        if missing:
            self._instrument_cache.update(self.client.instruments(missing))
        return self._instrument_cache

    def _portfolio_from_broker(
        self,
        now: datetime,
        instruments: dict[str, FxInstrument],
        prices: dict[str, PriceSnapshot],
        conversions: dict[str, float],
        *,
        account: dict[str, Any],
        positions: list[dict[str, Any]],
    ) -> FxPortfolioState:
        currency_exposures: dict[str, float] = {}
        pair_exposures: dict[str, float] = {}
        gross_exposure = 0.0
        active_instruments: set[str] = set()
        for position in positions:
            name = str(position.get("instrument") or "").upper()
            instrument = instruments.get(name)
            price = prices.get(name)
            if instrument is None or price is None:
                continue
            long_units = float((position.get("long") or {}).get("units") or 0.0)
            short_units = float((position.get("short") or {}).get("units") or 0.0)
            net_units = long_units + short_units
            if abs(net_units) <= 0:
                continue
            active_instruments.add(name)
            side = Side.LONG if net_units > 0 else Side.SHORT
            exposure = position_value_home(
                instrument,
                abs(net_units),
                price.mid,
                self.settings.risk.account_currency,
                conversions,
                price.quote_to_home_factor,
            )
            gross_exposure += exposure
            pair_exposures[name] = pair_exposures.get(name, 0.0) + exposure
            currency_exposures[instrument.base_currency] = currency_exposures.get(instrument.base_currency, 0.0) + exposure * side.sign
            currency_exposures[instrument.quote_currency] = currency_exposures.get(instrument.quote_currency, 0.0) - exposure * side.sign
            financing_estimate = estimated_daily_financing_home(
                instrument,
                side=side.value,
                units=abs(net_units),
                price=price.mid,
                account_currency=self.settings.risk.account_currency,
                timestamp=now,
                conversion_rates=conversions,
                snapshot_factor=price.quote_to_home_factor,
            )
            self.journal.record_position_snapshot(
                timestamp=now,
                instrument=name,
                side=side.value,
                units=abs(net_units),
                avg_price=float((position.get("long") or {}).get("averagePrice") or (position.get("short") or {}).get("averagePrice") or 0.0),
                unrealized_pl=float(position.get("unrealizedPL") or 0.0),
                margin_used=float(position.get("marginUsed") or 0.0),
                price=price.mid,
                payload={**position, "estimated_daily_financing": financing_estimate},
                estimated_daily_financing=financing_estimate,
            )
        self.journal.mark_current_positions_closed(active_instruments, now)
        equity = float(account.get("NAV") or account.get("balance") or 0.0)
        balance = float(account.get("balance") or equity)
        return FxPortfolioState(
            equity=equity,
            balance=balance,
            margin_used=float(account.get("marginUsed") or 0.0),
            open_positions=int(account.get("openPositionCount") or 0),
            account_currency=str(account.get("currency") or self.settings.risk.account_currency).upper(),
            portfolio_risk=self.journal.open_risk_amount(),
            gross_exposure=float(account.get("positionValue") or gross_exposure),
            pair_exposures=pair_exposures,
            currency_exposures=currency_exposures,
        )

    def _sync_open_trades(
        self,
        now: datetime,
        instruments: dict[str, FxInstrument],
        prices: dict[str, PriceSnapshot],
        conversions: dict[str, float],
    ) -> None:
        try:
            trades = self.client.open_trades()
        except Mt5Error as exc:
            self.journal.log_event("open_trade_sync_failed", str(exc), level="warning")
            return
        for trade in trades:
            trade_id = str(trade.get("id") or "")
            instrument_name = str(trade.get("instrument") or "").upper()
            if not trade_id or not instrument_name:
                continue
            units = _safe_float(trade.get("currentUnits") or trade.get("initialUnits"))
            side = Side.LONG if units >= 0 else Side.SHORT
            instrument = instruments.get(instrument_name)
            price = prices.get(instrument_name)
            financing_estimate = 0.0
            if instrument and price:
                financing_estimate = estimated_daily_financing_home(
                    instrument,
                    side=side.value,
                    units=abs(units),
                    price=price.mid,
                    account_currency=self.settings.risk.account_currency,
                    timestamp=now,
                    conversion_rates=conversions,
                    snapshot_factor=price.quote_to_home_factor,
                )
                self._maybe_move_stop_to_breakeven(trade, instrument, price)
            self.journal.upsert_trade(
                broker_trade_id=trade_id,
                instrument=instrument_name,
                side=side.value,
                units=abs(units),
                state="open",
                entry_time=_parse_broker_time(trade.get("openTime")),
                entry_price=_safe_float(trade.get("price")),
                realized_pl=_safe_float(trade.get("realizedPL")),
                financing=_safe_float(trade.get("financing")),
                payload={**trade, "estimated_daily_financing": financing_estimate},
            )

    def _sync_trade_history(self, now: datetime) -> None:
        state = self.journal.get_state()
        since = _parse_broker_time(state.last_transaction_id) if state.last_transaction_id else None
        if since is None:
            since = now - timedelta(days=1)
        try:
            closed_trades = self.client.closed_trades_since(since, now)
        except Mt5Error as exc:
            self.journal.log_event("trade_history_sync_failed", str(exc), level="warning")
            return
        for trade in closed_trades:
            trade_id = str(trade.get("broker_trade_id") or "")
            if not trade_id:
                continue
            self.journal.upsert_trade(
                broker_trade_id=trade_id,
                instrument=str(trade.get("instrument") or ""),
                side=str(trade.get("side") or ""),
                units=_safe_float(trade.get("units")),
                state="closed",
                exit_time=_parse_broker_time(trade.get("exit_time")),
                exit_price=_safe_float(trade.get("exit_price")),
                realized_pl=_safe_float(trade.get("realized_pl")),
                financing=_safe_float(trade.get("financing")),
                exit_reason=str(trade.get("exit_reason") or "mt5_history_deal"),
                payload=trade,
            )
            self.journal.mark_trade_orders_closed(trade_id)
        self.journal.set_last_transaction_id(now.isoformat())

    def _record_fill_trade(
        self,
        response: dict[str, Any],
        intent: FxSignalIntent,
        instrument: FxInstrument,
        units: float,
        broker_trade_id: str | None,
    ) -> None:
        fill = response.get("orderFillTransaction") or {}
        opened = fill.get("tradeOpened") or {}
        trade_id = broker_trade_id or str(opened.get("tradeID") or "")
        if not trade_id:
            return
        signed_units = _safe_float(opened.get("units") or fill.get("units") or units * intent.side.broker_units_sign)
        self.journal.upsert_trade(
            broker_trade_id=trade_id,
            instrument=instrument.name,
            side=(Side.LONG if signed_units >= 0 else Side.SHORT).value,
            units=abs(signed_units) or units,
            state="open",
            entry_time=_parse_broker_time(fill.get("time")) or intent.timestamp,
            entry_price=_safe_float(fill.get("price"), intent.entry_price),
            payload=response,
        )

    def _maybe_move_stop_to_breakeven(
        self,
        trade: dict[str, Any],
        instrument: FxInstrument,
        price: PriceSnapshot,
    ) -> None:
        trade_id = str(trade.get("id") or "")
        entry = _safe_float(trade.get("price"))
        units = _safe_float(trade.get("currentUnits") or trade.get("initialUnits"))
        stop_price = _nested_price(trade.get("stopLossOrder"))
        if not trade_id or entry <= 0 or units == 0 or stop_price is None:
            return
        side = Side.LONG if units > 0 else Side.SHORT
        current_exit = price.bid if side is Side.LONG else price.ask
        risk_distance = abs(entry - stop_price)
        profit_distance = (current_exit - entry) * side.sign
        if risk_distance <= 0 or profit_distance < risk_distance:
            return
        buffer = self.settings.strategy.breakeven_buffer_pips * instrument.pip_size
        new_stop = instrument.round_price(entry + buffer * side.sign)
        if side is Side.LONG and stop_price >= new_stop:
            return
        if side is Side.SHORT and stop_price <= new_stop:
            return
        take_profit = _nested_price(trade.get("takeProfitOrder"))
        try:
            response = self.client.set_trade_dependent_orders(
                trade_id=trade_id,
                instrument=instrument,
                stop_loss=new_stop,
                take_profit=take_profit,
            )
            self.journal.log_event(
                "breakeven_stop_updated",
                f"{instrument.name} trade {trade_id} stop moved to breakeven",
                payload={"new_stop": new_stop, "response": response},
            )
        except Mt5Error as exc:
            self.journal.log_event("breakeven_stop_failed", str(exc), level="warning", payload={"trade_id": trade_id})

    def _skip(self, timestamp: datetime, instrument: str, reason: str, payload: dict[str, Any] | None = None) -> None:
        self.journal.record_signal(
            timestamp=timestamp,
            instrument=instrument,
            status="skipped",
            reason=reason,
            payload=payload or {},
        )

    async def _publish(self, payload: dict[str, Any]) -> None:
        if self.publisher is None:
            return
        result = self.publisher(payload)
        if result is not None:
            await result


def client_order_id(intent: FxSignalIntent, leg: str) -> str:
    raw_timestamp = intent.timestamp.isoformat()
    key = f"{intent.instrument}:{intent.side.value}:{raw_timestamp}:{leg}:{intent.entry_price:.8f}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"fxft-{intent.instrument.replace('_', '')}-{leg}-{digest}"[:64]


def instrument_price_round(intent: FxSignalIntent, price: float) -> float:
    if intent.instrument.endswith("_JPY"):
        return round(price, 3)
    return round(price, 5)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_broker_time(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _nested_price(payload: Any) -> float | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("price")
    if value in (None, ""):
        return None
    return _safe_float(value)
