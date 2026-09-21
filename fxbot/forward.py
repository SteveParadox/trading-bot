"""Unattended MetaTrader 5 forward-test worker."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from uuid import uuid4
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

import pandas as pd

from fxbot.config import FxBotSettings, NewsEvent, ensure_runtime_dirs, settings_from_env
from fxbot.instruments import (
    FxInstrument,
    PriceSnapshot,
    estimated_daily_financing_home,
    position_value_home,
)
from fxbot.journal import StructuredJournal
from fxbot.market_hours import active_sessions, can_trade, news_blackout_reason
from fxbot.models import BotRunState, FxPortfolioState, FxSignalIntent, Side
from fxbot.mt5 import Mt5Client, Mt5CredentialsMissing, Mt5Error, Mt5RejectedError, extract_order_ids
from fxbot.operations import freshness
from fxbot.recovery import RecoveryState, reconcile_order
from fxbot.risk import FxRiskDecision, FxRiskManager, reward_covers_spread
from fxbot.strategy import (
    TIMEFRAME_DELTAS,
    build_signal_intent,
    evaluate_signal_frame,
    last_closed_row,
    prepare_indicators,
)
from fxbot.database import set_lock_retry_callback
from fxbot.monitoring import OperationalMonitor
from fxbot.news import NewsGateway, build_news_gateway
from fxbot.ai_deliberation import (
    AiAuditResponse,
    AiDeliberationResult,
    AiDeliberationService,
    apply_ai_execution_policy,
    build_signal_evidence,
    validate_ai_audit_response,
)
from fxbot.operations import clock_health
from fxbot.security import code_version, data_hash, experiment_manifest, strategy_config_hash

log = logging.getLogger(__name__)

# How far back (in hours) closed-trade history is always re-fetched for
# reconciliation, so trades that closed between scans are picked up even when
# the sync cursor has already advanced past their close deals. Kept larger than
# the broker-feed clock skew (~2.5h on MetaQuotes demo) plus the loop interval.
TRADE_HISTORY_RECONCILE_HOURS = 72

Publisher = Callable[[dict[str, Any]], Awaitable[None] | None]


class ForwardTestWorker:
    def __init__(
        self,
        settings: FxBotSettings | None = None,
        *,
        client: Mt5Client | None = None,
        journal: StructuredJournal | None = None,
        publisher: Publisher | None = None,
        monitor: OperationalMonitor | None = None,
        news: NewsGateway | None = None,
        deliberator: AiDeliberationService | None = None,
    ) -> None:
        self.settings = settings or settings_from_env()
        ensure_runtime_dirs(self.settings)
        self.client = client or Mt5Client(self.settings.broker)
        self.journal = journal or StructuredJournal(
            self.settings.runtime.database_url,
            self.settings.runtime.log_jsonl_path,
        )
        self.publisher = publisher
        self.monitor = monitor or OperationalMonitor(
            stale_price_threshold_seconds=self.settings.runtime.max_price_age_seconds,
            news_data_max_age_seconds=self.settings.strategy.news_data_max_age_seconds,
        )
        self.news = news or build_news_gateway(
            self.settings.strategy,
            database_url=self.settings.runtime.database_url,
            monitor=self.monitor,
            static_events=self.settings.news_events,
        )
        self.deliberator = deliberator or AiDeliberationService(self.settings.ai)
        # Feed SQLite lock retries into the operational monitor.
        monitor_ref = self.monitor
        set_lock_retry_callback(monitor_ref.record_db_lock_retry)
        self.risk = FxRiskManager(self.settings.risk, self.settings.strategy)
        self.strategy_hash = strategy_config_hash(self.settings)
        self.code_version = code_version()
        self.run_id = uuid4().hex
        self.journal.strategy_hash = self.strategy_hash
        self.journal.code_version = self.code_version
        self.journal.data_hash = self.journal.data_hash or data_hash({"source": "mt5", "run_id": self.run_id})
        manifest = experiment_manifest(
            strategy_hash=self.strategy_hash,
            code=self.code_version,
            data={"source": "mt5", "run_id": self.run_id},
            splits={"type": "forward_test"},
            parameters={"instruments": self.settings.instruments},
        )
        self.journal.experiment_manifest_hash = manifest["manifest_hash"]
        self.journal.record_run_manifest(
            run_id=self.run_id,
            strategy_hash=self.strategy_hash,
            code_version=self.code_version,
            data_hash=manifest["data_hash"],
            manifest_hash=manifest["manifest_hash"],
            manifest=manifest,
        )
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
                self.monitor.record_broker_reconnect()
            except Mt5CredentialsMissing as exc:
                self.journal.set_state(BotRunState.PAUSED, "missing_mt5_connection")
                self.journal.log_event("credentials_missing", str(exc), level="error")
                self.monitor.record_broker_disconnect(str(exc))
                await self._publish({"type": "worker_paused", "reason": "missing_mt5_connection"})
            except Mt5Error as exc:
                self.journal.set_state(BotRunState.PAUSED, "broker_disconnected")
                self.journal.log_event("broker_disconnected", str(exc), level="error")
                self.monitor.record_broker_disconnect(str(exc))
                await self._publish({"type": "worker_paused", "reason": "broker_disconnected"})
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
        self.deliberator.close()
        self.journal.close()

    def scan_once(self) -> None:
        now = datetime.now(timezone.utc)
        if self.journal.get_state().state != BotRunState.RUNNING.value:
            return
        if not self.settings.broker.demo_only and not self.settings.runtime.live_release_approved:
            self.journal.set_state(BotRunState.HALTED, "live_release_gate_required")
            self.journal.log_event(
                "live_release_gate_blocked",
                "Live trading is blocked until the audited release gate is approved",
                level="critical",
            )
            self.monitor.record_operator_review("live release gate not approved")
            return
        instruments = self._load_instruments()
        prices = self.client.pricing(self.settings.instruments)
        if not self._check_price_freshness(now, prices.prices):
            return
        # Record price freshness for the monitoring dashboard
        for name, price in prices.prices.items():
            self.monitor.record_price(name, price.time)
        self._reconcile_unknown_orders()
        self.monitor.record_scan_heartbeat()
        self.journal.log_event("scan_heartbeat", "broker scan completed", payload={"run_id": self.run_id})
        # Check broker-host clock health every scan
        try:
            broker_time = now  # MT5 server time is approximated from local clock
            ch = clock_health(now, broker_time, max_skew_seconds=300.0)
            self.monitor.record_clock_health(ch)
        except Exception:
            pass
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
        news_snapshot = self.news.ensure_current(now)
        self._protect_positions_for_news(now, instruments, prices.prices, news_snapshot.events)
        halt_reason = self.journal.update_protection_state(
            timestamp=now,
            equity=portfolio.equity,
            max_daily_loss_pct=self.settings.risk.max_daily_loss_pct,
            max_drawdown_pct=self.settings.risk.max_drawdown_pct,
        )
        if halt_reason:
            self.journal.log_event("risk_halt", halt_reason, level="warning")
            self.monitor.record_risk_halt(halt_reason)
            return
        else:
            self.monitor.clear_risk_halt()

        for name in self.settings.instruments:
            instrument = instruments.get(name)
            price = prices.prices.get(name)
            if instrument is None or price is None:
                self._skip(now, name, "instrument_or_price_unavailable")
                continue
            if portfolio.pair_exposures.get(name, 0.0) > 0:
                self._skip(now, name, "pair_position_open")
                continue
            permission = can_trade(
                name,
                now,
                account_state=portfolio,
                news_state=news_snapshot,
                settings=self.settings.strategy,
            )
            if not permission.allowed:
                self._skip(now, name, permission.reason, permission.as_dict())
                continue
            if price.spread_pips(instrument) > self.settings.strategy.max_spread_pips:
                self._skip(now, name, "spread_filter", {"spread_pips": price.spread_pips(instrument)})
                continue
            self._scan_instrument(now, instrument, price, portfolio, prices.conversion_rates, news_snapshot)

    def _check_price_freshness(self, now: datetime, prices: dict[str, PriceSnapshot]) -> bool:
        stale = []
        for name, price in prices.items():
            ok, age = freshness(now, price.time, max_age_seconds=self.settings.runtime.max_price_age_seconds)
            if not ok:
                item = {"instrument": name, "age_seconds": age}
                stale.append(item)
                self.journal.log_event("stale_price", f"stale price for {name}", level="warning", payload=item)
                self.monitor.record_price(name, price.time)
            else:
                self.monitor.record_price(name, price.time)
        return not stale

    def _reconcile_unknown_orders(self) -> None:
        for row in self.journal.recovery_orders():
            # An already-recorded definitive rejection means the order was never
            # placed: promote it out of the 'unknown' recovery set instead of
            # re-running the broker lookup (and re-alerting) every scan.
            if _is_definite_rejection_error(row.error):
                self.journal.update_order(row.client_order_id, status=RecoveryState.REJECTED.value, error=row.error)
                self.monitor.clear_unknown_order(row.client_order_id)
                self.journal.log_event(
                    "order_rejected",
                    f"order {row.client_order_id} was rejected by the broker and is not recoverable",
                    level="warning",
                    payload={"status": row.status},
                )
                continue
            result = reconcile_order(row, self.client.order_by_client_id)
            if result.after == RecoveryState.UNKNOWN:
                self.journal.log_event(
                    "unknown_order",
                    f"order {row.client_order_id} remains unresolved",
                    level="critical",
                    payload={"status": row.status},
                )
                self.monitor.record_unknown_order(row.client_order_id)
                continue
            broker = self.client.order_by_client_id(row.client_order_id)
            self.journal.update_order(
                row.client_order_id,
                status=result.after.value,
                broker_order_id=str(broker.get("id")) if broker else None,
                response=broker,
            )
            self.monitor.clear_unknown_order(row.client_order_id)
            self.journal.log_event("order_reconciled", f"order {row.client_order_id} reconciled", payload={"state": result.after.value})

    def _scan_instrument(
        self,
        now: datetime,
        instrument: FxInstrument,
        price: PriceSnapshot,
        portfolio: FxPortfolioState,
        conversion_rates: dict[str, float],
        news_snapshot: Any,
    ) -> None:
        entry_frame = prepare_indicators(
            self.client.candles(instrument.name, self.settings.strategy.entry_timeframe, self.settings.strategy.candle_limit)
        )
        htf_frame = prepare_indicators(
            self.client.candles(instrument.name, self.settings.strategy.htf_timeframe, self.settings.strategy.candle_limit)
        )
        decision_time = feed_decision_time(
            entry_frame,
            self.settings.strategy.entry_timeframe,
            fallback=now,
        )
        decision = evaluate_signal_frame(
            entry_frame,
            htf_frame,
            instrument=instrument,
            settings=self.settings.strategy,
            timestamp=decision_time,
        )
        if decision.signal is None:
            self._skip(now, instrument.name, decision.reason, decision.details)
            return

        signal_row = last_closed_row(
            entry_frame,
            self.settings.strategy.entry_timeframe,
            timestamp=decision_time,
        )
        if signal_row is None:
            self._skip(now, instrument.name, "closed_signal_candle_unavailable")
            return
        last_close = float(signal_row["close"])
        atr_price = float(signal_row.get("atr") or 0.0)
        spread_price = price.ask - price.bid
        if atr_price <= 0 or spread_price <= 0:
            self._skip(now, instrument.name, "volatility_or_spread_unavailable")
            return
        spread_atr_ratio = spread_price / atr_price
        if spread_atr_ratio > self.settings.strategy.max_spread_atr_ratio:
            self._skip(
                now,
                instrument.name,
                "spread_to_atr_filter",
                {"spread_atr_ratio": spread_atr_ratio, "max": self.settings.strategy.max_spread_atr_ratio},
            )
            return
        executable_entry = executable_entry_price(price, decision.signal)
        deviation_pips = abs(executable_entry - last_close) / instrument.pip_size
        if deviation_pips > self.settings.strategy.max_entry_deviation_pips:
            self._skip(now, instrument.name, "entry_deviation_filter", {"deviation_pips": deviation_pips})
            return

        intent = build_signal_intent(
            entry_frame,
            htf_frame,
            instrument=instrument,
            settings=self.settings.strategy,
            entry_price=executable_entry,
            timestamp=decision_time,
            entry_price_source="broker_executable_bid_ask",
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
                entry_price=executable_entry,
                payload={"risk": asdict(risk), "intent": asdict(intent)},
                data_hash=data_hash({"instrument": instrument.name, "decision_time": decision_time.isoformat(), "signal": intent.signal_row}),
            )
            return

        if not reward_covers_spread(
            risk.exit_plan,
            spread_price,
            self.settings.strategy.min_reward_to_spread_ratio,
        ):
            ratio = risk.exit_plan.reward_distance / spread_price if spread_price > 0 else 0.0
            self.journal.record_signal(
                timestamp=now,
                instrument=instrument.name,
                status="rejected",
                reason="target_cost_filter",
                side=intent.side.value,
                score=intent.score,
                entry_price=executable_entry,
                stop_loss=risk.exit_plan.stop_loss,
                take_profit=risk.exit_plan.take_profit,
                risk_amount=risk.risk_amount,
                payload={
                    "risk": asdict(risk),
                    "intent": asdict(intent),
                    "reward_to_spread_ratio": ratio,
                    "required_reward_to_spread_ratio": self.settings.strategy.min_reward_to_spread_ratio,
                },
                data_hash=data_hash({"instrument": instrument.name, "decision_time": decision_time.isoformat(), "signal": intent.signal_row}),
            )
            return

        signal_row = self.journal.record_signal(
            timestamp=now,
            instrument=instrument.name,
            status="accepted",
            reason="signal_and_risk_accepted",
            side=intent.side.value,
            score=intent.score,
            entry_price=executable_entry,
            stop_loss=risk.exit_plan.stop_loss,
            take_profit=risk.exit_plan.take_profit,
            risk_amount=risk.risk_amount,
            payload={"risk": asdict(risk), "intent": asdict(intent)},
            data_hash=data_hash({"instrument": instrument.name, "decision_time": decision_time.isoformat(), "signal": intent.signal_row}),
        )
        # This deterministic key identifies the parent setup across retries and
        # its partial exit legs. It is intentionally independent of SQLite's
        # surrogate signal-row id so a restarted scan cannot create another AI
        # deliberation for the same closed candle.
        parent_signal_id = parent_signal_id_for(intent)
        ai_result: AiDeliberationResult | None = None
        ai_row = None
        if self.settings.ai.mode != "off":
            ai_row = self.journal.find_ai_deliberation(parent_signal_id)
            if ai_row is not None:
                ai_result = _ai_result_from_row(ai_row)
            else:
                evidence = build_signal_evidence(
                    signal_id=parent_signal_id,
                    intent=intent,
                    instrument=instrument,
                    price=price,
                    portfolio=portfolio,
                    risk=risk,
                    strategy=self.settings.strategy,
                    news_events=news_snapshot.events,
                    news_stale=news_snapshot.stale,
                    active_sessions=active_sessions(intent.timestamp),
                    now=now,
                    demo_only=self.settings.broker.demo_only,
                )
                ai_result = self.deliberator.deliberate(evidence)
                try:
                    ai_row, _ = self.journal.record_ai_deliberation(
                        payload=_ai_journal_payload(
                            signal_id=parent_signal_id,
                            timestamp=now,
                            instrument=instrument.name,
                            side=intent.side.value,
                            settings=self.settings.ai,
                            evidence=evidence.to_dict(),
                            evidence_hash=evidence.evidence_hash(),
                            result=ai_result,
                        )
                    )
                except Exception as exc:
                    # An optional audit persistence issue must not stop the
                    # deterministic shadow path. Advisory confirmation treats
                    # it as a failed AI result below, never as an accidental
                    # allow. Core signal/order journaling has already occurred.
                    log.exception("AI deliberation persistence failed for %s", parent_signal_id)
                    ai_result = AiDeliberationResult(
                        response=None,
                        latency_ms=ai_result.latency_ms,
                        failure_reason=f"ai_persistence_failed:{type(exc).__name__}",
                    )
                response = ai_result.response
                log.info(
                    "ai_deliberation signal_id=%s pair=%s direction=%s score=%.2f mode=%s decision=%s confidence=%s reasoning=%s context=%s latency_ms=%s failure=%s",
                    parent_signal_id, instrument.name, intent.side.value, intent.score, self.settings.ai.mode,
                    response.decision if response else None, response.confidence if response else None,
                    response.reasoning_audit["status"] if response else None,
                    response.market_context["status"] if response else None,
                    ai_result.latency_ms, ai_result.failure_reason,
                )
        policy = apply_ai_execution_policy(
            hard_safety_allowed=risk.allowed and risk.exit_plan is not None,
            settings=self.settings.ai,
            result=ai_result,
        )
        if self.settings.ai.mode != "off":
            self.journal.update_signal(
                signal_row.id,
                status="accepted" if policy.allowed else "advisory_blocked",
                reason="signal_and_risk_accepted" if policy.allowed else policy.reason,
                payload_update={
                    "parent_signal_id": parent_signal_id,
                    "ai_deliberation_id": ai_row.id if ai_row is not None else None,
                    "ai_execution_policy": asdict(policy),
                },
            )
            if not policy.allowed:
                return
            intent = replace(
                intent,
                metadata={
                    **intent.metadata,
                    "parent_signal_id": parent_signal_id,
                    "parent_signal_row_id": signal_row.id,
                    "ai_deliberation_id": ai_row.id if ai_row is not None else None,
                },
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
                "signal_score": intent.score,
                "signal_time": intent.timestamp.isoformat(),
                "strategy_decision": intent.metadata.get("decision"),
                "signal_features": intent.metadata.get("score_details", {}),
                "risk_metadata": risk.metadata,
                "parent_signal_id": intent.metadata.get("parent_signal_id"),
                "parent_signal_row_id": intent.metadata.get("parent_signal_row_id"),
                "ai_deliberation_id": intent.metadata.get("ai_deliberation_id"),
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
                strategy_hash=self.strategy_hash,
                code_version=self.code_version,
                data_hash=data_hash(payload),
                experiment_manifest_hash=self.journal.experiment_manifest_hash,
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
                # A definitive broker rejection (retcode) means the order was
                # NOT placed and never can be reconciled, so mark it rejected
                # instead of leaving it in the recovery loop forever. A timeout
                # / connection fault (plain Mt5Error) stays "unknown" and is
                # surfaced through the unknown-order monitor because the order
                # may still exist at the broker.
                status = "rejected" if isinstance(exc, Mt5RejectedError) else "unknown"
                self.journal.update_order(client_id, status=status, error=str(exc))
                if status == "unknown":
                    self.monitor.record_unknown_order(client_id)
                raise

    def _order_legs(
        self,
        intent: FxSignalIntent,
        risk: FxRiskDecision,
        instrument: FxInstrument,
    ) -> list[tuple[str, float, float | None]]:
        if risk.exit_plan is None or not self.settings.strategy.partial_tp_enabled:
            return [("full", risk.units, risk.exit_plan.take_profit if risk.exit_plan else intent.entry_price)]
        tp1_units = instrument.round_units(risk.units * self.settings.strategy.tp1_units_pct)
        tp2_units = instrument.round_units(risk.units - tp1_units)
        if tp1_units <= 0 or tp2_units <= 0:
            return [("full", risk.units, risk.exit_plan.take_profit)]
        runner_r = self.settings.strategy.runner_take_profit_r
        tp2 = None
        if runner_r is not None:
            tp2_distance = risk.exit_plan.risk_distance * runner_r
            tp2 = instrument_price_round(
                intent,
                intent.entry_price + tp2_distance
                if intent.side is Side.LONG
                else intent.entry_price - tp2_distance,
            )
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
                self._maybe_update_trailing_stop(trade, instrument, price)
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
            self.journal.record_external_order(
                broker_order_id=trade_id,
                broker_trade_id=trade_id,
                timestamp=_parse_broker_time(trade.get("openTime")) or now,
                instrument=instrument_name,
                side=side.value,
                units=abs(units),
                payload={**trade, "source": "mt5_reconciliation"},
            )

    def _sync_trade_history(self, now: datetime) -> None:
        state = self.journal.get_state()
        cursor = _parse_broker_time(state.last_transaction_id) if state.last_transaction_id else None
        floor = now - timedelta(hours=TRADE_HISTORY_RECONCILE_HOURS)
        # Use the earlier of the sync cursor and a fixed reconcile floor so that
        # trades which closed between scans (whose close deals fall before the
        # cursor) are still re-fetched and flipped to `closed`. The upsert is
        # keyed on broker_trade_id, so re-processing is idempotent.
        since = min(cursor, floor) if cursor else floor
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
            payload={
                **response,
                "strategy_context": {
                    "signal_score": intent.score,
                    "signal_time": intent.timestamp.isoformat(),
                    "strategy_decision": intent.metadata.get("decision"),
                    "signal_features": intent.metadata.get("score_details", {}),
                    "entry_price_source": intent.metadata.get("entry_price_source"),
                    "parent_signal_id": intent.metadata.get("parent_signal_id"),
                    "parent_signal_row_id": intent.metadata.get("parent_signal_row_id"),
                    "ai_deliberation_id": intent.metadata.get("ai_deliberation_id"),
                },
            },
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

    def _maybe_update_trailing_stop(
        self,
        trade: dict[str, Any],
        instrument: FxInstrument,
        price: PriceSnapshot,
    ) -> None:
        """Ratchet a profitable trade's stop using the latest closed-candle ATR."""
        trade_id = str(trade.get("id") or "")
        entry = _safe_float(trade.get("price"))
        units = _safe_float(trade.get("currentUnits") or trade.get("initialUnits"))
        current_stop = _nested_price(trade.get("stopLossOrder"))
        if not trade_id or entry <= 0 or units == 0 or current_stop is None:
            return
        side = Side.LONG if units > 0 else Side.SHORT
        current_exit = price.bid if side is Side.LONG else price.ask
        risk_distance = abs(entry - current_stop)
        profit_distance = (current_exit - entry) * side.sign
        if risk_distance <= 0 or profit_distance < risk_distance:
            return
        try:
            frame = prepare_indicators(
                self.client.candles(
                    instrument.name,
                    self.settings.strategy.entry_timeframe,
                    self.settings.strategy.candle_limit,
                )
            )
            row = last_closed_row(frame, self.settings.strategy.entry_timeframe)
            atr = _safe_float(row.get("atr")) if row is not None else 0.0
        except (Mt5Error, ValueError, KeyError) as exc:
            self.journal.log_event(
                "trailing_stop_unavailable",
                str(exc),
                level="warning",
                payload={"trade_id": trade_id},
            )
            return
        if atr <= 0:
            return

        distance = atr * self.settings.strategy.trailing_atr_multiplier
        candidate = current_exit - distance if side is Side.LONG else current_exit + distance
        if instrument.minimum_stop_distance:
            if side is Side.LONG:
                candidate = min(candidate, current_exit - instrument.minimum_stop_distance)
            else:
                candidate = max(candidate, current_exit + instrument.minimum_stop_distance)
        candidate = instrument.round_price(candidate)
        minimum_move = instrument.pip_size
        if side is Side.LONG and candidate <= current_stop + minimum_move:
            return
        if side is Side.SHORT and candidate >= current_stop - minimum_move:
            return
        try:
            self.client.set_trade_dependent_orders(
                trade_id=trade_id,
                instrument=instrument,
                stop_loss=candidate,
                take_profit=_nested_price(trade.get("takeProfitOrder")),
            )
            self.journal.log_event(
                "trailing_stop_updated",
                f"{instrument.name} trade {trade_id} trailing stop ratcheted",
                payload={"old_stop": current_stop, "new_stop": candidate, "atr": atr},
            )
        except Mt5Error as exc:
            self.journal.log_event(
                "trailing_stop_update_failed",
                str(exc),
                level="warning",
                payload={"trade_id": trade_id},
            )

    def _protect_positions_for_news(
        self,
        now: datetime,
        instruments: dict[str, FxInstrument],
        prices: dict[str, PriceSnapshot],
        events: list[NewsEvent],
    ) -> None:
        """Apply the configured news risk policy to existing open positions.

        ``news_risk_action`` controls what happens to open exposure before a
        high-impact event:
          - ``block_entries``     HOLD positions untouched (entries are already
                                  blocked by the blackout window).
          - ``protect_and_block`` REDUCE risk by tightening stops to breakeven
                                  on profitable trades.
          - ``close_positions``   CLOSE exposed positions before the event.
        """
        action = self.settings.strategy.news_risk_action
        if action == "block_entries":
            return
        if not events:
            return
        try:
            trades = self.client.open_trades()
        except Mt5Error as exc:
            self.journal.log_event("news_position_protection_unavailable", str(exc), level="warning")
            return
        for trade in trades:
            name = str(trade.get("instrument") or "").upper()
            instrument = instruments.get(name)
            price = prices.get(name)
            if instrument is None or price is None:
                continue
            reason = news_blackout_reason(
                name,
                events,
                now,
                self.settings.strategy.news_blackout_before_minutes,
                self.settings.strategy.news_blackout_after_minutes,
                impact_score_min=self.settings.strategy.news_blackout_impact_score_min,
            )
            if not reason:
                continue
            trade_id = str(trade.get("id") or "")
            entry = _safe_float(trade.get("price"))
            units = _safe_float(trade.get("currentUnits") or trade.get("initialUnits"))
            if not trade_id or entry <= 0 or units == 0:
                continue
            if action == "close_positions":
                self._close_position_for_news(trade_id, name, instrument, units, reason)
                continue
            # protect_and_block: tighten profitable trades to breakeven.
            side = Side.LONG if units > 0 else Side.SHORT
            current_stop = _nested_price(trade.get("stopLossOrder"))
            current_exit = price.bid if side is Side.LONG else price.ask
            if current_stop is None:
                continue
            if (current_exit - entry) * side.sign <= 0:
                continue
            buffer = self.settings.strategy.breakeven_buffer_pips * instrument.pip_size
            candidate = instrument.round_price(entry + buffer * side.sign)
            if side is Side.LONG and current_stop >= candidate:
                continue
            if side is Side.SHORT and current_stop <= candidate:
                continue
            try:
                self.client.set_trade_dependent_orders(
                    trade_id=trade_id,
                    instrument=instrument,
                    stop_loss=candidate,
                    take_profit=_nested_price(trade.get("takeProfitOrder")),
                )
                self.journal.log_event(
                    "news_position_protected",
                    f"{name} trade {trade_id} stop tightened before high-impact news",
                    payload={"reason": reason, "new_stop": candidate},
                )
            except Mt5Error as exc:
                self.journal.log_event(
                    "news_position_protection_failed",
                    str(exc),
                    level="warning",
                    payload={"trade_id": trade_id, "reason": reason},
                )

    def _close_position_for_news(
        self,
        trade_id: str,
        name: str,
        instrument: FxInstrument,
        signed_units: float,
        reason: str,
    ) -> None:
        try:
            self.client.close_position(
                trade_id=trade_id,
                instrument=instrument,
                signed_units=signed_units,
                comment="fxft-news-close",
            )
            self.journal.log_event(
                "news_position_closed",
                f"{name} trade {trade_id} closed before high-impact news",
                payload={"reason": reason, "units": signed_units},
            )
        except Mt5Error as exc:
            self.journal.log_event(
                "news_position_close_failed",
                str(exc),
                level="warning",
                payload={"trade_id": trade_id, "reason": reason},
            )

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


def feed_decision_time(
    entry_frame: Any,
    timeframe: str,
    *,
    fallback: datetime,
) -> datetime:
    """Return the strategy decision time anchored to the broker data feed.

    MT5 candle timestamps use the broker server's simulated clock, which on the
    MetaQuotes demo terminal runs ~2.5h ahead of the host system clock. Passing
    the raw system ``datetime.now(UTC)`` into evaluate_signal_frame /
    last_closed_row therefore misclassifies every candle as
    ``future_candle_in_frame`` and blocks signal generation.

    Because the worker fetches candles with ``pos=1``, the last row is the most
    recent fully-closed candle, so its open time plus one timeframe is exactly
    the moment that candle completed -- a correct, feed-consistent "now" that is
    never in the future of the data and is identical to the system clock on
    non-skewed feeds.
    """
    if entry_frame is None or len(entry_frame) == 0:
        return fallback
    try:
        delta = TIMEFRAME_DELTAS[timeframe]
    except KeyError:
        return fallback
    index = entry_frame.index
    if not hasattr(index, "__len__") or len(index) == 0:
        return fallback
    last_open = index[-1]
    if isinstance(last_open, pd.Timestamp):
        last_open = last_open.to_pydatetime()
    closed_at = last_open + delta
    if closed_at.tzinfo is None:
        closed_at = closed_at.replace(tzinfo=timezone.utc)
    return closed_at.astimezone(timezone.utc)


def executable_entry_price(price: PriceSnapshot, side: Side) -> float:
    """Use the price available for a market fill, never an optimistic mid.

    Longs execute at ask and shorts at bid. Risk sizing, stop distance, reward
    distance, journaled expected price, and entry-deviation checks must all use
    that same side-aware price to avoid understating live trading friction.
    """

    return price.ask if side is Side.LONG else price.bid


def client_order_id(intent: FxSignalIntent, leg: str) -> str:
    raw_timestamp = intent.timestamp.isoformat()
    key = f"{intent.instrument}:{intent.side.value}:{raw_timestamp}:{leg}:{intent.entry_price:.8f}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"fxft-{intent.instrument.replace('_', '')}-{leg}-{digest}"[:64]


def parent_signal_id_for(intent: FxSignalIntent) -> str:
    """Stable idempotency key for a parent signal, not an order leg."""
    timestamp = intent.timestamp.replace(tzinfo=timezone.utc) if intent.timestamp.tzinfo is None else intent.timestamp.astimezone(timezone.utc)
    raw_timestamp = timestamp.isoformat()
    # The closed-candle decision, not a changing live bid/ask, defines the
    # parent signal. Price remains in evidence and order idempotency but must
    # not create duplicate audits during repeated scans of one signal candle.
    key = f"{intent.instrument}:{intent.side.value}:{raw_timestamp}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return f"fxsig-{intent.instrument.replace('_', '')}-{digest}"[:64]


def _ai_journal_payload(
    *,
    signal_id: str,
    timestamp: datetime,
    instrument: str,
    side: str,
    settings: Any,
    evidence: dict[str, Any],
    evidence_hash: str,
    result: AiDeliberationResult,
) -> dict[str, Any]:
    response = result.response.to_dict() if result.response else None
    reasoning = (response or {}).get("reasoning_audit", {})
    context = (response or {}).get("market_context", {})
    output_hash = data_hash(response) if response is not None else None
    return {
        "signal_id": signal_id,
        "timestamp": timestamp,
        "instrument": instrument,
        "side": side,
        "model": settings.model,
        "prompt_version": settings.prompt_version,
        "mode": settings.mode,
        "status": "completed" if response else "failed",
        "decision": response.get("decision") if response else None,
        "confidence": response.get("confidence") if response else None,
        "reasoning_audit_status": reasoning.get("status"),
        "reasoning_issues": reasoning.get("issues", []),
        "reasoning_supporting_factors": reasoning.get("supporting_factors", []),
        "market_context_status": context.get("status"),
        "market_context_issues": context.get("issues", []),
        "market_context_supporting_factors": context.get("supporting_factors", []),
        "contradictions": response.get("contradictions", []) if response else [],
        "recommended_action": response.get("recommended_action") if response else None,
        "summary": response.get("summary", "") if response else "",
        "evidence_hash": evidence_hash,
        "output_hash": output_hash,
        "latency_ms": result.latency_ms,
        "failure_reason": result.failure_reason,
        "evidence": evidence,
        "response": response,
    }


def _ai_result_from_row(row: Any) -> AiDeliberationResult:
    response = row.response
    if not isinstance(response, dict):
        return AiDeliberationResult(None, int(row.latency_ms or 0), row.failure_reason or "previous_ai_failure")
    try:
        audit = validate_ai_audit_response(response)
    except Exception:
        return AiDeliberationResult(None, int(row.latency_ms or 0), "stored_ai_response_invalid")
    return AiDeliberationResult(audit, int(row.latency_ms or 0), row.failure_reason)


def _is_definite_rejection_error(error: str | None) -> bool:
    """True for a stored Mt5RejectedError-style message: a broker retcode (e.g.
    10027/10030) or a local request-validation failure (last_error ``-2`` /
    ``Invalid ... argument``) means the order was definitively declined and can
    never reconcile. Timeouts / connection faults are NOT classified this way."""
    if not error:
        return False
    if "failed with retcode" in error:
        return True
    # MT5/RES_S_PARAMS returns a negative last_error code for invalid request
    # parameters — the order was rejected locally before submission.
    if "Invalid" in error and "argument" in error:
        return True
    return False


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
