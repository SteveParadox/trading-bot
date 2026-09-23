"""Structured SQL/JSONL journal for unattended forward testing."""

from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, close_all_sessions, sessionmaker

from fxbot.database import (
    AiDeliberationRow,
    BotStateRow,
    CurrentPositionRow,
    EquitySnapshotRow,
    EventLogRow,
    OrderJournalRow,
    PositionSnapshotRow,
    RunManifestRow,
    SignalJournalRow,
    TradeJournalRow,
    session_factory,
    utc_now,
)
from fxbot.models import BotRunState
from fxbot.security import redact


class StructuredJournal:
    def __init__(
        self,
        database_url: str,
        jsonl_path: str | None = None,
        *,
        strategy_hash: str | None = None,
        code_version: str | None = None,
        data_hash: str | None = None,
        experiment_manifest_hash: str | None = None,
    ) -> None:
        self.sessions = session_factory(database_url)
        self._engine = self.sessions.kw.get("bind")
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self.strategy_hash = strategy_hash
        self.code_version = code_version
        self.data_hash = data_hash
        self.experiment_manifest_hash = experiment_manifest_hash
        self._jsonl_lock = threading.Lock()
        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_state()

    def close(self) -> None:
        close_all_sessions()
        if self._engine is not None:
            self._engine.dispose()

    def _ensure_state(self) -> None:
        with self.sessions.begin() as session:
            row = session.get(BotStateRow, 1)
            if row is None:
                session.add(BotStateRow(id=1, state=BotRunState.STOPPED.value, updated_at=utc_now()))

    def get_state(self) -> BotStateRow:
        with self.sessions() as session:
            row = session.get(BotStateRow, 1)
            if row is None:
                return BotStateRow(id=1, state=BotRunState.STOPPED.value, updated_at=utc_now())
            session.expunge(row)
            return row

    def set_state(self, state: BotRunState | str, reason: str = "") -> BotStateRow:
        value = state.value if isinstance(state, BotRunState) else str(state)
        with self.sessions.begin() as session:
            row = session.get(BotStateRow, 1)
            if row is None:
                row = BotStateRow(id=1)
                session.add(row)
            row.state = value
            row.reason = reason
            row.updated_at = utc_now()
            session.flush()
            session.expunge(row)
        self.write_jsonl("bot_state", {"state": value, "reason": reason})
        return row

    def record_signal(
        self,
        *,
        timestamp: datetime,
        instrument: str,
        status: str,
        reason: str,
        side: str | None = None,
        score: float = 0.0,
        entry_price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        risk_amount: float | None = None,
        payload: dict[str, Any] | None = None,
        strategy_hash: str | None = None,
        code_version: str | None = None,
        data_hash: str | None = None,
        experiment_manifest_hash: str | None = None,
    ) -> SignalJournalRow:
        row = SignalJournalRow(
            timestamp=_aware(timestamp),
            instrument=instrument.upper(),
            side=side,
            status=status,
            reason=reason,
            score=score,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_amount=risk_amount,
            payload=_jsonable(payload or {}),
            strategy_hash=strategy_hash or self.strategy_hash,
            code_version=code_version or self.code_version,
            data_hash=data_hash or self.data_hash,
            experiment_manifest_hash=experiment_manifest_hash or self.experiment_manifest_hash,
        )
        with self.sessions.begin() as session:
            session.add(row)
            session.flush()
            session.expunge(row)
        self.write_jsonl("signal", row)
        return row

    def update_signal(
        self,
        signal_id: int,
        *,
        status: str | None = None,
        reason: str | None = None,
        payload_update: dict[str, Any] | None = None,
    ) -> SignalJournalRow | None:
        """Annotate a parent signal without creating a duplicate signal row."""
        with self.sessions.begin() as session:
            row = session.get(SignalJournalRow, signal_id)
            if row is None:
                return None
            if status is not None:
                row.status = status
            if reason is not None:
                row.reason = reason
            if payload_update:
                row.payload = _jsonable({**(row.payload or {}), **payload_update})
            session.flush()
            session.expunge(row)
        self.write_jsonl("signal_updated", row)
        return row

    def record_ai_deliberation(self, *, payload: dict[str, Any]) -> tuple[AiDeliberationRow, bool]:
        """Persist a validated audit or failure idempotently by parent signal."""
        signal_id = str(payload["signal_id"])
        try:
            with self.sessions.begin() as session:
                existing = session.scalar(select(AiDeliberationRow).where(AiDeliberationRow.signal_id == signal_id))
                if existing is not None:
                    session.expunge(existing)
                    return existing, False
                # Keep SQLAlchemy DateTime values typed; only JSON columns need
                # recursive conversion for dataclasses/enums.
                normalized = dict(payload)
                normalized["evidence"] = _jsonable(normalized.get("evidence") or {})
                normalized["response"] = _jsonable(normalized.get("response")) if normalized.get("response") is not None else None
                for name in (
                    "reasoning_issues", "reasoning_supporting_factors", "market_context_issues",
                    "market_context_supporting_factors", "contradictions",
                ):
                    normalized[name] = _jsonable(normalized.get(name) or [])
                row = AiDeliberationRow(**normalized)
                session.add(row)
                session.flush()
                session.expunge(row)
        except IntegrityError:
            # A second worker can observe no row before the unique insert from
            # the first worker commits. The uniqueness constraint is the final
            # idempotency authority; return the winner instead of surfacing a
            # harmless duplicate-race failure to the scan loop.
            existing = self.find_ai_deliberation(signal_id)
            if existing is not None:
                return existing, False
            raise
        self.write_jsonl("ai_deliberation", row)
        return row, True

    def find_ai_deliberation(self, signal_id: str) -> AiDeliberationRow | None:
        with self.sessions() as session:
            row = session.scalar(select(AiDeliberationRow).where(AiDeliberationRow.signal_id == signal_id))
            if row is not None:
                session.expunge(row)
            return row

    def recent_ai_deliberations(self, limit: int = 200) -> list[AiDeliberationRow]:
        return _recent(self.sessions, AiDeliberationRow, limit)

    def reserve_order(
        self,
        *,
        client_order_id: str,
        timestamp: datetime,
        instrument: str,
        side: str,
        units: float,
        order_type: str,
        risk_amount: float,
        payload: dict[str, Any],
        strategy_hash: str | None = None,
        code_version: str | None = None,
        data_hash: str | None = None,
        experiment_manifest_hash: str | None = None,
    ) -> tuple[OrderJournalRow, bool]:
        existing = self.find_order(client_order_id)
        if existing is not None:
            return existing, False
        row = OrderJournalRow(
            client_order_id=client_order_id,
            timestamp=_aware(timestamp),
            instrument=instrument.upper(),
            side=side,
            units=units,
            order_type=order_type,
            status="pending",
            risk_amount=risk_amount,
            payload=_jsonable(payload),
            strategy_hash=strategy_hash or self.strategy_hash,
            code_version=code_version or self.code_version,
            data_hash=data_hash or self.data_hash,
            experiment_manifest_hash=experiment_manifest_hash or self.experiment_manifest_hash,
        )
        with self.sessions.begin() as session:
            session.add(row)
            session.flush()
            session.expunge(row)
        self.write_jsonl("order_reserved", row)
        return row, True

    def update_order(
        self,
        client_order_id: str,
        *,
        status: str,
        broker_order_id: str | None = None,
        broker_trade_id: str | None = None,
        response: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> OrderJournalRow | None:
        with self.sessions.begin() as session:
            row = session.scalar(
                select(OrderJournalRow).where(OrderJournalRow.client_order_id == client_order_id)
            )
            if row is None:
                return None
            row.status = status
            row.broker_order_id = broker_order_id or row.broker_order_id
            row.broker_trade_id = broker_trade_id or row.broker_trade_id
            row.response = _jsonable(response) if response is not None else row.response
            row.error = error
            session.flush()
            session.expunge(row)
        self.write_jsonl("order_updated", row)
        return row

    def record_external_order(
        self,
        *,
        broker_order_id: str,
        broker_trade_id: str,
        timestamp: datetime,
        instrument: str,
        side: str,
        units: float,
        payload: dict[str, Any],
        strategy_hash: str | None = None,
        code_version: str | None = None,
        data_hash: str | None = None,
        experiment_manifest_hash: str | None = None,
    ) -> OrderJournalRow:
        client_order_id = f"mt5-external-{broker_trade_id}"
        with self.sessions.begin() as session:
            row = session.scalar(
                select(OrderJournalRow).where(OrderJournalRow.broker_trade_id == broker_trade_id)
            )
            if row is None:
                row = OrderJournalRow(
                    client_order_id=client_order_id,
                    timestamp=_aware(timestamp),
                    instrument=instrument.upper(),
                    side=side,
                    units=units,
                    order_type="MARKET",
                    status="filled",
                    broker_order_id=broker_order_id or None,
                    broker_trade_id=broker_trade_id,
                    risk_amount=0.0,
                    payload=_jsonable(payload),
                    response=_jsonable(payload),
                    strategy_hash=strategy_hash or self.strategy_hash,
                    code_version=code_version or self.code_version,
                    data_hash=data_hash or self.data_hash,
                    experiment_manifest_hash=experiment_manifest_hash or self.experiment_manifest_hash,
                )
                session.add(row)
            else:
                row.status = "filled"
                row.broker_order_id = broker_order_id or row.broker_order_id
                row.payload = _jsonable(payload)
                row.response = _jsonable(payload)
            session.flush()
            session.expunge(row)
        self.write_jsonl("external_order", row)
        return row

    def mark_trade_orders_closed(self, broker_trade_id: str) -> None:
        if not broker_trade_id:
            return
        with self.sessions.begin() as session:
            rows = list(
                session.scalars(
                    select(OrderJournalRow).where(OrderJournalRow.broker_trade_id == str(broker_trade_id))
                )
            )
            for row in rows:
                if row.status in {"pending", "submitted", "filled", "unknown"}:
                    row.status = "closed"
                    row.error = None
                    self.write_jsonl("order_updated", row)

    def find_order(self, client_order_id: str) -> OrderJournalRow | None:
        with self.sessions() as session:
            row = session.scalar(
                select(OrderJournalRow).where(OrderJournalRow.client_order_id == client_order_id)
            )
            if row is not None:
                session.expunge(row)
            return row

    def record_equity(self, *, timestamp: datetime, payload: dict[str, Any]) -> EquitySnapshotRow:
        row = EquitySnapshotRow(
            timestamp=_aware(timestamp),
            equity=float(payload.get("equity") or payload.get("NAV") or 0.0),
            balance=float(payload.get("balance") or 0.0),
            margin_used=float(payload.get("marginUsed") or payload.get("margin_used") or 0.0),
            open_positions=int(payload.get("openPositionCount") or payload.get("open_positions") or 0),
            gross_exposure=float(payload.get("positionValue") or payload.get("gross_exposure") or 0.0),
            portfolio_risk=float(payload.get("portfolio_risk") or 0.0),
            payload=_jsonable(payload),
        )
        with self.sessions.begin() as session:
            session.add(row)
            session.flush()
            session.expunge(row)
        self.write_jsonl("equity", row)
        return row

    def record_position_snapshot(
        self,
        *,
        timestamp: datetime,
        instrument: str,
        side: str,
        units: float,
        avg_price: float,
        unrealized_pl: float,
        margin_used: float,
        price: float | None,
        payload: dict[str, Any],
        estimated_daily_financing: float = 0.0,
    ) -> PositionSnapshotRow:
        row = PositionSnapshotRow(
            timestamp=_aware(timestamp),
            instrument=instrument.upper(),
            side=side,
            units=units,
            avg_price=avg_price,
            unrealized_pl=unrealized_pl,
            margin_used=margin_used,
            price=price,
            payload=_jsonable(payload),
        )
        with self.sessions.begin() as session:
            session.add(row)
            session.flush()
            session.expunge(row)
        self.write_jsonl("position", row)
        self.upsert_current_position(
            timestamp=timestamp,
            instrument=instrument,
            side=side,
            units=units,
            avg_price=avg_price,
            unrealized_pl=unrealized_pl,
            margin_used=margin_used,
            price=price,
            estimated_daily_financing=estimated_daily_financing,
            payload=payload,
        )
        return row

    def upsert_current_position(
        self,
        *,
        timestamp: datetime,
        instrument: str,
        side: str | None,
        units: float,
        avg_price: float,
        unrealized_pl: float,
        margin_used: float,
        price: float | None,
        estimated_daily_financing: float = 0.0,
        payload: dict[str, Any] | None = None,
        strategy_hash: str | None = None,
        code_version: str | None = None,
        data_hash: str | None = None,
        experiment_manifest_hash: str | None = None,
    ) -> CurrentPositionRow:
        name = instrument.upper()
        with self.sessions.begin() as session:
            row = session.get(CurrentPositionRow, name)
            if row is None:
                row = CurrentPositionRow(instrument=name)
                session.add(row)
            row.side = side
            row.units = units
            row.avg_price = avg_price
            row.unrealized_pl = unrealized_pl
            row.margin_used = margin_used
            row.price = price
            row.estimated_daily_financing = estimated_daily_financing
            row.updated_at = _aware(timestamp)
            row.payload = _jsonable(payload or {})
            session.flush()
            session.expunge(row)
        return row

    def mark_current_positions_closed(self, active_instruments: set[str], timestamp: datetime) -> None:
        active = {name.upper() for name in active_instruments}
        with self.sessions.begin() as session:
            rows = list(session.scalars(select(CurrentPositionRow)))
            for row in rows:
                if row.instrument not in active and row.units != 0:
                    row.side = None
                    row.units = 0.0
                    row.avg_price = 0.0
                    row.unrealized_pl = 0.0
                    row.margin_used = 0.0
                    row.price = None
                    row.estimated_daily_financing = 0.0
                    row.updated_at = _aware(timestamp)
                    row.payload = {}

    def find_trade(self, broker_trade_id: str) -> TradeJournalRow | None:
        with self.sessions() as session:
            row = session.scalar(select(TradeJournalRow).where(TradeJournalRow.broker_trade_id == broker_trade_id))
            if row is not None:
                session.expunge(row)
            return row

    def reconcile_duplicate_open_trade(
        self,
        *,
        canonical_trade_id: str,
        instrument: str,
        units: float,
        exit_time: datetime | None,
        exit_price: float | None,
    ) -> str | None:
        """Archive an old order-ticket row after MT5 position-ID recovery.

        Older worker versions journaled ``order`` as the trade ID, while MT5
        close history uses ``position_id``. If both rows exist, marking both
        closed would double-count P&L. Only an unambiguous same-symbol,
        same-volume open row is archived as a reconciliation alias.
        """
        name = instrument.upper()
        with self.sessions.begin() as session:
            rows = list(session.scalars(
                select(TradeJournalRow).where(
                    TradeJournalRow.state == "open",
                    TradeJournalRow.instrument == name,
                    TradeJournalRow.broker_trade_id != str(canonical_trade_id),
                )
            ))
            candidates = [row for row in rows if math.isclose(
                float(row.units or 0.0), float(units or 0.0), rel_tol=1e-6, abs_tol=0.01
            )]
            if len(candidates) != 1:
                return None
            row = candidates[0]
            row.state = "reconciled_alias"
            row.exit_time = _aware(exit_time) if exit_time else row.exit_time
            row.exit_price = exit_price if exit_price is not None else row.exit_price
            row.exit_reason = "mt5_position_id_reconciliation"
            row.payload = _jsonable({
                **(row.payload or {}),
                "canonical_broker_trade_id": str(canonical_trade_id),
            })
            return row.broker_trade_id

    def update_trade_payload(self, broker_trade_id: str, payload: dict[str, Any]) -> None:
        with self.sessions.begin() as session:
            row = session.scalar(select(TradeJournalRow).where(TradeJournalRow.broker_trade_id == broker_trade_id))
            if row is not None:
                row.payload = _jsonable({**(row.payload or {}), **payload})

    def has_unresolved_orders(self) -> bool:
        with self.sessions() as session:
            return session.scalar(select(OrderJournalRow.id).where(
                OrderJournalRow.status.in_(("pending", "unknown", "submitted", "operator_review"))).limit(1)) is not None

    def upsert_trade(
        self,
        *,
        broker_trade_id: str,
        instrument: str,
        side: str,
        units: float,
        state: str,
        entry_time: datetime | None = None,
        entry_price: float | None = None,
        exit_time: datetime | None = None,
        exit_price: float | None = None,
        realized_pl: float = 0.0,
        financing: float = 0.0,
        exit_reason: str | None = None,
        payload: dict[str, Any] | None = None,
        strategy_hash: str | None = None,
        code_version: str | None = None,
        data_hash: str | None = None,
        experiment_manifest_hash: str | None = None,
    ) -> TradeJournalRow:
        with self.sessions.begin() as session:
            row = session.scalar(
                select(TradeJournalRow).where(TradeJournalRow.broker_trade_id == broker_trade_id)
            )
            if row is None:
                row = TradeJournalRow(
                    broker_trade_id=broker_trade_id,
                    instrument=instrument.upper(),
                    side=side,
                    units=units,
                    state=state,
                    strategy_hash=strategy_hash or self.strategy_hash,
                    code_version=code_version or self.code_version,
                    data_hash=data_hash or self.data_hash,
                    experiment_manifest_hash=experiment_manifest_hash or self.experiment_manifest_hash,
                )
                session.add(row)
            row.entry_time = _aware(entry_time) if entry_time else row.entry_time
            row.entry_price = entry_price if entry_price is not None else row.entry_price
            row.exit_time = _aware(exit_time) if exit_time else row.exit_time
            row.exit_price = exit_price if exit_price is not None else row.exit_price
            row.realized_pl = realized_pl
            row.financing = financing
            row.state = state
            row.exit_reason = exit_reason
            # Broker close-history payloads arrive after the initial fill
            # record. Merge rather than replace so the strategy context
            # (quality score, entry feature snapshot, decision time) remains
            # available for unbiased post-trade calibration.
            row.payload = _jsonable({**(row.payload or {}), **(payload or {})})
            session.flush()
            session.expunge(row)
        self.write_jsonl("trade", row)
        return row

    def add_trade_financing(
        self,
        *,
        broker_trade_id: str,
        financing: float,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if not broker_trade_id:
            return
        unmatched = False
        with self.sessions.begin() as session:
            row = session.scalar(
                select(TradeJournalRow).where(TradeJournalRow.broker_trade_id == broker_trade_id)
            )
            if row is None:
                unmatched = True
            else:
                row.financing = float(row.financing or 0.0) + financing
                row.payload = _jsonable({**(row.payload or {}), "latest_financing_transaction": payload or {}})
        if unmatched:
            self.log_event(
                "financing_unmatched_trade",
                f"financing received for unknown trade {broker_trade_id}",
                payload=payload or {},
            )

    def record_run_manifest(
        self,
        *,
        run_id: str,
        strategy_hash: str,
        code_version: str,
        data_hash: str,
        manifest_hash: str,
        manifest: dict[str, Any],
    ) -> RunManifestRow:
        """Persist an immutable research/forward-test manifest in SQLite."""

        with self.sessions.begin() as session:
            row = session.get(RunManifestRow, run_id)
            if row is None:
                row = RunManifestRow(
                    run_id=run_id,
                    strategy_hash=strategy_hash,
                    code_version=code_version,
                    data_hash=data_hash,
                    manifest_hash=manifest_hash,
                    manifest=_jsonable(manifest),
                )
                session.add(row)
            elif row.manifest_hash != manifest_hash:
                raise ValueError(f"run manifest {run_id} is immutable")
            session.flush()
            session.expunge(row)
        return row

    def log_event(self, event_type: str, message: str, *, level: str = "info", payload: dict[str, Any] | None = None) -> None:
        row = EventLogRow(
            timestamp=utc_now(),
            level=level,
            event_type=event_type,
            message=message,
            payload=_jsonable(redact(payload or {})),
        )
        with self.sessions.begin() as session:
            session.add(row)
        self.write_jsonl("event", row)

    def recent_events(self, limit: int = 100) -> list[EventLogRow]:
        return _recent(self.sessions, EventLogRow, limit)

    def recovery_orders(self, limit: int = 100) -> list[OrderJournalRow]:
        with self.sessions() as session:
            rows = list(
                session.scalars(
                    select(OrderJournalRow)
                    .where(OrderJournalRow.status.in_(("pending", "unknown", "submitted")))
                    .order_by(OrderJournalRow.id)
                    .limit(limit)
                )
            )
            for row in rows:
                session.expunge(row)
            return rows

    def latest_equity(self, limit: int = 500) -> list[EquitySnapshotRow]:
        with self.sessions() as session:
            rows = list(session.scalars(select(EquitySnapshotRow).order_by(desc(EquitySnapshotRow.timestamp)).limit(limit)))
            for row in rows:
                session.expunge(row)
            return list(reversed(rows))

    def equity_history(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 1000,
    ) -> list[EquitySnapshotRow]:
        """Return chronologically ordered equity history, downsampled safely.

        The worker records a snapshot every loop, so returning the latest 500
        rows does not represent a seven-day chart. Query the requested window
        first, then keep evenly distributed points plus the first/last point.
        This keeps date ranges correct without sending tens of thousands of
        ten-second snapshots to the browser.
        """
        limit = max(2, int(limit))
        statement = select(EquitySnapshotRow)
        if start is not None:
            statement = statement.where(EquitySnapshotRow.timestamp >= _aware(start))
        if end is not None:
            statement = statement.where(EquitySnapshotRow.timestamp <= _aware(end))
        statement = statement.order_by(EquitySnapshotRow.timestamp, EquitySnapshotRow.id)
        with self.sessions() as session:
            rows = list(session.scalars(statement))
            for row in rows:
                session.expunge(row)
        if len(rows) <= limit:
            return rows
        # Include endpoints and select deterministic positions in between.
        indices = {0, len(rows) - 1}
        span = len(rows) - 1
        for position in range(1, limit - 1):
            indices.add(round(position * span / (limit - 1)))
        return [rows[index] for index in sorted(indices)]

    def current_positions(self) -> list[CurrentPositionRow]:
        with self.sessions() as session:
            rows = list(
                session.scalars(
                    select(CurrentPositionRow)
                    .where(CurrentPositionRow.units > 0)
                    .order_by(CurrentPositionRow.instrument)
                )
            )
            for row in rows:
                session.expunge(row)
            return rows

    def recent_orders(self, limit: int = 200) -> list[OrderJournalRow]:
        return _recent(self.sessions, OrderJournalRow, limit)

    def recent_signals(self, limit: int = 200) -> list[SignalJournalRow]:
        return _recent(self.sessions, SignalJournalRow, limit)

    def recent_trades(self, limit: int = 200) -> list[TradeJournalRow]:
        return _recent(self.sessions, TradeJournalRow, limit)

    def filtered_trades(
        self,
        *,
        instrument: str | None = None,
        state: str | None = None,
        outcome: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> list[TradeJournalRow]:
        statement = select(TradeJournalRow)
        if instrument:
            statement = statement.where(TradeJournalRow.instrument == instrument.upper())
        if state:
            statement = statement.where(TradeJournalRow.state == state.lower())
        else:
            # Old MT5 versions could leave an opening-order-ticket duplicate
            # after the canonical position-ticket row was reconciled. Keep the
            # audit row in SQLite, but hide it from normal dashboard/analytics
            # queries so it cannot look like a second live trade.
            statement = statement.where(TradeJournalRow.state != "reconciled_alias")
        if start:
            statement = statement.where(TradeJournalRow.entry_time >= _aware(start))
        if end:
            statement = statement.where(TradeJournalRow.entry_time <= _aware(end))
        total_pnl = TradeJournalRow.realized_pl + TradeJournalRow.financing
        if outcome == "win":
            statement = statement.where(total_pnl > 0)
        elif outcome == "loss":
            statement = statement.where(total_pnl < 0)
        latest_trade_time = func.coalesce(TradeJournalRow.exit_time, TradeJournalRow.entry_time)
        statement = statement.order_by(desc(latest_trade_time), desc(TradeJournalRow.id)).limit(limit)
        with self.sessions() as session:
            rows = list(session.scalars(statement))
            for row in rows:
                session.expunge(row)
            return rows

    def open_risk_amount(self) -> float:
        with self.sessions() as session:
            rows = session.scalars(
                select(OrderJournalRow).where(OrderJournalRow.status.in_(("pending", "submitted", "filled")))
            )
            return sum(float(row.risk_amount or 0.0) for row in rows)

    def set_last_transaction_id(self, transaction_id: str | None) -> None:
        if not transaction_id:
            return
        with self.sessions.begin() as session:
            row = session.get(BotStateRow, 1)
            if row is None:
                row = BotStateRow(id=1)
                session.add(row)
            row.last_transaction_id = str(transaction_id)
            row.updated_at = utc_now()

    def update_protection_state(
        self,
        *,
        timestamp: datetime,
        equity: float,
        max_daily_loss_pct: float,
        max_drawdown_pct: float,
    ) -> str | None:
        day = _aware(timestamp).date().isoformat()
        reason: str | None = None
        with self.sessions.begin() as session:
            row = session.get(BotStateRow, 1)
            if row is None:
                row = BotStateRow(id=1)
                session.add(row)

            if row.daily_start_day != day:
                row.daily_start_day = day
                row.daily_start_equity = equity
                if row.halted_day and row.halted_day != day and row.reason == "daily_loss_halt":
                    row.halted_day = None

            row.peak_equity = max(float(row.peak_equity or 0.0), equity)
            start = float(row.daily_start_equity or equity)
            if row.state == BotRunState.HALTED.value and row.reason:
                reason = row.reason
            elif row.peak_equity and row.peak_equity > 0 and max_drawdown_pct > 0:
                drawdown = (row.peak_equity - equity) / row.peak_equity
                if drawdown >= max_drawdown_pct:
                    reason = "max_drawdown_halt"
            if reason is None and start > 0 and max_daily_loss_pct > 0:
                daily_loss = (start - equity) / start
                if daily_loss >= max_daily_loss_pct:
                    reason = "daily_loss_halt"
                    row.halted_day = day

            if reason:
                row.state = BotRunState.HALTED.value
                row.reason = reason
            row.updated_at = utc_now()
        if reason:
            self.write_jsonl("bot_state", {"state": BotRunState.HALTED.value, "reason": reason})
        return reason

    def write_jsonl(self, event_type: str, payload: Any) -> None:
        if self.jsonl_path is None:
            return
        record = {
            "type": event_type,
            "timestamp": utc_now().isoformat(),
            "payload": _jsonable(payload),
        }
        # JSONL is a compatibility export only; SQLite remains canonical.  A
        # lock file keeps concurrent worker/API processes from interleaving
        # records when an operator explicitly enables this export.
        lock_path = self.jsonl_path.with_suffix(self.jsonl_path.suffix + ".lock")
        with self._jsonl_lock, lock_path.open("a+b") as lock_handle:
            _lock_file(lock_handle)
            try:
                with self.jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(redact(record), sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                _unlock_file(lock_handle)


def row_to_dict(row: Any) -> dict[str, Any]:
    data = {column.name: getattr(row, column.name) for column in row.__table__.columns}
    return _jsonable(data)


def _recent(sessions: sessionmaker[Session], model: Any, limit: int) -> list[Any]:
    with sessions() as session:
        rows = list(session.scalars(select(model).order_by(desc(model.id)).limit(limit)))
        for row in rows:
            session.expunge(row)
        return rows


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "__table__"):
        return row_to_dict(value)
    if isinstance(value, datetime):
        # SQLite returns DateTime columns without tzinfo even when the model
        # column is timezone-aware. Always emit an explicit UTC offset so the
        # frontend does not reinterpret journal dates in the browser's local
        # timezone at midnight or during chart range filtering.
        return _aware(value).isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _lock_file(handle: Any) -> None:
    try:
        import msvcrt

        handle.seek(0)
        handle.write(b"0")
        handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
    except ImportError:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_file(handle: Any) -> None:
    try:
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    except ImportError:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
