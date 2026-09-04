"""Structured SQL/JSONL journal for unattended forward testing."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session, close_all_sessions, sessionmaker

from fxbot.database import (
    BotStateRow,
    CurrentPositionRow,
    EquitySnapshotRow,
    EventLogRow,
    OrderJournalRow,
    PositionSnapshotRow,
    SignalJournalRow,
    TradeJournalRow,
    session_factory,
    utc_now,
)
from fxbot.models import BotRunState


class StructuredJournal:
    def __init__(self, database_url: str, jsonl_path: str | None = None) -> None:
        self.sessions = session_factory(database_url)
        self._engine = self.sessions.kw.get("bind")
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
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
        )
        with self.sessions.begin() as session:
            session.add(row)
            session.flush()
            session.expunge(row)
        self.write_jsonl("signal", row)
        return row

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
            row.payload = _jsonable(payload or row.payload or {})
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

    def log_event(self, event_type: str, message: str, *, level: str = "info", payload: dict[str, Any] | None = None) -> None:
        row = EventLogRow(
            timestamp=utc_now(),
            level=level,
            event_type=event_type,
            message=message,
            payload=_jsonable(payload or {}),
        )
        with self.sessions.begin() as session:
            session.add(row)
        self.write_jsonl("event", row)

    def latest_equity(self, limit: int = 500) -> list[EquitySnapshotRow]:
        with self.sessions() as session:
            rows = list(session.scalars(select(EquitySnapshotRow).order_by(desc(EquitySnapshotRow.timestamp)).limit(limit)))
            for row in rows:
                session.expunge(row)
            return list(reversed(rows))

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
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


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
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
