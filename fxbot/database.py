"""SQLAlchemy persistence models for forward testing and the API."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, UniqueConstraint, create_engine, event, inspect, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

log = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class BotStateRow(Base):
    __tablename__ = "bot_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    state: Mapped[str] = mapped_column(String(32), default="stopped", nullable=False)
    reason: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    daily_start_day: Mapped[str | None] = mapped_column(String(16), nullable=True)
    daily_start_equity: Mapped[float | None] = mapped_column(Float, nullable=True)
    halted_day: Mapped[str | None] = mapped_column(String(16), nullable=True)
    peak_equity: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_transaction_id: Mapped[str | None] = mapped_column(String(64), nullable=True)


class SignalJournalRow(Base):
    __tablename__ = "signal_journal"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    instrument: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    side: Mapped[str | None] = mapped_column(String(16), nullable=True)
    status: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    reason: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    risk_amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    strategy_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    code_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    data_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    experiment_manifest_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)


class OrderJournalRow(Base):
    __tablename__ = "order_journal"
    __table_args__ = (UniqueConstraint("client_order_id", name="uq_order_client_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_order_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    instrument: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    side: Mapped[str] = mapped_column(String(16), nullable=False)
    units: Mapped[float] = mapped_column(Float, nullable=False)
    order_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    broker_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    broker_trade_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    risk_amount: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    response: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    strategy_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    code_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    data_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    experiment_manifest_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)


class TradeJournalRow(Base):
    __tablename__ = "trade_journal"
    __table_args__ = (UniqueConstraint("broker_trade_id", name="uq_trade_broker_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    broker_trade_id: Mapped[str] = mapped_column(String(64), nullable=False)
    instrument: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    side: Mapped[str] = mapped_column(String(16), nullable=False)
    units: Mapped[float] = mapped_column(Float, nullable=False)
    entry_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    realized_pl: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    financing: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    state: Mapped[str] = mapped_column(String(32), index=True, default="open", nullable=False)
    exit_reason: Mapped[str | None] = mapped_column(String(128), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    strategy_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    code_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    data_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    experiment_manifest_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)


class AiDeliberationRow(Base):
    """One immutable AI audit per parent signal, never per exit/order leg."""

    __tablename__ = "ai_deliberations"
    __table_args__ = (UniqueConstraint("signal_id", name="uq_ai_deliberation_signal_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    signal_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    instrument: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(16), nullable=False)
    model: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    decision: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    reasoning_audit_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reasoning_issues: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    reasoning_supporting_factors: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    market_context_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    market_context_issues: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    market_context_supporting_factors: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    contradictions: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    recommended_action: Mapped[str | None] = mapped_column(String(16), nullable=True)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    evidence_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    output_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    response: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class RunManifestRow(Base):
    __tablename__ = "run_manifest"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    strategy_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    code_version: Mapped[str] = mapped_column(String(64), nullable=False)
    data_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    manifest_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class PositionSnapshotRow(Base):
    __tablename__ = "position_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    instrument: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    side: Mapped[str] = mapped_column(String(16), nullable=False)
    units: Mapped[float] = mapped_column(Float, nullable=False)
    avg_price: Mapped[float] = mapped_column(Float, nullable=False)
    unrealized_pl: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    margin_used: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class CurrentPositionRow(Base):
    __tablename__ = "current_positions"

    instrument: Mapped[str] = mapped_column(String(32), primary_key=True)
    side: Mapped[str | None] = mapped_column(String(16), nullable=True)
    units: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    avg_price: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    unrealized_pl: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    margin_used: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    estimated_daily_financing: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class EquitySnapshotRow(Base):
    __tablename__ = "equity_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    equity: Mapped[float] = mapped_column(Float, nullable=False)
    balance: Mapped[float] = mapped_column(Float, nullable=False)
    margin_used: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    open_positions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    gross_exposure: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    portfolio_risk: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class NewsEventRow(Base):
    __tablename__ = "news_events"
    __table_args__ = (UniqueConstraint("event_id", name="uq_news_event_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    currency: Mapped[str] = mapped_column(String(8), index=True, nullable=False)
    country: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    scheduled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    impact: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    impact_score: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    forecast: Mapped[float | None] = mapped_column(Float, nullable=True)
    previous: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False, onupdate=utc_now)


class EventLogRow(Base):
    __tablename__ = "event_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    level: Mapped[str] = mapped_column(String(16), default="info", nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


def session_factory(database_url: str) -> sessionmaker[Session]:
    if database_url.startswith("sqlite:///"):
        path = database_url.replace("sqlite:///", "", 1)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False, "timeout": _SQLITE_BUSY_TIMEOUT_S},
        )
        _configure_sqlite(engine)
    else:
        engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    if database_url.startswith("sqlite:///"):
        _ensure_sqlite_columns(engine)
    return sessionmaker(engine, expire_on_commit=False)


_SQLITE_BUSY_TIMEOUT_S = 30
_SQLITE_BUSY_TIMEOUT_MS = _SQLITE_BUSY_TIMEOUT_S * 1000
_sqlite_lock_retries = 0
_lock_retry_callback: Any = None


def set_lock_retry_callback(callback: Any) -> None:
    """Register an optional callback invoked on every SQLite lock retry.

    The callback receives no arguments and is called from the SQLAlchemy
    ``handle_error`` event, so it must be lightweight and thread-safe.
    A typical use is ``monitor.record_db_lock_retry``.
    """
    global _lock_retry_callback
    _lock_retry_callback = callback


def _configure_sqlite(engine: Any) -> None:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection: Any, connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute(f"PRAGMA busy_timeout={_SQLITE_BUSY_TIMEOUT_MS}")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
        finally:
            cursor.close()

    @event.listens_for(engine, "handle_error")
    def _count_sqlite_lock(error_context: Any) -> None:
        global _sqlite_lock_retries
        message = str(error_context.original_exception).lower()
        if "locked" in message or "busy" in message:
            _sqlite_lock_retries += 1
            if _lock_retry_callback is not None:
                try:
                    _lock_retry_callback()
                except Exception:
                    pass


def sqlite_lock_metrics() -> dict[str, int]:
    return {"database_lock_retries": _sqlite_lock_retries}


def sqlite_retry_operation(
    func: Any,
    *,
    max_retries: int = 3,
    base_delay: float = 0.05,
    max_delay: float = 2.0,
) -> Any:
    """Execute *func* with exponential-backoff retry on SQLite lock errors.

    *func* must be a zero-argument callable that returns a value or ``None``.
    Raises the original ``OperationalError`` after exhausting retries.
    """
    last_exc: OperationalError | None = None
    for attempt in range(max_retries):
        try:
            return func()
        except OperationalError as exc:
            message = str(exc).lower()
            if "locked" not in message and "busy" not in message:
                raise
            last_exc = exc
            delay = min(base_delay * (2 ** attempt), max_delay)
            log.warning("SQLite lock on attempt %d/%d, retrying in %.3fs", attempt + 1, max_retries, delay)
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def _ensure_sqlite_columns(engine: Any) -> None:
    existing = {column["name"] for column in inspect(engine).get_columns("bot_state")}
    additions = {
        "daily_start_day": "ALTER TABLE bot_state ADD COLUMN daily_start_day VARCHAR(16)",
        "peak_equity": "ALTER TABLE bot_state ADD COLUMN peak_equity FLOAT",
        "last_transaction_id": "ALTER TABLE bot_state ADD COLUMN last_transaction_id VARCHAR(64)",
    }
    with engine.begin() as connection:
        for name, ddl in additions.items():
            if name not in existing:
                connection.execute(text(ddl))
    for table in ("signal_journal", "order_journal", "trade_journal"):
        existing = {column["name"] for column in inspect(engine).get_columns(table)}
        additions = {
            "strategy_hash": f"ALTER TABLE {table} ADD COLUMN strategy_hash VARCHAR(64)",
            "code_version": f"ALTER TABLE {table} ADD COLUMN code_version VARCHAR(64)",
            "data_hash": f"ALTER TABLE {table} ADD COLUMN data_hash VARCHAR(128)",
            "experiment_manifest_hash": f"ALTER TABLE {table} ADD COLUMN experiment_manifest_hash VARCHAR(128)",
        }
        with engine.begin() as connection:
            for name, ddl in additions.items():
                if name not in existing:
                    connection.execute(text(ddl))
