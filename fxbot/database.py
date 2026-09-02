"""SQLAlchemy persistence models for forward testing and the API."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, UniqueConstraint, create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


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
        engine = create_engine(database_url, connect_args={"check_same_thread": False})
    else:
        engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    if database_url.startswith("sqlite:///"):
        _ensure_sqlite_columns(engine)
    return sessionmaker(engine, expire_on_commit=False)


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
