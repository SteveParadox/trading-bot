from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from forex_agent.data.schemas import MarketRegime, TradeRecord
from forex_agent.log import get_logger

logger = get_logger(__name__)


def load_trades_from_jsonl(path: str | Path) -> list[TradeRecord]:
    """Load trades from a JSONL journal.

    Supports two formats:
      1. Flat format: each line is a flat dict of trade fields.
      2. fxbot journal format: each line is ``{"type": "trade", "timestamp":
         ..., "payload": {...}}``.

    Only ``type == "trade"`` records are parsed from the journal format.
    """
    trades: list[TradeRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Skipping malformed JSONL line %d in %s", line_no, path)
                continue

            log_type = data.get("type")
            if log_type is not None:
                if log_type != "trade":
                    continue
                payload = data.get("payload")
                if not isinstance(payload, dict):
                    continue
                trade_data = payload
            else:
                trade_data = data

            trade = _parse_trade_record(trade_data)
            if trade is not None:
                trades.append(trade)

    return trades


def _parse_trade_record(data: dict) -> Optional[TradeRecord]:
    """Parse a trade payload dict into a TradeRecord (or None if invalid)."""

    entry_price = data.get("entry_price")
    if entry_price is None or entry_price <= 0:
        return None

    try:
        entry_time = _parse_dt(data.get("entry_time"))
    except (ValueError, TypeError):
        return None

    exit_price = data.get("exit_price")
    exit_time = _parse_optional_dt(data.get("exit_time"))
    if exit_price is not None:
        try:
            exit_price = float(exit_price)
        except (TypeError, ValueError):
            exit_price = None
    if exit_time is None and exit_price is not None:
        # Exit time unknown but exit price present: keep exit price.
        pass

    direction = (data.get("direction") or data.get("side") or "LONG").upper()
    if direction not in ("LONG", "SHORT"):
        direction = "LONG"

    stop_loss = data.get("stop_loss") or data.get("sl")
    if stop_loss is not None:
        try:
            stop_loss = float(stop_loss)
        except (TypeError, ValueError):
            stop_loss = None
    if stop_loss is None or stop_loss <= 0:
        # MT5 often leaves SL at 0.0, meaning no stop placed.
        stop_loss = None

    take_profit = data.get("take_profit") or data.get("tp")
    if take_profit is not None and take_profit == 0:
        take_profit = None

    symbol = data.get("symbol") or data.get("instrument") or "UNKNOWN"
    if "_" in symbol and len(symbol) == 7:
        # EUR_USD -> EURUSD
        symbol = symbol.replace("_", "")

    position_size = data.get("position_size")
    if position_size is None:
        position_size = data.get("units")
    try:
        position_size = float(position_size) if position_size is not None else 1.0
    except (TypeError, ValueError):
        position_size = 1.0

    sponsor = TradeRecord(
        trade_id=str(data.get("trade_id") or data.get("broker_trade_id") or data.get("id") or ""),
        symbol=symbol,
        direction=direction,
        entry_price=float(entry_price),
        exit_price=exit_price,
        stop_loss=stop_loss if stop_loss is not None else (entry_price * 0.99 if direction == "LONG" else entry_price * 1.01),
        take_profit=take_profit,
            position_size=position_size,
            account_balance=float(data.get("account_balance") or 10000.0),
            risk_amount=float(data.get("risk_amount") or 0.0),
            realized_pl=_optional_float(data.get("realized_pl")),
            entry_time=entry_time,
        exit_time=exit_time,
        spread_at_entry=float(data.get("spread_at_entry") or data.get("spread") or 0.0),
        slippage_pips=float(data.get("slippage_pips") or data.get("slippage") or 0.0),
        commission=float(data.get("commission") or 0.0),
        financing=float(data.get("financing") or 0.0),
        notes=data.get("notes") or data.get("exit_reason") or "",
        mfe=_optional_float(data.get("mfe")),
        mae=_optional_float(data.get("mae")),
        exit_reason=data.get("exit_reason"),
        strategy_name=data.get("strategy_name") or data.get("strategy"),
        timeframe=data.get("timeframe"),
        entry_reason=data.get("entry_reason"),
        signal_strength=_optional_float(data.get("signal_strength") or data.get("score")),
        volatility=_optional_float(data.get("volatility")),
        atr=_optional_float(data.get("atr") or data.get("atr_at_entry")),
        trend_characteristics=data.get("trend_characteristics"),
        news_context=data.get("news_context"),
        payload=dict(data.get("payload") or {}),
    )

    regime = data.get("regime")
    if regime:
        try:
            sponsor.regime = MarketRegime(regime)
        except ValueError:
            sponsor.regime = None

    return sponsor


def _parse_dt(value) -> datetime:
    if value is None:
        raise ValueError("missing datetime")
    if isinstance(value, datetime):
        return value
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed
    return parsed.astimezone(datetime.now().astimezone().tzinfo).replace(tzinfo=None)


def _parse_optional_dt(value) -> Optional[datetime]:
    if value is None or value == "":
        return None
    try:
        return _parse_dt(value)
    except (ValueError, TypeError):
        return None


def _optional_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_r_multiple(trade: TradeRecord) -> Optional[float]:
    if trade.exit_price is None:
        return None
    risk = abs(trade.entry_price - trade.stop_loss)
    if risk == 0:
        return None
    if trade.direction == "LONG":
        reward = trade.exit_price - trade.entry_price
    else:
        reward = trade.entry_price - trade.exit_price
    return reward / risk


def compute_r_values(trades: list[TradeRecord]) -> list[float]:
    """Collect R-multiples for a batch of trades, skipping un-computable ones.

    A trade is skipped when it has no exit price (open) or a zero stop
    distance (no risk to measure against). This is the canonical batched
    R-multiple helper; other modules should rely on it for a single source
    of truth.
    """
    return [r for r in (compute_r_multiple(t) for t in trades) if r is not None]


def compute_trade_duration(trade: TradeRecord) -> Optional[float]:
    if trade.exit_time is None:
        return None
    delta = trade.exit_time - trade.entry_time
    return delta.total_seconds() / 60.0


def extract_session(dt: datetime) -> str:
    hour = dt.hour
    if 0 <= hour < 7:
        return "asian"
    elif 7 <= hour < 13:
        return "london"
    elif 13 <= hour < 21:
        return "new_york"
    return "off_hours"


def extract_day_of_week(dt: datetime) -> str:
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    return days[dt.weekday()]


def load_trades_from_database(database_url: str) -> list[TradeRecord]:
    """Load trades from the fxbot SQLite database (trade_journal table)."""
    import sqlite3

    db_path = database_url
    if database_url.startswith("sqlite:///"):
        db_path = database_url.replace("sqlite:///", "", 1)
    if not Path(db_path).exists():
        logger.info("Trade database not found; skipping: %s", db_path)
        return []

    trades: list[TradeRecord] = []
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM trade_journal ORDER BY id"
        ).fetchall()
        conn.close()
    except sqlite3.Error as exc:
        logger.warning("Failed to read trade database %s: %s", db_path, exc)
        return []

    for row in rows:
        row_dict = dict(row)
        # Build a payload dict compatible with _parse_trade_record.
        raw_payload = row_dict.get("payload") or {}
        if isinstance(raw_payload, str):
            try:
                raw_payload = json.loads(raw_payload)
            except (ValueError, TypeError):
                raw_payload = {}
        if not isinstance(raw_payload, dict):
            raw_payload = {}

        payload = {
            "broker_trade_id": str(row_dict.get("broker_trade_id") or row_dict.get("id") or ""),
            "instrument": row_dict.get("instrument") or "UNKNOWN",
            "side": (row_dict.get("side") or "LONG").upper(),
            "units": row_dict.get("units"),
            "entry_time": row_dict.get("entry_time"),
            "entry_price": row_dict.get("entry_price"),
            "exit_time": row_dict.get("exit_time"),
            "exit_price": row_dict.get("exit_price"),
            "realized_pl": row_dict.get("realized_pl"),
            "financing": row_dict.get("financing"),
            "exit_reason": row_dict.get("exit_reason"),
            "state": row_dict.get("state"),
            "payload": raw_payload,
        }
        trade = _parse_trade_record(payload)
        if trade is not None:
            trades.append(trade)
    return trades
