from __future__ import annotations

import sqlite3
from datetime import datetime

import pytest

from forex_agent.data.ingestion import load_trades_from_database

SCHEMA = """
CREATE TABLE trade_journal (
    id INTEGER PRIMARY KEY,
    broker_trade_id VARCHAR(64) NOT NULL,
    instrument VARCHAR(32) NOT NULL,
    side VARCHAR(16) NOT NULL,
    units FLOAT NOT NULL,
    entry_time DATETIME,
    entry_price FLOAT,
    exit_time DATETIME,
    exit_price FLOAT,
    realized_pl FLOAT DEFAULT 0.0 NOT NULL,
    financing FLOAT DEFAULT 0.0 NOT NULL,
    state VARCHAR(32) DEFAULT 'open' NOT NULL,
    exit_reason VARCHAR(128),
    payload JSON DEFAULT '{}' NOT NULL
)
"""


def build_db(db_path, rows):
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)
    for i, row in enumerate(rows, start=1):
        conn.execute(
            "INSERT INTO trade_journal (id, broker_trade_id, instrument, side, units, "
            "entry_time, entry_price, exit_time, exit_price, realized_pl, financing, "
            "state, exit_reason, payload) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (i, *row),
        )
    conn.commit()
    conn.close()


def now_iso():
    return datetime(2025, 3, 10, 8, 30, 0).isoformat()


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "test.db")


class TestLoadTradesFromDatabase:
    def test_empty_database(self, db):
        conn = sqlite3.connect(db)
        conn.execute(SCHEMA)
        conn.commit()
        conn.close()
        assert load_trades_from_database(db) == []

    def test_missing_database_returns_empty(self):
        assert load_trades_from_database("C:/nonexistent/path.db") == []

    def test_sqlite_url_prefix(self, db):
        build_db(db, [])
        assert load_trades_from_database("sqlite:///" + db) == []

    def test_loads_closed_trade(self, db):
        build_db(db, [
            (
                "BROKER1", "EURUSD", "buy", 1.0,
                now_iso(), 1.1000, now_iso(), 1.1050,
                50.0, 0.0, "filled", "take_profit",
                "{}",
            ),
        ])
        trades = load_trades_from_database(db)
        assert len(trades) == 1
        t = trades[0]
        assert t.trade_id == "BROKER1"
        assert t.symbol == "EURUSD"
        assert t.exit_price == 1.1050
        assert t.realized_pl == 50.0
        assert t.pnl == 50.0

    def test_pnl_prefers_realized_pl(self, db):
        # realized_pl (50.0) dominates the lots-based calc
        build_db(db, [
            (
                "BROKER1", "EURUSD", "buy", 1.0,
                now_iso(), 1.1000, now_iso(), 1.1050,
                50.0, 0.0, "filled", "take_profit",
                "{}",
            ),
        ])
        trades = load_trades_from_database(db)
        assert trades[0].pnl == 50.0

    def test_json_string_payload_parsed(self, db):
        build_db(db, [
            (
                "BROKER2", "GBPUSD", "buy", 1.0,
                now_iso(), 1.2500, now_iso(), 1.2550,
                -10.0, 0.0, "filled", "stop_loss",
                '{"strategy_name": "trend_follow", "regime": "trending"}',
            ),
        ])
        trades = load_trades_from_database(db)
        assert len(trades) == 1
        # Nested payload is captured faithfully on the record.
        assert trades[0].payload["strategy_name"] == "trend_follow"
        assert trades[0].payload["regime"] == "trending"

    def test_open_trade_has_null_exit(self, db):
        build_db(db, [
            (
                "BROKER3", "EURUSD", "sell", 1.0,
                now_iso(), 1.1000, None, None,
                0.0, 0.0, "open", None,
                "{}",
            ),
        ])
        trades = load_trades_from_database(db)
        assert len(trades) == 1
        assert trades[0].exit_price is None
        assert trades[0].pnl == 0.0

    def test_missing_entry_price_skipped(self, db):
        build_db(db, [
            (
                "BROKER4", "EURUSD", "buy", 1.0,
                now_iso(), None, now_iso(), 1.1050,
                0.0, 0.0, "filled", None,
                "{}",
            ),
        ])
        trades = load_trades_from_database(db)
        assert trades == []

    def test_normalizes_underscore_instrument(self, db):
        build_db(db, [
            (
                "BROKER5", "EUR_USD", "buy", 1.0,
                now_iso(), 1.1000, now_iso(), 1.1050,
                10.0, 0.0, "filled", None,
                "{}",
            ),
        ])
        trades = load_trades_from_database(db)
        assert trades[0].symbol == "EURUSD"

    def test_multiple_trades_preserved_order(self, db):
        build_db(db, [
            (
                "BROKER1", "EURUSD", "buy", 1.0,
                now_iso(), 1.1000, now_iso(), 1.1050,
                50.0, 0.0, "filled", None,
                "{}",
            ),
            (
                "BROKER2", "GBPUSD", "sell", 1.0,
                now_iso(), 1.2500, now_iso(), 1.2450,
                25.0, 0.0, "filled", None,
                "{}",
            ),
        ])
        trades = load_trades_from_database(db)
        assert [t.trade_id for t in trades] == ["BROKER1", "BROKER2"]
