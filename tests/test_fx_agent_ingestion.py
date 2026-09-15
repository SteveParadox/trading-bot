from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta

import pytest

from forex_agent.data.schemas import TradeRecord
from forex_agent.data.ingestion import (
    compute_r_multiple,
    compute_r_values,
    compute_trade_duration,
    extract_day_of_week,
    extract_session,
    load_trades_from_jsonl,
)


def make_trade(**overrides) -> TradeRecord:
    defaults = dict(
        trade_id="T001",
        symbol="EURUSD",
        direction="LONG",
        entry_price=1.1000,
        exit_price=1.1050,
        stop_loss=1.0950,
        take_profit=1.1100,
        position_size=1.0,
        account_balance=10000.0,
        risk_amount=50.0,
        entry_time=datetime(2025, 3, 10, 8, 30, 0),
        exit_time=datetime(2025, 3, 10, 10, 30, 0),
        spread_at_entry=1.5,
        slippage_pips=0.0,
        commission=0.0,
    )
    defaults.update(overrides)
    return TradeRecord(**defaults)


class TestMakeTradeHelper:
    def test_default_trade(self):
        t = make_trade()
        assert t.symbol == "EURUSD"
        assert t.direction == "LONG"
        assert t.entry_price == 1.1000
        assert t.exit_price == 1.1050
        assert t.trade_id == "T001"

    def test_override_fields(self):
        t = make_trade(symbol="GBPUSD", direction="SHORT", entry_price=1.2500)
        assert t.symbol == "GBPUSD"
        assert t.direction == "SHORT"
        assert t.entry_price == 1.2500


class TestComputeRMultiple:
    def test_long_winner_1r(self):
        t = make_trade(entry_price=1.1000, stop_loss=1.0950, exit_price=1.1050, direction="LONG")
        r = compute_r_multiple(t)
        assert r == pytest.approx(1.0)

    def test_long_winner_2r(self):
        t = make_trade(entry_price=1.1000, stop_loss=1.0950, exit_price=1.1100, direction="LONG")
        r = compute_r_multiple(t)
        assert r == pytest.approx(2.0)

    def test_long_loser(self):
        t = make_trade(entry_price=1.1000, stop_loss=1.0950, exit_price=1.0975, direction="LONG")
        r = compute_r_multiple(t)
        assert r == pytest.approx(-0.5)

    def test_long_full_stop(self):
        t = make_trade(entry_price=1.1000, stop_loss=1.0950, exit_price=1.0950, direction="LONG")
        r = compute_r_multiple(t)
        assert r == pytest.approx(-1.0)

    def test_short_winner(self):
        t = make_trade(entry_price=1.1000, stop_loss=1.1050, exit_price=1.0950, direction="SHORT")
        r = compute_r_multiple(t)
        assert r == pytest.approx(1.0)

    def test_short_loser(self):
        t = make_trade(entry_price=1.1000, stop_loss=1.1050, exit_price=1.1030, direction="SHORT")
        r = compute_r_multiple(t)
        assert r == pytest.approx(-0.6)

    def test_no_exit(self):
        t = make_trade(exit_price=None)
        r = compute_r_multiple(t)
        assert r is None

    def test_zero_risk(self):
        t = make_trade(entry_price=1.1000, stop_loss=1.1000, exit_price=1.1050)
        r = compute_r_multiple(t)
        assert r is None


class TestComputeTradeDuration:
    def test_two_hours(self):
        t = make_trade(
            entry_time=datetime(2025, 1, 1, 8, 0),
            exit_time=datetime(2025, 1, 1, 10, 0),
        )
        assert compute_trade_duration(t) == pytest.approx(120.0)

    def test_thirty_minutes(self):
        t = make_trade(
            entry_time=datetime(2025, 1, 1, 8, 0),
            exit_time=datetime(2025, 1, 1, 8, 30),
        )
        assert compute_trade_duration(t) == pytest.approx(30.0)

    def test_no_exit(self):
        t = make_trade(exit_time=None)
        assert compute_trade_duration(t) is None

    def test_exact_one_minute(self):
        t = make_trade(
            entry_time=datetime(2025, 1, 1, 8, 0),
            exit_time=datetime(2025, 1, 1, 8, 1),
        )
        assert compute_trade_duration(t) == pytest.approx(1.0)

    def test_multi_day(self):
        t = make_trade(
            entry_time=datetime(2025, 1, 1, 8, 0),
            exit_time=datetime(2025, 1, 3, 8, 0),
        )
        assert compute_trade_duration(t) == pytest.approx(2880.0)


class TestExtractSession:
    def test_asian(self):
        dt = datetime(2025, 1, 1, 3, 0)
        assert extract_session(dt) == "asian"

    def test_asian_boundary_start(self):
        dt = datetime(2025, 1, 1, 0, 0)
        assert extract_session(dt) == "asian"

    def test_asian_boundary_end(self):
        dt = datetime(2025, 1, 1, 6, 59)
        assert extract_session(dt) == "asian"

    def test_london(self):
        dt = datetime(2025, 1, 1, 10, 0)
        assert extract_session(dt) == "london"

    def test_london_boundary_start(self):
        dt = datetime(2025, 1, 1, 7, 0)
        assert extract_session(dt) == "london"

    def test_london_boundary_end(self):
        dt = datetime(2025, 1, 1, 12, 59)
        assert extract_session(dt) == "london"

    def test_new_york(self):
        dt = datetime(2025, 1, 1, 15, 0)
        assert extract_session(dt) == "new_york"

    def test_new_york_boundary_start(self):
        dt = datetime(2025, 1, 1, 13, 0)
        assert extract_session(dt) == "new_york"

    def test_new_york_boundary_end(self):
        dt = datetime(2025, 1, 1, 20, 59)
        assert extract_session(dt) == "new_york"

    def test_off_hours(self):
        dt = datetime(2025, 1, 1, 21, 0)
        assert extract_session(dt) == "off_hours"

    def test_off_hours_late(self):
        dt = datetime(2025, 1, 1, 23, 59)
        assert extract_session(dt) == "off_hours"

    @pytest.mark.parametrize("hour,expected", [
        (0, "asian"),
        (6, "asian"),
        (7, "london"),
        (12, "london"),
        (13, "new_york"),
        (20, "new_york"),
        (21, "off_hours"),
        (23, "off_hours"),
    ])
    def test_all_hours(self, hour, expected):
        dt = datetime(2025, 1, 1, hour, 0)
        assert extract_session(dt) == expected


class TestExtractDayOfWeek:
    @pytest.mark.parametrize("dt,expected", [
        (datetime(2025, 3, 10, 12, 0), "monday"),
        (datetime(2025, 3, 11, 12, 0), "tuesday"),
        (datetime(2025, 3, 12, 12, 0), "wednesday"),
        (datetime(2025, 3, 13, 12, 0), "thursday"),
        (datetime(2025, 3, 14, 12, 0), "friday"),
        (datetime(2025, 3, 15, 12, 0), "saturday"),
        (datetime(2025, 3, 16, 12, 0), "sunday"),
    ])
    def test_all_days(self, dt, expected):
        assert extract_day_of_week(dt) == expected


class TestLoadTradesFromJsonl:
    def test_basic_load(self, tmp_path):
        path = tmp_path / "trades.jsonl"
        trades = [
            make_trade(trade_id="T1", symbol="EURUSD"),
            make_trade(trade_id="T2", symbol="GBPUSD"),
        ]
        with open(path, "w") as f:
            for t in trades:
                record = {
                    "trade_id": t.trade_id,
                    "symbol": t.symbol,
                    "direction": t.direction,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "stop_loss": t.stop_loss,
                    "take_profit": t.take_profit,
                    "position_size": t.position_size,
                    "account_balance": t.account_balance,
                    "risk_amount": t.risk_amount,
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat(),
                    "spread_at_entry": t.spread_at_entry,
                    "slippage_pips": t.slippage_pips,
                    "commission": t.commission,
                }
                f.write(json.dumps(record) + "\n")

        loaded = load_trades_from_jsonl(path)
        assert len(loaded) == 2
        assert loaded[0].trade_id == "T1"
        assert loaded[1].symbol == "GBPUSD"

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        loaded = load_trades_from_jsonl(path)
        assert loaded == []

    def test_blank_lines(self, tmp_path):
        path = tmp_path / "blanks.jsonl"
        path.write_text("\n\n\n")
        loaded = load_trades_from_jsonl(path)
        assert loaded == []

    def test_no_exit_time(self, tmp_path):
        path = tmp_path / "open.jsonl"
        record = {
            "trade_id": "OPEN1",
            "symbol": "EURUSD",
            "direction": "LONG",
            "entry_price": 1.1000,
            "exit_price": None,
            "stop_loss": 1.0950,
            "take_profit": 1.1100,
            "position_size": 1.0,
            "account_balance": 10000.0,
            "risk_amount": 50.0,
            "entry_time": datetime(2025, 1, 1, 8, 0).isoformat(),
            "exit_time": None,
            "spread_at_entry": 1.5,
            "slippage_pips": 0.0,
            "commission": 0.0,
        }
        with open(path, "w") as f:
            f.write(json.dumps(record) + "\n")
        loaded = load_trades_from_jsonl(path)
        assert len(loaded) == 1
        assert loaded[0].exit_time is None

    def test_malformed_numeric_record_does_not_abort_other_records(self, tmp_path):
        path = tmp_path / "mixed.jsonl"
        valid = {
            "trade_id": "VALID",
            "symbol": "EURUSD",
            "direction": "LONG",
            "entry_price": "1.1000",
            "exit_price": "1.1050",
            "stop_loss": "1.0950",
            "take_profit": "1.1100",
            "account_balance": "10000.0",
            "risk_amount": "50.0",
            "entry_time": "2025-01-01T08:00:00",
            "commission": "0.1",
        }
        malformed = {
            **valid,
            "trade_id": "BAD",
            "entry_price": "not-a-price",
        }
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(malformed) + "\n")
            f.write(json.dumps(valid) + "\n")

        loaded = load_trades_from_jsonl(path)

        assert [trade.trade_id for trade in loaded] == ["VALID"]
        assert loaded[0].entry_price == pytest.approx(1.1)
        assert loaded[0].account_balance == pytest.approx(10000.0)

    def test_non_finite_numeric_values_are_rejected(self, tmp_path):
        path = tmp_path / "non_finite.jsonl"
        record = {
            "trade_id": "NAN",
            "symbol": "EURUSD",
            "entry_price": "nan",
            "entry_time": "2025-01-01T08:00:00",
        }
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")

        assert load_trades_from_jsonl(path) == []

    def test_non_object_json_lines_are_skipped(self, tmp_path):
        path = tmp_path / "mixed_types.jsonl"
        valid = {
            "trade_id": "VALID",
            "symbol": "EURUSD",
            "entry_price": 1.1,
            "entry_time": "2025-01-01T08:00:00",
        }
        path.write_text(
            json.dumps(["not", "a", "trade"]) + "\n" + json.dumps(valid) + "\n",
            encoding="utf-8",
        )

        loaded = load_trades_from_jsonl(path)

        assert [trade.trade_id for trade in loaded] == ["VALID"]

    def test_fxbot_trade_snapshots_are_deduplicated_using_latest_state(self, tmp_path):
        path = tmp_path / "fxbot.jsonl"
        base = {
            "type": "trade",
            "timestamp": "2025-01-01T08:00:00+00:00",
            "payload": {
                "broker_trade_id": "BROKER-1",
                "instrument": "EUR_USD",
                "side": "LONG",
                "entry_price": 1.1,
                "entry_time": "2025-01-01T08:00:00+00:00",
                "stop_loss": 1.095,
                "state": "open",
            },
        }
        latest = {
            **base,
            "timestamp": "2025-01-01T09:00:00+00:00",
            "payload": {
                **base["payload"],
                "state": "closed",
                "exit_price": 1.105,
                "exit_time": "2025-01-01T09:00:00+00:00",
            },
        }
        path.write_text(json.dumps(base) + "\n" + json.dumps(latest) + "\n", encoding="utf-8")

        loaded = load_trades_from_jsonl(path)

        assert len(loaded) == 1
        assert loaded[0].trade_id == "BROKER-1"
        assert loaded[0].exit_price == pytest.approx(1.105)


class TestComputeRValues:
    def test_maps_closed_trades_in_order(self):
        trades = [
            make_trade(entry_price=1.1000, stop_loss=1.0950, exit_price=1.1050, direction="LONG"),
            make_trade(entry_price=1.1000, stop_loss=1.0950, exit_price=1.1100, direction="LONG"),
            make_trade(entry_price=1.1000, stop_loss=1.0950, exit_price=1.0975, direction="LONG"),
        ]
        r_values = compute_r_values(trades)
        assert r_values == pytest.approx([1.0, 2.0, -0.5])

    def test_skips_open_trades(self):
        trades = [
            make_trade(entry_price=1.1000, stop_loss=1.0950, exit_price=1.1050, direction="LONG"),
            make_trade(exit_price=None),
            make_trade(entry_price=1.1000, stop_loss=1.0950, exit_price=1.1100, direction="LONG"),
        ]
        r_values = compute_r_values(trades)
        assert r_values == pytest.approx([1.0, 2.0])

    def test_skips_zero_risk_trades(self):
        trades = [
            make_trade(entry_price=1.1000, stop_loss=1.0950, exit_price=1.1050, direction="LONG"),
            make_trade(entry_price=1.1000, stop_loss=1.1000, exit_price=1.1050),
        ]
        r_values = compute_r_values(trades)
        assert r_values == pytest.approx([1.0])

    def test_empty_input(self):
        assert compute_r_values([]) == []
