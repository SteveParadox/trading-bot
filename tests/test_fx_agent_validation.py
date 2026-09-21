from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from forex_agent.data.schemas import TradeRecord
from forex_agent.data.validation import DataQualityReport, TradeIssue, validate_trades


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


class TestValidateCleanData:
    def test_no_issues(self):
        trades = [
            make_trade(trade_id="T1"),
            make_trade(trade_id="T2", symbol="GBPUSD"),
        ]
        report = validate_trades(trades)
        assert report.total_trades == 2
        assert report.has_issues is False
        assert report.error_count == 0
        assert report.clean_trades == 2

    def test_single_clean_trade(self):
        report = validate_trades([make_trade()])
        assert report.total_trades == 1
        assert report.has_issues is False

    def test_empty_list(self):
        report = validate_trades([])
        assert report.total_trades == 0
        assert report.has_issues is False


class TestDuplicateIds:
    def test_duplicate_ids(self):
        trades = [
            make_trade(trade_id="DUP"),
            make_trade(trade_id="DUP"),
        ]
        report = validate_trades(trades)
        assert report.has_issues is True
        dup_issues = [i for i in report.issues if "Duplicate" in i.message]
        assert len(dup_issues) == 1
        assert dup_issues[0].severity == "error"

    def test_triple_duplicate(self):
        trades = [
            make_trade(trade_id="X"),
            make_trade(trade_id="X"),
            make_trade(trade_id="X"),
        ]
        report = validate_trades(trades)
        dup_issues = [i for i in report.issues if "Duplicate" in i.message]
        assert len(dup_issues) == 1
        assert "3 times" in dup_issues[0].message

    def test_no_duplicates(self):
        trades = [make_trade(trade_id="A"), make_trade(trade_id="B")]
        report = validate_trades(trades)
        dup_issues = [i for i in report.issues if "Duplicate" in i.message]
        assert len(dup_issues) == 0


class TestMissingExitPrice:
    def test_open_trade(self):
        trades = [make_trade(exit_price=None, exit_time=None)]
        report = validate_trades(trades)
        exit_issues = [i for i in report.issues if i.field == "exit_price"]
        assert len(exit_issues) == 1
        assert exit_issues[0].severity == "warning"
        assert "no exit price" in exit_issues[0].message.lower()


class TestImpossiblePrices:
    def test_long_exit_above_tp(self):
        trades = [
            make_trade(
                direction="LONG",
                entry_price=1.1000,
                exit_price=1.1150,
                take_profit=1.1100,
            )
        ]
        report = validate_trades(trades)
        tp_issues = [
            i for i in report.issues if "take profit" in i.message.lower()
        ]
        assert len(tp_issues) == 1
        assert tp_issues[0].severity == "error"

    def test_short_exit_below_tp(self):
        trades = [
            make_trade(
                direction="SHORT",
                entry_price=1.1000,
                exit_price=1.0850,
                stop_loss=1.1050,
                take_profit=1.0900,
            )
        ]
        report = validate_trades(trades)
        tp_issues = [
            i for i in report.issues if "take profit" in i.message.lower()
        ]
        assert len(tp_issues) == 1
        assert tp_issues[0].severity == "error"

    def test_long_exit_at_tp_is_fine(self):
        trades = [
            make_trade(
                direction="LONG",
                entry_price=1.1000,
                exit_price=1.1100,
                take_profit=1.1100,
            )
        ]
        report = validate_trades(trades)
        tp_issues = [
            i for i in report.issues if "take profit" in i.message.lower()
        ]
        assert len(tp_issues) == 0


class TestFutureTimestamps:
    def test_future_entry(self):
        future = datetime(2099, 1, 1, 0, 0)
        trades = [make_trade(entry_time=future, exit_time=None)]
        report = validate_trades(trades)
        future_issues = [i for i in report.issues if "future" in i.message.lower()]
        assert len(future_issues) == 1
        assert future_issues[0].severity == "error"

    def test_exit_before_entry(self):
        entry = datetime(2025, 6, 1, 12, 0)
        exit_t = datetime(2025, 6, 1, 10, 0)
        trades = [make_trade(entry_time=entry, exit_time=exit_t)]
        report = validate_trades(trades)
        time_issues = [
            i for i in report.issues if "exit time is before" in i.message.lower()
        ]
        assert len(time_issues) == 1
        assert time_issues[0].severity == "error"


class TestEdgeCases:
    def test_zero_entry_price(self):
        trades = [make_trade(entry_price=0.0)]
        report = validate_trades(trades)
        price_issues = [i for i in report.issues if i.field == "entry_price"]
        assert len(price_issues) == 1
        assert price_issues[0].severity == "error"

    def test_negative_risk(self):
        trades = [make_trade(risk_amount=-100.0)]
        report = validate_trades(trades)
        risk_issues = [i for i in report.issues if i.field == "risk_amount"]
        assert len(risk_issues) == 1
        assert risk_issues[0].severity == "error"

    def test_multiple_issues_same_trade(self):
        future = datetime(2099, 1, 1, 0, 0)
        trades = [
            make_trade(
                trade_id="BAD",
                entry_price=0.0,
                entry_time=future,
                risk_amount=-10.0,
                exit_price=None,
            )
        ]
        report = validate_trades(trades)
        bad_issues = [i for i in report.issues if i.trade_id == "BAD"]
        assert len(bad_issues) >= 3


class TestDataQualityReport:
    def test_clean_report(self):
        report = DataQualityReport(total_trades=5)
        assert report.has_issues is False
        assert report.error_count == 0
        assert report.warning_count == 0

    def test_report_with_errors_and_warnings(self):
        report = DataQualityReport(
            total_trades=10,
            issues=[
                TradeIssue("T1", "field", "error", "err1"),
                TradeIssue("T2", "field", "error", "err2"),
                TradeIssue("T3", "field", "warning", "warn1"),
            ],
        )
        assert report.has_issues is True
        assert report.error_count == 2
        assert report.warning_count == 1
