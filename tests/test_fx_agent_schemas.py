from __future__ import annotations

from datetime import datetime

import pytest

from forex_agent.data.schemas import (
    ConfidenceLevel,
    FailureCategory,
    MarketRegime,
    PerformanceMetrics,
    StrategyHealthScore,
    TradeFailureAnalysis,
    TradeRecord,
)


class TestTradeRecord:
    def test_default_creation(self):
        t = TradeRecord()
        assert t.symbol == "EURUSD"
        assert t.direction == "LONG"
        assert t.entry_price == 1.1000
        assert t.exit_price is None
        assert t.position_size == 1.0
        assert t.account_balance == 10000.0
        assert t.trade_id  # auto-generated, non-empty

    def test_is_winner_long_win(self):
        t = TradeRecord(entry_price=1.1000, exit_price=1.1050, direction="LONG")
        assert t.is_winner is True

    def test_is_winner_long_loss(self):
        t = TradeRecord(entry_price=1.1000, exit_price=1.0950, direction="LONG")
        assert t.is_winner is False

    def test_is_winner_short_win(self):
        t = TradeRecord(entry_price=1.1000, exit_price=1.0950, direction="SHORT")
        assert t.is_winner is True

    def test_is_winner_short_loss(self):
        t = TradeRecord(entry_price=1.1000, exit_price=1.1050, direction="SHORT")
        assert t.is_winner is False

    def test_is_winner_no_exit(self):
        t = TradeRecord(exit_price=None)
        assert t.is_winner is False

    def test_pnl_long(self):
        t = TradeRecord(entry_price=1.1000, exit_price=1.1050, direction="LONG", position_size=1.0)
        assert t.pnl == pytest.approx(500.0)

    def test_pnl_short(self):
        t = TradeRecord(entry_price=1.1000, exit_price=1.0950, direction="SHORT", position_size=1.0)
        assert t.pnl == pytest.approx(500.0)

    def test_pnl_no_exit(self):
        t = TradeRecord(exit_price=None)
        assert t.pnl == 0.0

    def test_pnl_two_lots(self):
        t = TradeRecord(entry_price=1.1000, exit_price=1.1050, direction="LONG", position_size=2.0)
        assert t.pnl == pytest.approx(1000.0)

    def test_custom_fields(self):
        t = TradeRecord(
            trade_id="T001",
            symbol="GBPUSD",
            direction="SHORT",
            entry_price=1.2500,
            exit_price=1.2400,
            stop_loss=1.2600,
            take_profit=1.2300,
            spread_at_entry=2.0,
            slippage_pips=0.5,
        )
        assert t.trade_id == "T001"
        assert t.symbol == "GBPUSD"
        assert t.spread_at_entry == 2.0


class TestTradeFailureAnalysis:
    def test_defaults(self):
        a = TradeFailureAnalysis(
            trade_id="T1",
            failure_category=FailureCategory.MARKET_CONDITION,
            confidence=ConfidenceLevel.MEDIUM,
            description="Market moved against position",
        )
        assert a.trade_id == "T1"
        assert a.contributing_factors == []
        assert a.counterfactual == ""
        assert a.recommended_action == ""


class TestFailureCategory:
    def test_values(self):
        assert FailureCategory.INVALID_SETUP.value == "invalid_setup"
        assert FailureCategory.EXECUTION_ERROR.value == "execution_error"
        assert FailureCategory.UNKNOWN.value == "unknown"

    def test_all_members(self):
        members = list(FailureCategory)
        assert len(members) == 12


class TestConfidenceLevel:
    def test_values(self):
        assert ConfidenceLevel.LOW.value == "low"
        assert ConfidenceLevel.MEDIUM.value == "medium"
        assert ConfidenceLevel.HIGH.value == "high"


class TestMarketRegime:
    def test_all_members(self):
        members = list(MarketRegime)
        assert len(members) == 6
        assert MarketRegime.TRENDING_UP.value == "trending_up"


class TestStrategyHealthScore:
    def test_defaults(self):
        s = StrategyHealthScore()
        assert s.score == 0.0
        assert s.components == {}
        assert s.grade == "F"
        assert s.recommendations == []

    def test_compute_grade_a_plus(self):
        s = StrategyHealthScore(score=95)
        assert s.compute_grade() == "A+"

    def test_compute_grade_a(self):
        s = StrategyHealthScore(score=85)
        assert s.compute_grade() == "A"

    def test_compute_grade_b(self):
        s = StrategyHealthScore(score=75)
        assert s.compute_grade() == "B"

    def test_compute_grade_c(self):
        s = StrategyHealthScore(score=65)
        assert s.compute_grade() == "C"

    def test_compute_grade_d(self):
        s = StrategyHealthScore(score=55)
        assert s.compute_grade() == "D"

    def test_compute_grade_f(self):
        s = StrategyHealthScore(score=40)
        assert s.compute_grade() == "F"

    def test_compute_grade_boundary(self):
        assert StrategyHealthScore(score=90).compute_grade() == "A+"
        assert StrategyHealthScore(score=80).compute_grade() == "A"
        assert StrategyHealthScore(score=70).compute_grade() == "B"
        assert StrategyHealthScore(score=60).compute_grade() == "C"
        assert StrategyHealthScore(score=50).compute_grade() == "D"


class TestPerformanceMetrics:
    def test_defaults(self):
        m = PerformanceMetrics()
        assert m.total_trades == 0
        assert m.win_rate == 0.0
        assert m.profit_factor == 0.0
        assert m.max_drawdown == 0.0
