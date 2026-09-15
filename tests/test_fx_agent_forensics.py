from __future__ import annotations

from datetime import datetime

import pytest

from forex_agent.data.schemas import (
    ConfidenceLevel,
    FailureCategory,
    MarketRegime,
    TradeRecord,
)
from forex_agent.analysis.trade_forensics import (
    analyze_trade_failure,
    classify_failure,
    compute_counterfactuals,
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
        follow_stop_loss=True,
        follow_take_profit=True,
    )
    defaults.update(overrides)
    return TradeRecord(**defaults)


class TestClassifyFailure:
    def test_valid_loss_market_condition(self):
        t = make_trade(
            trade_id="T1",
            direction="LONG",
            entry_price=1.1000,
            exit_price=1.0950,
            stop_loss=1.0950,
            spread_at_entry=1.0,
            follow_stop_loss=True,
            follow_take_profit=True,
        )
        cat = classify_failure(t)
        # Clean full-stop hit with no defects -> valid strategy loss.
        assert cat == FailureCategory.VALID_STRATEGY_LOSS

    def test_execution_error_high_spread(self):
        t = make_trade(
            trade_id="T2",
            direction="LONG",
            entry_price=1.1000,
            exit_price=1.0950,
            stop_loss=1.0950,
            spread_at_entry=8.0,
        )
        cat = classify_failure(t, context={"avg_spread": 8.0})
        assert cat == FailureCategory.EXECUTION_ERROR

    def test_risk_mismanagment_no_follow(self):
        t = make_trade(
            trade_id="T3",
            direction="LONG",
            entry_price=1.1000,
            exit_price=1.0950,
            stop_loss=1.0950,
            follow_stop_loss=False,
            follow_take_profit=False,
        )
        cat = classify_failure(t)
        assert cat == FailureCategory.RISK_MISMANAGEMENT

    def test_risk_mismanagement_deep_loss(self):
        t = make_trade(
            trade_id="T4",
            direction="LONG",
            entry_price=1.1000,
            exit_price=1.0900,
            stop_loss=1.0950,
            follow_stop_loss=True,
        )
        cat = classify_failure(t)
        assert cat == FailureCategory.RISK_MISMANAGEMENT

    def test_regime_mismatch(self):
        t = make_trade(
            trade_id="T5",
            direction="LONG",
            entry_price=1.1000,
            exit_price=1.0980,
            stop_loss=1.0950,
            regime=MarketRegime.TRENDING_UP,
        )
        cat = classify_failure(t, context={"regime": "trending_down"})
        assert cat == FailureCategory.REGIME_MISMATCH

    def test_invalid_setup_against_trend(self):
        t = make_trade(
            trade_id="T6",
            direction="LONG",
            entry_price=1.1000,
            exit_price=1.0990,
            stop_loss=1.0950,
        )
        cat = classify_failure(t, context={"traded_against_trend": True})
        assert cat == FailureCategory.INVALID_SETUP


class TestAnalyzeTradeFailure:
    def test_winner_returns_unknown(self):
        t = make_trade(entry_price=1.1000, exit_price=1.1050)
        analysis = analyze_trade_failure(t)
        assert analysis.description == "This trade was a winner, not a failure."

    def test_no_exit_returns_unknown(self):
        t = make_trade(exit_price=None, exit_time=None)
        analysis = analyze_trade_failure(t)
        assert "no exit price" in analysis.description.lower()

    def test_loser_has_category(self):
        t = make_trade(
            entry_price=1.1000,
            exit_price=1.0950,
            stop_loss=1.0950,
        )
        analysis = analyze_trade_failure(t)
        assert analysis.failure_category in [
            FailureCategory.MARKET_CONDITION,
            FailureCategory.VALID_STRATEGY_LOSS,
            FailureCategory.SIGNAL_FAILURE,
            FailureCategory.POOR_ENTRY,
            FailureCategory.RULE_VIOLATION,
            FailureCategory.DATA_QUALITY_FAILURE,
            FailureCategory.EXECUTION_ERROR,
            FailureCategory.RISK_MISMANAGEMENT,
            FailureCategory.REGIME_MISMATCH,
            FailureCategory.INVALID_SETUP,
        ]

    def test_analysis_has_contributing_factors(self):
        t = make_trade(
            entry_price=1.1000,
            exit_price=1.0950,
            stop_loss=1.0950,
            spread_at_entry=6.0,
            slippage_pips=2.0,
        )
        analysis = analyze_trade_failure(t, context={"avg_spread": 6.0})
        assert len(analysis.contributing_factors) > 0

    def test_analysis_has_recommended_action(self):
        t = make_trade(
            entry_price=1.1000,
            exit_price=1.0950,
            stop_loss=1.0950,
            follow_stop_loss=False,
            follow_take_profit=False,
        )
        analysis = analyze_trade_failure(t)
        assert analysis.recommended_action != ""

    def test_analysis_has_trade_id(self):
        t = make_trade(trade_id="FORENSIC1", entry_price=1.1000, exit_price=1.0950, stop_loss=1.0950)
        analysis = analyze_trade_failure(t)
        assert analysis.trade_id == "FORENSIC1"


class TestComputeCounterfactuals:
    def test_high_spread_counterfactual(self):
        t = make_trade(
            entry_price=1.1000,
            exit_price=1.0950,
            stop_loss=1.0950,
            spread_at_entry=5.0,
        )
        cf = compute_counterfactuals(t)
        assert "spread" in cf.lower() or "spread" in cf

    def test_no_follow_sl_counterfactual(self):
        t = make_trade(
            entry_price=1.1000,
            exit_price=1.0900,
            stop_loss=1.0950,
            follow_stop_loss=False,
        )
        cf = compute_counterfactuals(t)
        assert "stop loss" in cf.lower()

    def test_no_counterfactual_for_perfect_trade(self):
        t = make_trade(
            entry_price=1.1000,
            exit_price=1.0950,
            stop_loss=1.0950,
            spread_at_entry=1.0,
            follow_stop_loss=True,
        )
        cf = compute_counterfactuals(t)
        # Even with no obvious defect, we provide a stop-variation analysis.
        assert cf != ""


class TestConfidenceLevel:
    def test_high_confidence_many_factors(self):
        t = make_trade(
            entry_price=1.1000,
            exit_price=1.0950,
            stop_loss=1.0950,
            spread_at_entry=6.0,
            slippage_pips=2.0,
            follow_stop_loss=False,
        )
        analysis = analyze_trade_failure(t)
        assert analysis.confidence == ConfidenceLevel.HIGH

    def test_low_confidence_few_factors(self):
        t = make_trade(
            entry_price=1.1000,
            exit_price=1.0980,
            stop_loss=1.0950,
            spread_at_entry=1.0,
        )
        analysis = analyze_trade_failure(t)
        assert analysis.confidence in [ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM]
