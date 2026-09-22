from __future__ import annotations

from datetime import datetime, timedelta

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
from forex_agent.data.ingestion import (
    compute_r_multiple,
    compute_trade_duration,
    extract_day_of_week,
    extract_session,
)
from forex_agent.data.validation import validate_trades
from forex_agent.analysis.performance import (
    calculate_performance_metrics,
    calculate_rolling_expectancy,
)
from forex_agent.analysis.statistics import (
    analyze_r_distribution,
    performance_by_day,
    performance_by_instrument,
    performance_by_session,
)
from forex_agent.analysis.trade_forensics import (
    analyze_trade_failure,
    classify_failure,
    compute_counterfactuals,
)
from forex_agent.analysis.regime import (
    classify_trend,
    classify_volatility,
    detect_regime,
    performance_by_regime,
)
from forex_agent.analysis.anomaly_detection import (
    detect_drawdown_anomalies,
    detect_expectancy_shift,
    detect_losing_streaks,
)
from forex_agent.analysis.risk import (
    calculate_max_drawdown,
    calculate_position_sizing,
    calculate_tail_risk,
)
from forex_agent.analysis.execution import (
    analyze_execution_quality,
    analyze_slippage,
    analyze_spread_quality,
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
        regime=MarketRegime.UNKNOWN,
        follow_stop_loss=True,
        follow_take_profit=True,
    )
    defaults.update(overrides)
    return TradeRecord(**defaults)


def _make_realistic_dataset() -> list[TradeRecord]:
    base_time = datetime(2025, 1, 6, 8, 0)
    trades = []

    winning_trades = [
        ("W01", "EURUSD", "LONG", 1.1000, 1.1050, 1.0950, 1.1100, 1.5),
        ("W02", "GBPUSD", "SHORT", 1.2500, 1.2450, 1.2550, 1.2400, 2.0),
        ("W03", "USDJPY", "LONG", 110.000, 110.500, 109.500, 111.000, 1.0),
        ("W04", "EURUSD", "LONG", 1.1050, 1.1120, 1.1000, 1.1150, 1.5),
        ("W05", "AUDUSD", "LONG", 0.7500, 0.7530, 0.7480, 0.7560, 1.2),
        ("W06", "GBPUSD", "SHORT", 1.2400, 1.2350, 1.2450, 1.2300, 2.0),
        ("W07", "EURUSD", "LONG", 1.0980, 1.1030, 1.0930, 1.1080, 1.5),
        ("W08", "USDJPY", "SHORT", 110.500, 110.000, 111.000, 109.500, 1.0),
    ]

    losing_trades = [
        ("L01", "EURUSD", "LONG", 1.1100, 1.1050, 1.1050, 1.1150, 1.5),
        ("L02", "GBPUSD", "LONG", 1.2550, 1.2500, 1.2500, 1.2600, 2.5),
        ("L03", "EURUSD", "SHORT", 1.1000, 1.1030, 1.1050, 1.0950, 1.5),
        ("L04", "AUDUSD", "LONG", 0.7550, 0.7500, 0.7500, 0.7600, 1.2),
        ("L05", "EURUSD", "LONG", 1.1020, 1.0970, 1.0970, 1.1070, 1.5),
        ("L06", "USDJPY", "LONG", 110.000, 109.500, 109.500, 110.500, 1.0),
    ]

    regimes = [
        MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN,
        MarketRegime.RANGING, MarketRegime.HIGH_VOLATILITY,
    ]

    for i, (tid, sym, dir_, entry, exit_, sl, tp, spread) in enumerate(winning_trades):
        entry_time = base_time + timedelta(hours=i * 3)
        exit_time = entry_time + timedelta(minutes=60 + i * 15)
        trades.append(make_trade(
            trade_id=tid,
            symbol=sym,
            direction=dir_,
            entry_price=entry,
            exit_price=exit_,
            stop_loss=sl,
            take_profit=tp,
            spread_at_entry=spread,
            entry_time=entry_time,
            exit_time=exit_time,
            account_balance=10000.0,
            regime=regimes[i % len(regimes)],
        ))

    for i, (tid, sym, dir_, entry, exit_, sl, tp, spread) in enumerate(losing_trades):
        entry_time = base_time + timedelta(hours=(len(winning_trades) + i) * 3)
        exit_time = entry_time + timedelta(minutes=45 + i * 10)
        trades.append(make_trade(
            trade_id=tid,
            symbol=sym,
            direction=dir_,
            entry_price=entry,
            exit_price=exit_,
            stop_loss=sl,
            take_profit=tp,
            spread_at_entry=spread,
            entry_time=entry_time,
            exit_time=exit_time,
            account_balance=10000.0,
            regime=regimes[i % len(regimes)],
            follow_stop_loss=True if i != 2 else False,
        ))

    return trades


class TestFullPipeline:
    def test_validation_passes(self):
        trades = _make_realistic_dataset()
        report = validate_trades(trades)
        assert report.total_trades == 14
        assert report.error_count == 0

    def test_performance_metrics(self):
        trades = _make_realistic_dataset()
        metrics = calculate_performance_metrics(trades)
        assert metrics.total_trades == 14
        assert metrics.winning_trades == 8
        assert metrics.losing_trades == 6
        assert 0 < metrics.win_rate < 1
        assert metrics.total_pnl != 0

    def test_r_distribution(self):
        trades = _make_realistic_dataset()
        r_stats = analyze_r_distribution(trades)
        assert r_stats["count"] > 0
        assert r_stats["min"] <= r_stats["mean"] <= r_stats["max"]

    def test_instrument_breakdown(self):
        trades = _make_realistic_dataset()
        by_inst = performance_by_instrument(trades)
        assert "EURUSD" in by_inst
        assert "GBPUSD" in by_inst
        assert "USDJPY" in by_inst
        for symbol, stats in by_inst.items():
            assert stats["total_trades"] > 0
            assert 0 <= stats["win_rate"] <= 1

    def test_session_breakdown(self):
        trades = _make_realistic_dataset()
        by_session = performance_by_session(trades)
        assert len(by_session) > 0

    def test_day_breakdown(self):
        trades = _make_realistic_dataset()
        by_day = performance_by_day(trades)
        assert len(by_day) > 0

    def test_regime_breakdown(self):
        trades = _make_realistic_dataset()
        by_regime = performance_by_regime(trades)
        assert len(by_regime) > 0

    def test_losing_streaks(self):
        trades = _make_realistic_dataset()
        streaks = detect_losing_streaks(trades, min_streak=2)
        assert isinstance(streaks, list)

    def test_drawdown(self):
        trades = _make_realistic_dataset()
        dd = calculate_max_drawdown(trades)
        assert dd["max_drawdown"] >= 0
        assert dd["max_drawdown_pct"] >= 0

    def test_tail_risk(self):
        trades = _make_realistic_dataset()
        tail = calculate_tail_risk(trades)
        assert tail["worst_trade"] <= 0

    def test_spread_quality(self):
        trades = _make_realistic_dataset()
        spread = analyze_spread_quality(trades)
        assert spread["avg_spread"] > 0
        assert spread["max_spread"] >= spread["avg_spread"]

    def test_slippage_analysis(self):
        trades = _make_realistic_dataset()
        slip = analyze_slippage(trades)
        assert slip["avg_slippage"] >= 0

    def test_execution_quality(self):
        trades = _make_realistic_dataset()
        exec_q = analyze_execution_quality(trades)
        assert "spread" in exec_q
        assert "slippage" in exec_q
        assert "total_execution_cost" in exec_q


class TestForensicsOnRealisticData:
    def test_analyze_losing_trades(self):
        trades = _make_realistic_dataset()
        losers = [t for t in trades if t.exit_price is not None and not t.is_winner]
        for t in losers:
            analysis = analyze_trade_failure(t)
            assert isinstance(analysis.failure_category, FailureCategory)
            assert isinstance(analysis.confidence, ConfidenceLevel)
            assert analysis.recommended_action != ""


class TestRollingExpectancyPipeline:
    def test_rolling_with_dataset(self):
        trades = _make_realistic_dataset()
        rolling = calculate_rolling_expectancy(trades, window=5)
        assert len(rolling) == 14
        assert all(isinstance(v, float) for v in rolling)


class TestAnomalyDetectionPipeline:
    def test_drawdown_anomalies(self):
        trades = _make_realistic_dataset()
        anomalies = detect_drawdown_anomalies(trades, threshold_pct=0.05)
        assert isinstance(anomalies, list)


class TestRiskMetricsPipeline:
    def test_position_sizing(self):
        result = calculate_position_sizing(
            account_balance=10000.0,
            risk_pct=1.0,
            entry_price=1.1000,
            stop_loss=1.0950,
        )
        assert result["risk_amount"] == pytest.approx(100.0)
        assert result["position_size"] > 0
        assert result["risk_pips"] == pytest.approx(0.0050)

    def test_position_sizing_zero_risk_pips(self):
        result = calculate_position_sizing(
            account_balance=10000.0,
            risk_pct=1.0,
            entry_price=1.1000,
            stop_loss=1.1000,
        )
        assert result["position_size"] == 0.0


class TestEdgeCaseDatasets:
    def test_single_trade(self):
        trades = [make_trade()]
        metrics = calculate_performance_metrics(trades)
        assert metrics.total_trades == 1
        assert metrics.win_rate == pytest.approx(1.0)

    def test_single_trade_open(self):
        trades = [make_trade(exit_price=None, exit_time=None)]
        metrics = calculate_performance_metrics(trades)
        assert metrics.total_trades == 0

    def test_all_identical_trades(self):
        trades = [make_trade(trade_id=f"T{i}") for i in range(10)]
        metrics = calculate_performance_metrics(trades)
        assert metrics.total_trades == 10
        assert metrics.win_rate == pytest.approx(1.0)

    def test_alternating_wins_losses(self):
        trades = [
            make_trade(
                trade_id=f"T{i}",
                entry_price=1.1000,
                exit_price=1.1050 if i % 2 == 0 else 1.0950,
                stop_loss=1.0950,
            )
            for i in range(20)
        ]
        metrics = calculate_performance_metrics(trades)
        assert metrics.win_rate == pytest.approx(0.5)
        assert metrics.total_pnl == pytest.approx(0.0, abs=0.01)


class TestCrossModuleInteractions:
    def test_r_multiple_feeds_into_r_distribution(self):
        trades = _make_realistic_dataset()
        closed = [t for t in trades if t.exit_price is not None]
        r_values = [compute_r_multiple(t) for t in closed]
        r_values = [r for r in r_values if r is not None]
        assert len(r_values) > 0
        r_stats = analyze_r_distribution(trades)
        assert r_stats["count"] == len(r_values)

    def test_validation_then_analysis(self):
        trades = _make_realistic_dataset()
        report = validate_trades(trades)
        assert report.error_count == 0
        metrics = calculate_performance_metrics(trades)
        assert metrics.total_trades == report.total_trades

    def test_regime_detection_feeds_into_performance(self):
        prices = [1.1000 + i * 0.001 for i in range(60)]
        regime = detect_regime(prices)
        trades = [make_trade(regime=regime) for _ in range(5)]
        by_regime = performance_by_regime(trades)
        assert regime.value in by_regime


class TestComputeRMultipleInContext:
    def test_import_from_ingestion(self):
        t = make_trade(entry_price=1.1000, stop_loss=1.0950, exit_price=1.1050)
        r = compute_r_multiple(t)
        assert r == pytest.approx(1.0)


class TestTradeDurationInContext:
    def test_import_from_ingestion(self):
        t = make_trade(
            entry_time=datetime(2025, 1, 1, 8, 0),
            exit_time=datetime(2025, 1, 1, 10, 30),
        )
        dur = compute_trade_duration(t)
        assert dur == pytest.approx(150.0)
