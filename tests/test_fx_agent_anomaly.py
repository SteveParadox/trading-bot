from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from forex_agent.data.schemas import TradeRecord
from forex_agent.analysis.anomaly_detection import (
    detect_distribution_shift,
    detect_drawdown_anomalies,
    detect_expectancy_shift,
    detect_loss_clustering,
    detect_losing_streaks,
    detect_mae_mfe_shift,
    detect_spread_anomalies,
    detect_trade_duration_anomalies,
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


class TestLosingStreaks:
    def test_no_streaks(self):
        trades = [
            make_trade(trade_id="T1", entry_price=1.1000, exit_price=1.1050),
            make_trade(trade_id="T2", entry_price=1.1000, exit_price=1.0950),
            make_trade(trade_id="T3", entry_price=1.1000, exit_price=1.1050),
        ]
        streaks = detect_losing_streaks(trades, min_streak=3)
        assert streaks == []

    def test_single_streak(self):
        trades = [
            make_trade(trade_id="W1", entry_price=1.1000, exit_price=1.1050),
            make_trade(trade_id="L1", entry_price=1.1000, exit_price=1.0950),
            make_trade(trade_id="L2", entry_price=1.1000, exit_price=1.0950),
            make_trade(trade_id="L3", entry_price=1.1000, exit_price=1.0950),
            make_trade(trade_id="W2", entry_price=1.1000, exit_price=1.1050),
        ]
        streaks = detect_losing_streaks(trades, min_streak=3)
        assert len(streaks) == 1
        assert streaks[0]["length"] == 3
        assert streaks[0]["start_index"] == 1
        assert streaks[0]["end_index"] == 3

    def test_two_streaks(self):
        trades = [
            make_trade(trade_id="L1", entry_price=1.1000, exit_price=1.0950),
            make_trade(trade_id="L2", entry_price=1.1000, exit_price=1.0950),
            make_trade(trade_id="L3", entry_price=1.1000, exit_price=1.0950),
            make_trade(trade_id="W1", entry_price=1.1000, exit_price=1.1050),
            make_trade(trade_id="L4", entry_price=1.1000, exit_price=1.0950),
            make_trade(trade_id="L5", entry_price=1.1000, exit_price=1.0950),
            make_trade(trade_id="L6", entry_price=1.1000, exit_price=1.0950),
            make_trade(trade_id="L7", entry_price=1.1000, exit_price=1.0950),
        ]
        streaks = detect_losing_streaks(trades, min_streak=3)
        assert len(streaks) == 2
        assert streaks[0]["length"] == 3
        assert streaks[1]["length"] == 4

    def test_streak_at_end(self):
        trades = [
            make_trade(trade_id="W1", entry_price=1.1000, exit_price=1.1050),
            make_trade(trade_id="L1", entry_price=1.1000, exit_price=1.0950),
            make_trade(trade_id="L2", entry_price=1.1000, exit_price=1.0950),
            make_trade(trade_id="L3", entry_price=1.1000, exit_price=1.0950),
        ]
        streaks = detect_losing_streaks(trades, min_streak=3)
        assert len(streaks) == 1
        assert streaks[0]["end_index"] == 3

    def test_streak_total_loss(self):
        trades = [
            make_trade(trade_id="L1", entry_price=1.1000, exit_price=1.0950),
            make_trade(trade_id="L2", entry_price=1.1000, exit_price=1.0950),
            make_trade(trade_id="L3", entry_price=1.1000, exit_price=1.0950),
        ]
        streaks = detect_losing_streaks(trades, min_streak=3)
        assert len(streaks) == 1
        assert streaks[0]["total_loss"] == pytest.approx(-1500.0)

    def test_min_streak_higher(self):
        trades = [
            make_trade(trade_id=f"L{i}", entry_price=1.1000, exit_price=1.0950)
            for i in range(5)
        ]
        streaks = detect_losing_streaks(trades, min_streak=10)
        assert streaks == []

    def test_empty_trades(self):
        streaks = detect_losing_streaks([], min_streak=3)
        assert streaks == []


class TestDrawdownAnomalies:
    def test_no_anomalies(self):
        trades = [
            make_trade(trade_id=f"W{i}", entry_price=1.1000, exit_price=1.1050)
            for i in range(5)
        ]
        anomalies = detect_drawdown_anomalies(trades, threshold_pct=0.10)
        assert anomalies == []

    def test_drawdown_detected(self):
        trades = [
            make_trade(trade_id="W1", entry_price=1.1000, exit_price=1.1100),
            make_trade(trade_id="L1", entry_price=1.1000, exit_price=1.0800),
            make_trade(trade_id="L2", entry_price=1.1000, exit_price=1.0750),
            make_trade(trade_id="L3", entry_price=1.1000, exit_price=1.0700),
            make_trade(trade_id="L4", entry_price=1.1000, exit_price=1.0650),
        ]
        anomalies = detect_drawdown_anomalies(trades, threshold_pct=0.10)
        assert len(anomalies) > 0
        assert all(a["drawdown_pct"] > 0.10 for a in anomalies)

    def test_low_threshold(self):
        trades = [
            make_trade(trade_id="L1", entry_price=1.1000, exit_price=1.0950),
            make_trade(trade_id="L2", entry_price=1.1000, exit_price=1.0900),
        ]
        anomalies = detect_drawdown_anomalies(trades, threshold_pct=0.01)
        assert len(anomalies) > 0

    def test_empty_trades(self):
        anomalies = detect_drawdown_anomalies([], threshold_pct=0.10)
        assert anomalies == []

    def test_anomaly_has_trade_info(self):
        trades = [
            make_trade(trade_id="W1", entry_price=1.1000, exit_price=1.1100),
            make_trade(trade_id="BIG_LOSS", entry_price=1.1000, exit_price=1.0600),
        ]
        anomalies = detect_drawdown_anomalies(trades, threshold_pct=0.01)
        assert len(anomalies) > 0
        assert "trade_id" in anomalies[0]
        assert "drawdown_pct" in anomalies[0]


class TestExpectancyShift:
    def test_no_shift(self):
        trades = [
            make_trade(trade_id=f"T{i}", entry_price=1.1000,
                       exit_price=1.1050 if i % 2 == 0 else 1.0950)
            for i in range(40)
        ]
        shifts = detect_expectancy_shift(trades, window=10, threshold=1000.0)
        assert shifts == []

    def test_shift_detected(self):
        trades = []
        for i in range(30):
            if i < 15:
                trades.append(make_trade(trade_id=f"W{i}", entry_price=1.1000, exit_price=1.1050))
            else:
                trades.append(make_trade(trade_id=f"L{i}", entry_price=1.1000, exit_price=1.0950))
        shifts = detect_expectancy_shift(trades, window=10, threshold=100.0)
        assert len(shifts) > 0
        assert all(s["direction"] in ["improvement", "degradation"] for s in shifts)

    def test_insufficient_data(self):
        trades = [
            make_trade(trade_id=f"T{i}", entry_price=1.1000, exit_price=1.1050)
            for i in range(5)
        ]
        shifts = detect_expectancy_shift(trades, window=20, threshold=0.5)
        assert shifts == []

    def test_shift_has_info(self):
        trades = []
        for i in range(30):
            if i < 15:
                trades.append(make_trade(trade_id=f"W{i}", entry_price=1.1000, exit_price=1.1050))
            else:
                trades.append(make_trade(trade_id=f"L{i}", entry_price=1.1000, exit_price=1.0950))
        shifts = detect_expectancy_shift(trades, window=10, threshold=100.0)
        for s in shifts:
            assert "previous_expectancy" in s
            assert "current_expectancy" in s
            assert "shift" in s


class TestLossClustering:
    def test_no_cluster_returns_empty(self):
        trades = [
            make_trade(trade_id=f"W{i}", entry_price=1.1000, exit_price=1.1050)
            for i in range(15)
        ]
        assert detect_loss_clustering(trades, window=10, ratio_threshold=0.8) == []

    def test_cluster_detected(self):
        trades = [
            make_trade(trade_id=f"L{i}", entry_price=1.1000, exit_price=1.0950)
            for i in range(10)
        ]
        anomalies = detect_loss_clustering(trades, window=10, ratio_threshold=0.8)
        assert len(anomalies) == 1
        assert anomalies[0]["loss_ratio"] >= 0.8
        assert anomalies[0]["loss_count"] == 10

    def test_merges_overlapping_windows(self):
        # 12 consecutive losses -> overlapping 10-windows all above threshold.
        trades = [
            make_trade(trade_id=f"L{i}", entry_price=1.1000, exit_price=1.0950)
            for i in range(12)
        ]
        anomalies = detect_loss_clustering(trades, window=10, ratio_threshold=0.8)
        # All windows merged into one contiguous run => exactly 1 anomaly
        assert len(anomalies) == 1
        assert anomalies[0]["loss_count"] == 10

    def test_insufficient_data(self):
        trades = [make_trade() for _ in range(5)]
        assert detect_loss_clustering(trades, window=10) == []

    def test_empty(self):
        assert detect_loss_clustering([], window=10) == []


class TestDistributionShift:
    def test_no_shift_when_insufficient(self):
        trades = [make_trade() for _ in range(40)]
        shifts = detect_distribution_shift(trades, window=30)
        assert shifts == []

    def test_disjoint_regions_produce_shift(self):
        trades = []
        # First half all winners (positive R), second half all losers (negative R)
        for i in range(40):
            if i < 20:
                trades.append(make_trade(trade_id=f"W{i}", entry_price=1.1000, exit_price=1.1100))
            else:
                trades.append(make_trade(trade_id=f"L{i}", entry_price=1.1000, exit_price=1.0950))
        shifts = detect_distribution_shift(trades, window=15)
        assert len(shifts) == 1
        assert shifts[0]["type"] == "distribution_shift"


class TestTradeDurationAnomalies:
    def test_no_duration_anomaly_for_uniform(self):
        base = datetime(2025, 3, 10, 8, 30, 0)
        trades = [
            make_trade(trade_id=f"T{i}",
                       entry_time=base + timedelta(minutes=30 * i),
                       exit_time=base + timedelta(minutes=30 * i + 60))
            for i in range(15)
        ]
        anomalies = detect_trade_duration_anomalies(trades, z_threshold=2.5)
        assert anomalies == []

    def test_outlier_duration_detected(self):
        base = datetime(2025, 3, 10, 8, 30, 0)
        trades = [
            make_trade(trade_id=f"T{i}",
                       entry_time=base + timedelta(minutes=30 * i),
                       exit_time=base + timedelta(minutes=30 * i + 60))
            for i in range(15)
        ]
        # Last trade held far longer
        trades.append(make_trade(
            trade_id="OUTLIER",
            entry_time=base,
            exit_time=base + timedelta(hours=200),
        ))
        anomalies = detect_trade_duration_anomalies(trades, z_threshold=2.5)
        assert any(a["trade_id"] == "OUTLIER" for a in anomalies)

    def test_insufficient_data(self):
        trades = [make_trade() for _ in range(5)]
        assert detect_trade_duration_anomalies(trades) == []


class TestSpreadAnomalies:
    def test_no_spread_anomaly(self):
        trades = [
            make_trade(trade_id=f"T{i}", spread_at_entry=1.5)
            for i in range(20)
        ]
        anomalies = detect_spread_anomalies(trades, z_threshold=2.5)
        assert anomalies == []

    def test_spike_detected(self):
        trades = [
            make_trade(trade_id=f"T{i}", spread_at_entry=1.5)
            for i in range(20)
        ]
        trades.append(make_trade(trade_id="SPIKE", spread_at_entry=50.0))
        anomalies = detect_spread_anomalies(trades, z_threshold=2.5)
        assert len(anomalies) == 1
        assert anomalies[0]["trade_id"] == "SPIKE"

    def test_insufficient_data(self):
        trades = [make_trade() for _ in range(5)]
        assert detect_spread_anomalies(trades) == []


class TestMaeMfeShift:
    def test_no_shift(self):
        trades = [
            make_trade(trade_id=f"T{i}", mae=0.002)
            for i in range(40)
        ]
        assert detect_mae_mfe_shift(trades) == []

    def test_mae_increase_detected(self):
        trades = []
        for i in range(40):
            mae = 0.002 if i < 20 else 0.005
            trades.append(make_trade(trade_id=f"T{i}", mae=mae))
        anomalies = detect_mae_mfe_shift(trades)
        assert len(anomalies) == 1
        assert anomalies[0]["type"] == "mae_mfe_shift"

    def test_insufficient_data(self):
        trades = [make_trade(mae=0.002) for _ in range(15)]
        assert detect_mae_mfe_shift(trades) == []
