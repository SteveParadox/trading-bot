from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from forex_agent.data.schemas import TradeRecord
from forex_agent.analysis.performance import (
    calculate_performance_metrics,
    calculate_rolling_expectancy,
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


class TestAllWins:
    def test_all_wins(self):
        trades = [
            make_trade(trade_id="W1", entry_price=1.1000, exit_price=1.1050),
            make_trade(trade_id="W2", entry_price=1.1000, exit_price=1.1100),
            make_trade(trade_id="W3", entry_price=1.1000, exit_price=1.1025),
        ]
        m = calculate_performance_metrics(trades)
        assert m.total_trades == 3
        assert m.winning_trades == 3
        assert m.losing_trades == 0
        assert m.win_rate == pytest.approx(1.0)
        assert m.total_pnl == pytest.approx(1750.0)


class TestAllLosses:
    def test_all_losses(self):
        trades = [
            make_trade(trade_id="L1", entry_price=1.1000, exit_price=1.0950),
            make_trade(trade_id="L2", entry_price=1.1000, exit_price=1.0925),
            make_trade(trade_id="L3", entry_price=1.1000, exit_price=1.0975),
        ]
        m = calculate_performance_metrics(trades)
        assert m.total_trades == 3
        assert m.winning_trades == 0
        assert m.losing_trades == 3
        assert m.win_rate == pytest.approx(0.0)


class TestMixedTrades:
    def test_mixed(self):
        trades = [
            make_trade(trade_id="W1", entry_price=1.1000, exit_price=1.1050),
            make_trade(trade_id="L1", entry_price=1.1000, exit_price=1.0950),
            make_trade(trade_id="W2", entry_price=1.1000, exit_price=1.1100),
            make_trade(trade_id="L2", entry_price=1.1000, exit_price=1.0925),
        ]
        m = calculate_performance_metrics(trades)
        assert m.total_trades == 4
        assert m.winning_trades == 2
        assert m.losing_trades == 2
        assert m.win_rate == pytest.approx(0.5)
        assert m.total_pnl == pytest.approx(250.0, abs=0.01)


class TestWinRate:
    def test_win_rate_75pct(self):
        trades = [
            make_trade(trade_id=f"T{i}", entry_price=1.1000,
                       exit_price=1.1050 if i < 3 else 1.0950)
            for i in range(4)
        ]
        m = calculate_performance_metrics(trades)
        assert m.win_rate == pytest.approx(0.75)

    def test_win_rate_single_trade(self):
        trades = [make_trade(entry_price=1.1000, exit_price=1.1050)]
        m = calculate_performance_metrics(trades)
        assert m.win_rate == pytest.approx(1.0)


class TestProfitFactor:
    def test_profit_factor_balanced(self):
        trades = [
            make_trade(trade_id="W1", entry_price=1.1000, exit_price=1.1050),
            make_trade(trade_id="L1", entry_price=1.1000, exit_price=1.0950),
        ]
        m = calculate_performance_metrics(trades)
        assert m.profit_factor == pytest.approx(1.0)

    def test_profit_factor_all_profit(self):
        trades = [
            make_trade(trade_id="W1", entry_price=1.1000, exit_price=1.1050),
            make_trade(trade_id="W2", entry_price=1.1000, exit_price=1.1100),
        ]
        m = calculate_performance_metrics(trades)
        assert m.profit_factor == float("inf")

    def test_profit_factor_losing(self):
        trades = [
            make_trade(trade_id="W1", entry_price=1.1000, exit_price=1.1025),
            make_trade(trade_id="L1", entry_price=1.1000, exit_price=1.0950),
        ]
        m = calculate_performance_metrics(trades)
        assert m.profit_factor == pytest.approx(0.5)

    def test_profit_factor_all_zero_pnl(self):
        t = make_trade(exit_price=1.1000)
        m = calculate_performance_metrics([t])
        assert m.profit_factor == pytest.approx(0.0)


class TestMaxDrawdown:
    def test_no_drawdown(self):
        trades = [
            make_trade(trade_id="W1", entry_price=1.1000, exit_price=1.1050),
            make_trade(trade_id="W2", entry_price=1.1000, exit_price=1.1100),
        ]
        m = calculate_performance_metrics(trades)
        assert m.max_drawdown == pytest.approx(0.0)
        assert m.max_drawdown_pct == pytest.approx(0.0)

    def test_known_drawdown(self):
        trades = [
            make_trade(trade_id="W1", entry_price=1.1000, exit_price=1.1100),
            make_trade(trade_id="L1", entry_price=1.1000, exit_price=1.0900),
            make_trade(trade_id="L2", entry_price=1.1000, exit_price=1.0850),
            make_trade(trade_id="W2", entry_price=1.1000, exit_price=1.1100),
        ]
        m = calculate_performance_metrics(trades)
        assert m.max_drawdown > 0
        assert m.max_drawdown_pct > 0

    def test_drawdown_sequence(self):
        trades = [
            make_trade(trade_id="W1", entry_price=1.1000, exit_price=1.1050),
            make_trade(trade_id="L1", entry_price=1.1000, exit_price=1.0950),
            make_trade(trade_id="L2", entry_price=1.1000, exit_price=1.0900),
            make_trade(trade_id="L3", entry_price=1.1000, exit_price=1.0850),
        ]
        m = calculate_performance_metrics(trades)
        assert m.max_drawdown == pytest.approx(3000.0, abs=0.01)


class TestExpectancy:
    def test_positive_expectancy(self):
        trades = [
            make_trade(trade_id=f"W{i}", entry_price=1.1000, exit_price=1.1050)
            for i in range(6)
        ] + [
            make_trade(trade_id=f"L{i}", entry_price=1.1000, exit_price=1.0980)
            for i in range(4)
        ]
        m = calculate_performance_metrics(trades)
        assert m.expectancy > 0

    def test_zero_expectancy(self):
        trades = [
            make_trade(trade_id="W1", entry_price=1.1000, exit_price=1.1050),
            make_trade(trade_id="W2", entry_price=1.1000, exit_price=1.1050),
            make_trade(trade_id="L1", entry_price=1.1000, exit_price=1.0950),
            make_trade(trade_id="L2", entry_price=1.1000, exit_price=1.0950),
        ]
        m = calculate_performance_metrics(trades)
        assert m.expectancy == pytest.approx(0.0, abs=0.01)


class TestRMultipleStats:
    def test_avg_r_multiple(self):
        trades = [
            make_trade(trade_id="T1", entry_price=1.1000, stop_loss=1.0950, exit_price=1.1050),
            make_trade(trade_id="T2", entry_price=1.1000, stop_loss=1.0950, exit_price=1.1000),
        ]
        m = calculate_performance_metrics(trades)
        assert m.avg_r_multiple == pytest.approx(0.5)


class TestEmptyTrades:
    def test_empty_list(self):
        m = calculate_performance_metrics([])
        assert m.total_trades == 0
        assert m.win_rate == 0.0

    def test_all_open_trades(self):
        trades = [make_trade(exit_price=None, exit_time=None)]
        m = calculate_performance_metrics(trades)
        assert m.total_trades == 0


class TestConsecutiveCounts:
    def test_consecutive_wins(self):
        trades = [
            make_trade(trade_id="W1", entry_price=1.1000, exit_price=1.1050),
            make_trade(trade_id="W2", entry_price=1.1000, exit_price=1.1050),
            make_trade(trade_id="W3", entry_price=1.1000, exit_price=1.1050),
            make_trade(trade_id="L1", entry_price=1.1000, exit_price=1.0950),
        ]
        m = calculate_performance_metrics(trades)
        assert m.consecutive_wins == 3
        assert m.consecutive_losses == 1

    def test_consecutive_losses(self):
        trades = [
            make_trade(trade_id="L1", entry_price=1.1000, exit_price=1.0950),
            make_trade(trade_id="L2", entry_price=1.1000, exit_price=1.0950),
            make_trade(trade_id="W1", entry_price=1.1000, exit_price=1.1050),
            make_trade(trade_id="L3", entry_price=1.1000, exit_price=1.0950),
        ]
        m = calculate_performance_metrics(trades)
        assert m.consecutive_wins == 1
        assert m.consecutive_losses == 2


class TestRollingExpectancy:
    def test_rolling_expectancy_basic(self):
        trades = [
            make_trade(trade_id=f"T{i}", entry_price=1.1000,
                       exit_price=1.1050 if i % 2 == 0 else 1.0950)
            for i in range(10)
        ]
        rolling = calculate_rolling_expectancy(trades, window=5)
        assert len(rolling) == 10
        assert all(isinstance(v, float) for v in rolling)

    def test_rolling_expectancy_window_larger_than_data(self):
        trades = [
            make_trade(trade_id="T1", entry_price=1.1000, exit_price=1.1050),
        ]
        rolling = calculate_rolling_expectancy(trades, window=20)
        assert len(rolling) == 1

    def test_rolling_expectancy_all_wins(self):
        trades = [
            make_trade(trade_id=f"W{i}", entry_price=1.1000, exit_price=1.1050)
            for i in range(5)
        ]
        rolling = calculate_rolling_expectancy(trades, window=3)
        assert all(v > 0 for v in rolling)

    def test_rolling_expectancy_empty(self):
        rolling = calculate_rolling_expectancy([], window=5)
        assert rolling == []
