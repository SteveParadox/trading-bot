from __future__ import annotations

from datetime import datetime

import pytest

from forex_agent.analysis.mae_mfe import (
    calculate_mfe_mae,
    efficiency_ratio,
    mfe_mae_in_r,
    price_recovered_from_mae,
    stopped_out_at_full_distance,
    was_take_profit_touched,
)
from forex_agent.data.schemas import TradeRecord


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


class TestCalculateMfeMae:
    def test_empty_path(self):
        assert calculate_mfe_mae(1.1000, "LONG", []) == (None, None)

    def test_long_mfe_mae(self):
        mfe, mae = calculate_mfe_mae(1.1000, "LONG", [1.1050, 1.1120, 1.0980])
        assert mfe == pytest.approx(0.0120)
        assert mae == pytest.approx(0.0020)

    def test_short_mfe_mae(self):
        mfe, mae = calculate_mfe_mae(1.1000, "SHORT", [1.0950, 1.0880, 1.1020])
        assert mfe == pytest.approx(0.0120)
        assert mae == pytest.approx(0.0020)

    def test_no_adverse_excursion(self):
        mfe, mae = calculate_mfe_mae(1.1000, "LONG", [1.1050, 1.1100])
        assert mfe == pytest.approx(0.0100)
        assert mae == pytest.approx(0.0)


class TestMfeMaeInR:
    def test_returns_r_terms(self):
        trade = make_trade(entry_price=1.1000, stop_loss=1.0950, mfe=0.01, mae=0.005)
        mfe_r, mae_r = mfe_mae_in_r(trade)
        assert mfe_r == pytest.approx(2.0)   # risk = 0.005
        assert mae_r == pytest.approx(1.0)

    def test_no_stop_returns_none(self):
        trade = make_trade(entry_price=1.1000, stop_loss=None, mfe=0.01, mae=0.005)
        assert mfe_mae_in_r(trade) == (None, None)


class TestWasTakeProfitTouched:
    def test_long_tp_touched(self):
        trade = make_trade(direction="LONG", entry_price=1.1000,
                           take_profit=1.1100, mfe=0.0120)
        assert was_take_profit_touched(trade) is True

    def test_long_tp_not_touched(self):
        trade = make_trade(direction="LONG", entry_price=1.1000,
                           take_profit=1.1100, mfe=0.0050)
        assert was_take_profit_touched(trade) is False

    def test_short_tp_touched(self):
        trade = make_trade(direction="SHORT", entry_price=1.1000,
                           take_profit=1.0900, mfe=0.0120)
        assert was_take_profit_touched(trade) is True

    def test_missing_data_returns_none(self):
        trade = make_trade(mfe=None)
        assert was_take_profit_touched(trade) is None


class TestStoppedOutAtFullDistance:
    def test_long_stopped(self):
        trade = make_trade(direction="LONG", entry_price=1.1000,
                           stop_loss=1.0950, mae=0.0060)
        assert stopped_out_at_full_distance(trade) is True

    def test_long_not_stopped(self):
        trade = make_trade(direction="LONG", entry_price=1.1000,
                           stop_loss=1.0950, mae=0.0020)
        assert stopped_out_at_full_distance(trade) is False

    def test_short_stopped(self):
        trade = make_trade(direction="SHORT", entry_price=1.1000,
                           stop_loss=1.1050, mae=0.0060)
        assert stopped_out_at_full_distance(trade) is True

    def test_missing_returns_none(self):
        trade = make_trade(mae=None)
        assert stopped_out_at_full_distance(trade) is None


class TestEfficiencyRatio:
    def test_efficiency(self):
        trade = make_trade(mfe=0.009, mae=0.001)
        assert efficiency_ratio(trade) == pytest.approx(0.9)

    def test_zero_total(self):
        trade = make_trade(mfe=0.0, mae=0.0)
        assert efficiency_ratio(trade) == pytest.approx(0.0)

    def test_missing_returns_none(self):
        trade = make_trade(mfe=None, mae=None)
        assert efficiency_ratio(trade) is None


class TestPriceRecoveredFromMae:
    def test_long_recovered(self):
        trade = make_trade(direction="LONG", entry_price=1.1000,
                           mae=0.01, exit_price=1.0980)
        assert price_recovered_from_mae(trade) is True

    def test_long_not_recovered(self):
        trade = make_trade(direction="LONG", entry_price=1.1000,
                           mae=0.01, exit_price=1.0905)
        assert price_recovered_from_mae(trade) is False

    def test_short_recovered(self):
        trade = make_trade(direction="SHORT", entry_price=1.1000,
                           mae=0.01, exit_price=1.1020)
        assert price_recovered_from_mae(trade) is True

    def test_missing_returns_none(self):
        trade = make_trade(mae=None)
        assert price_recovered_from_mae(trade) is None
