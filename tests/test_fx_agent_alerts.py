from __future__ import annotations

from datetime import datetime

import pytest

from forex_agent.analysis.alerts import AlertEngine
from forex_agent.data.schemas import AlertSeverity, PerformanceMetrics, TradeRecord


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


def make_metrics(**overrides) -> PerformanceMetrics:
    defaults = dict(
        total_trades=50,
        winning_trades=30,
        losing_trades=20,
        win_rate=0.60,
        expectancy=0.15,
        max_drawdown_pct=0.05,
        total_pnl=100.0,
        consecutive_losses=2,
        profit_factor=2.0,
        sharpe_ratio=1.2,
    )
    defaults.update(overrides)
    return PerformanceMetrics(**defaults)


class TestAlertEngine:
    def test_no_alerts_on_healthy_data(self):
        engine = AlertEngine(thresholds={})
        trades = [make_trade(trade_id=f"T{i}") for i in range(10)]
        metrics = make_metrics()
        alerts = engine.check(trades, metrics, anomalies=[])
        assert alerts == []

    def test_drawdown_alert(self):
        engine = AlertEngine(thresholds={"max_drawdown_pct": 0.10})
        metrics = make_metrics(max_drawdown_pct=0.20)
        alerts = engine.check([], metrics, anomalies=[])
        assert any(a.alert_type == "drawdown" for a in alerts)

    def test_drawdown_critical_at_high_level(self):
        engine = AlertEngine(thresholds={"max_drawdown_pct": 0.10})
        metrics = make_metrics(max_drawdown_pct=0.20)
        alert = [a for a in engine.check([], metrics, anomalies=[]) if a.alert_type == "drawdown"][0]
        assert alert.severity == AlertSeverity.CRITICAL

    def test_win_rate_alert(self):
        engine = AlertEngine(thresholds={"min_win_rate": 0.35})
        metrics = make_metrics(win_rate=0.25)
        alerts = engine.check([], metrics, anomalies=[])
        assert any(a.alert_type == "win_rate" for a in alerts)

    def test_expectancy_alert(self):
        engine = AlertEngine(thresholds={"min_expectancy_r": -0.3})
        metrics = make_metrics(expectancy=-0.5)
        alerts = engine.check([], metrics, anomalies=[])
        assert any(a.alert_type == "expectancy" for a in alerts)

    def test_consecutive_losses_alert(self):
        engine = AlertEngine(thresholds={"max_consecutive_losses": 3})
        trades = [
            make_trade(trade_id=f"T{i}", entry_price=1.1000, exit_price=1.0950)
            for i in range(5)
        ]
        metrics = make_metrics(consecutive_losses=5)
        alerts = engine.check(trades, metrics, anomalies=[])
        assert any(a.alert_type == "consecutive_losses" for a in alerts)

    def test_no_consecutive_loss_alert_below_threshold(self):
        engine = AlertEngine(thresholds={"max_consecutive_losses": 5})
        trades = [make_trade(trade_id=f"T{i}") for i in range(3)]
        metrics = make_metrics(consecutive_losses=2)
        alerts = engine.check(trades, metrics, anomalies=[])
        assert not any(a.alert_type == "consecutive_losses" for a in alerts)

    def test_duplicate_trade_id_alert(self):
        engine = AlertEngine(thresholds={})
        trades = [
            make_trade(trade_id="DUP1"),
            make_trade(trade_id="DUP1"),
        ]
        alerts = engine.check(trades, make_metrics(), anomalies=[])
        dup = [a for a in alerts if a.alert_type == "data_quality" and "Duplicate" in a.description]
        assert dup and dup[0].severity == AlertSeverity.CRITICAL

    def test_anomaly_alert_maps_severity(self):
        engine = AlertEngine(thresholds={})
        anomalies = [
            {"type": "test", "severity": "high", "description": "bad thing", "count": 3},
            {"type": "other", "severity": "medium", "description": "mild thing", "count": 1},
        ]
        alerts = engine.check([], make_metrics(), anomalies=anomalies)
        crit = [a for a in alerts if a.alert_type == "test"]
        warn = [a for a in alerts if a.alert_type == "other"]
        assert crit[0].severity == AlertSeverity.CRITICAL
        assert warn[0].severity == AlertSeverity.WARNING
