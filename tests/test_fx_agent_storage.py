from __future__ import annotations

import os

from forex_agent.data.schemas import (
    Alert,
    AlertSeverity,
    ConfidenceLevel,
    FailureCategory,
    PerformanceMetrics,
    TradeFailureAnalysis,
)
from forex_agent.storage.database import AnalysisStore, ensure_tables_exist


def make_failure(trade_id: str = "T1") -> TradeFailureAnalysis:
    return TradeFailureAnalysis(
        trade_id=trade_id,
        failure_category=FailureCategory.RULE_VIOLATION,
        confidence=ConfidenceLevel.HIGH,
        description="Overtraded against daily trend",
        contributing_factors=["signal skipped"],
        verdict="Strategy failure",
        research_implication="monitor",
        evidence=["sub-optimal entry"],
    )


def make_alert(**overrides) -> Alert:
    defaults = dict(
        severity=AlertSeverity.WARNING,
        alert_type="win_rate",
        description="Win rate below threshold",
        threshold=0.35,
        actual=0.20,
    )
    defaults.update(overrides)
    return Alert(**defaults)


def make_metrics(**overrides) -> PerformanceMetrics:
    defaults = dict(
        total_trades=50,
        win_rate=0.30,
        expectancy=-0.2,
        max_drawdown_pct=0.15,
        consecutive_losses=8,
    )
    defaults.update(overrides)
    return PerformanceMetrics(**defaults)


class TestAnalysisStore:
    def test_trade_failure_roundtrip(self, tmp_path):
        store = AnalysisStore(str(tmp_path / "a.db"))
        store.save_trade_failure(make_failure("T1"))
        loaded = store.get_trade_failure("T1")
        assert loaded is not None
        assert loaded.trade_id == "T1"
        assert loaded.failure_category == FailureCategory.RULE_VIOLATION
        assert loaded.verdict == "Strategy failure"
        assert loaded.confidence == ConfidenceLevel.HIGH

    def test_trade_failure_upsert(self, tmp_path):
        store = AnalysisStore(str(tmp_path / "a.db"))
        store.save_trade_failure(make_failure("T1"))
        store.save_trade_failure(make_failure("T1"))
        failures = store.list_trade_failures()
        assert len(failures) == 1

    def test_get_missing_returns_none(self, tmp_path):
        store = AnalysisStore(str(tmp_path / "a.db"))
        assert store.get_trade_failure("MISSING") is None

    def test_multiple_failures_listed(self, tmp_path):
        store = AnalysisStore(str(tmp_path / "a.db"))
        store.save_trade_failure(make_failure("T1"))
        store.save_trade_failure(make_failure("T2"))
        store.save_trade_failure(make_failure("T3"))
        assert {f.trade_id for f in store.list_trade_failures()} == {"T1", "T2", "T3"}

    def test_save_alerts(self, tmp_path):
        store = AnalysisStore(str(tmp_path / "a.db"))
        alerts = [make_alert(), make_alert(alert_type="drawdown", severity=AlertSeverity.CRITICAL)]
        saved = store.save_alerts(alerts)
        assert saved == 2
        assert store.count_alerts() == 2
        assert store.count_alerts("drawdown") == 1

    def test_save_alerts_empty(self, tmp_path):
        store = AnalysisStore(str(tmp_path / "a.db"))
        assert store.save_alerts([]) == 0

    def test_list_alerts_recent_first(self, tmp_path):
        store = AnalysisStore(str(tmp_path / "a.db"))
        store.save_alerts([make_alert(alert_type="a"), make_alert(alert_type="b")])
        alerts = store.list_alerts(limit=10)
        assert len(alerts) == 2
        assert alerts[-1].alert_type == "a"
        assert alerts[0].alert_type == "b"

    def test_save_report(self, tmp_path):
        store = AnalysisStore(str(tmp_path / "a.db"))
        store.save_report(
            "weekly",
            "Weekly research content",
            metrics=make_metrics(),
            title="Week 5",
        )
        reports = store.list_reports("weekly")
        assert len(reports) == 1
        assert reports[0]["report_key"] == "weekly"
        assert reports[0]["title"] == "Week 5"

    def test_list_reports_all(self, tmp_path):
        store = AnalysisStore(str(tmp_path / "a.db"))
        store.save_report("weekly", "Weekly")
        store.save_report("monthly", "Monthly")
        reports = store.list_reports(limit=10)
        assert {r["report_key"] for r in reports} == {"weekly", "monthly"}

    def test_metrics_history_roundtrip(self, tmp_path):
        store = AnalysisStore(str(tmp_path / "a.db"))
        store.record_metrics(make_metrics(win_rate=0.40))
        store.record_metrics(make_metrics(win_rate=0.32))
        history = store.list_metrics_history()
        assert [m.win_rate for m in history] == [0.40, 0.32]

    def test_sqlite_url_prefix(self, tmp_path):
        store = AnalysisStore("sqlite:///" + str(tmp_path / "b.db"))
        store.save_alerts([make_alert()])
        assert store.count_alerts() == 1

    def test_ensure_tables_exist(self, tmp_path):
        path = str(tmp_path / "c.db")
        ensure_tables_exist(path)
        assert os.path.exists(path)


class TestSchemaFromJson:
    def test_alert_roundtrip(self):
        a = make_alert()
        assert Alert.from_json(a.to_json()).to_dict() == a.to_dict()

    def test_alert_severity_restored(self):
        a = make_alert(severity=AlertSeverity.CRITICAL)
        restored = Alert.from_json(a.to_json())
        assert restored.severity == AlertSeverity.CRITICAL

    def test_failure_roundtrip(self):
        f = make_failure()
        assert TradeFailureAnalysis.from_json(f.to_json()).to_dict() == f.to_dict()

    def test_metrics_roundtrip(self):
        m = make_metrics()
        assert PerformanceMetrics.from_json(m.to_json()).to_dict() == m.to_dict()
