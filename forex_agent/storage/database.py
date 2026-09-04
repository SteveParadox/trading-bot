from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from forex_agent.data.schemas import Alert, PerformanceMetrics, TradeFailureAnalysis


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_db_path(db_path: str) -> str:
    if db_path.startswith("sqlite:///"):
        return db_path.replace("sqlite:///", "", 1)
    return db_path


class AnalysisStore:
    """SQLite-backed persistence for analysis artifacts.

    Stores trade failure analyses, alerts, generated reports and rolling
    performance-metric snapshots. Tables are created on first use.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = _normalize_db_path(db_path)
        self._ensure_schema()

    # -- schema ---------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        parent = Path(self.db_path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_failures (
                    trade_id TEXT PRIMARY KEY,
                    analysis_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    description TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reports (
                    report_key TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    PRIMARY KEY (report_key, generated_at)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recorded_at TEXT NOT NULL,
                    metrics_json TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    # -- trade failures ------------------------------------------------

    def save_trade_failure(self, analysis: TradeFailureAnalysis) -> None:
        conn = self._connect()
        try:
            now = _now_iso()
            conn.execute(
                """
                INSERT INTO trade_failures (trade_id, analysis_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(trade_id) DO UPDATE SET
                    analysis_json = excluded.analysis_json,
                    updated_at = excluded.updated_at
                """,
                (analysis.trade_id, analysis.to_json(), now, now),
            )
            conn.commit()
        finally:
            conn.close()

    def get_trade_failure(self, trade_id: str) -> Optional[TradeFailureAnalysis]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT analysis_json FROM trade_failures WHERE trade_id = ?",
                (trade_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return TradeFailureAnalysis.from_json(row["analysis_json"])

    def list_trade_failures(self) -> list[TradeFailureAnalysis]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT analysis_json FROM trade_failures ORDER BY updated_at"
            ).fetchall()
        finally:
            conn.close()
        return [TradeFailureAnalysis.from_json(r["analysis_json"]) for r in rows]

    # -- alerts ---------------------------------------------------------

    def save_alerts(self, alerts: list[Alert]) -> int:
        if not alerts:
            return 0
        conn = self._connect()
        try:
            for alert in alerts:
                conn.execute(
                    """
                    INSERT INTO alerts (alert_type, severity, description, payload_json, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        alert.alert_type,
                        alert.severity.value if hasattr(alert.severity, "value") else str(alert.severity),
                        alert.description,
                        alert.to_json(),
                        _now_iso(),
                    ),
                )
            conn.commit()
            return len(alerts)
        finally:
            conn.close()

    def list_alerts(self, limit: int = 100) -> list[Alert]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT payload_json FROM alerts ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        finally:
            conn.close()
        return [Alert.from_json(r["payload_json"]) for r in rows]

    def count_alerts(self, alert_type: Optional[str] = None) -> int:
        conn = self._connect()
        try:
            if alert_type:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM alerts WHERE alert_type = ?",
                    (alert_type,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) AS n FROM alerts").fetchone()
            return int(row["n"])
        finally:
            conn.close()

    # -- reports --------------------------------------------------------

    def save_report(
        self,
        report_key: str,
        content: str,
        metrics: Optional[PerformanceMetrics] = None,
        title: str = "",
    ) -> None:
        conn = self._connect()
        try:
            metrics_json = metrics.to_json() if metrics is not None else "{}"
            conn.execute(
                """
                INSERT INTO reports (report_key, title, content, metrics_json, generated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (report_key, title, content, metrics_json, _now_iso()),
            )
            conn.commit()
        finally:
            conn.close()

    def list_reports(self, report_key: Optional[str] = None, limit: int = 20) -> list[dict]:
        conn = self._connect()
        try:
            if report_key:
                rows = conn.execute(
                    "SELECT report_key, title, generated_at FROM reports WHERE report_key = ?"
                    " ORDER BY generated_at DESC LIMIT ?",
                    (report_key, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT report_key, title, generated_at FROM reports"
                    " ORDER BY generated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    # -- metrics history -------------------------------------------------

    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO metrics_history (recorded_at, metrics_json) VALUES (?, ?)",
                (_now_iso(), metrics.to_json()),
            )
            conn.commit()
        finally:
            conn.close()

    def list_metrics_history(self, limit: int = 100) -> list[PerformanceMetrics]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT metrics_json FROM metrics_history ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        finally:
            conn.close()
        return [PerformanceMetrics.from_json(r["metrics_json"]) for r in reversed(rows)]


def ensure_tables_exist(db_path: str) -> None:
    """Idempotently create the analysis tables in an existing database file."""
    AnalysisStore(db_path)

    # Ensure the database file is created even if missing.
    if not Path(_normalize_db_path(db_path)).exists():
        os.makedirs(Path(_normalize_db_path(db_path)).parent, exist_ok=True)
        open(_normalize_db_path(db_path), "a").close()
        AnalysisStore(db_path)
