from __future__ import annotations

from forex_agent.data.schemas import (
    Alert,
    AlertSeverity,
    PerformanceMetrics,
    TradeRecord,
)
from forex_agent.data.ingestion import compute_r_multiple


class AlertEngine:
    """Deterministic alert engine.

    Raises alerts only when configured thresholds are crossed. Thresholds come
    from the agent config and are configurable via environment variables.
    """

    def __init__(self, thresholds: dict[str, float]) -> None:
        self.thresholds = thresholds

    def check(
        self,
        trades: list[TradeRecord],
        metrics: PerformanceMetrics,
        anomalies: list[dict],
    ) -> list[Alert]:
        alerts: list[Alert] = []

        alerts.extend(self._check_drawdown(metrics))
        alerts.extend(self._check_win_rate(metrics))
        alerts.extend(self._check_expectancy(metrics))
        alerts.extend(self._check_consecutive_losses(trades, metrics))
        alerts.extend(self._check_data_quality(trades))
        alerts.extend(self._check_anomalies(anomalies))

        return alerts

    def _check_drawdown(self, metrics: PerformanceMetrics) -> list[Alert]:
        threshold = self.thresholds.get("max_drawdown_pct", 0.10)
        if metrics.max_drawdown_pct <= 0:
            return []
        if metrics.max_drawdown_pct >= threshold:
            return [Alert(
                severity=AlertSeverity.CRITICAL if metrics.max_drawdown_pct >= threshold * 1.5 else AlertSeverity.WARNING,
                alert_type="drawdown",
                description=f"Drawdown of {metrics.max_drawdown_pct:.1%} exceeds threshold {threshold:.1%}",
                threshold=threshold,
                actual=metrics.max_drawdown_pct,
            )]
        return []

    def _check_win_rate(self, metrics: PerformanceMetrics) -> list[Alert]:
        threshold = self.thresholds.get("min_win_rate", 0.35)
        if metrics.total_trades > 0 and metrics.win_rate < threshold:
            return [Alert(
                severity=AlertSeverity.WARNING,
                alert_type="win_rate",
                description=f"Win rate {metrics.win_rate:.1%} below threshold {threshold:.1%}",
                threshold=threshold,
                actual=metrics.win_rate,
            )]
        return []

    def _check_expectancy(self, metrics: PerformanceMetrics) -> list[Alert]:
        threshold = self.thresholds.get("min_expectancy_r", -0.3)
        if metrics.expectancy < threshold:
            return [Alert(
                severity=AlertSeverity.CRITICAL,
                alert_type="expectancy",
                description=f"Expectancy {metrics.expectancy:.2f}R below threshold {threshold:.2f}R",
                threshold=threshold,
                actual=metrics.expectancy,
            )]
        return []

    def _check_consecutive_losses(
        self, trades: list[TradeRecord], metrics: PerformanceMetrics
    ) -> list[Alert]:
        threshold = self.thresholds.get("max_consecutive_losses", 5)
        closed = [t for t in trades if t.exit_price is not None]
        if len(closed) < 3:
            return []

        streak = 0
        for t in reversed(closed):
            if not t.is_winner:
                streak += 1
            else:
                break
        if streak >= threshold:
            return [Alert(
                severity=AlertSeverity.CRITICAL if streak >= threshold + 2 else AlertSeverity.WARNING,
                alert_type="consecutive_losses",
                description=f"Current losing streak of {streak} (threshold {threshold})",
                threshold=threshold,
                actual=streak,
            )]
        return []

    def _check_data_quality(self, trades: list[TradeRecord]) -> list[Alert]:
        """Warn about suspicious missing/zero values across the dataset."""

        alerts = []

        if not trades:
            return []

        with_exit = sum(1 for t in trades if t.exit_price is not None)
        if with_exit == 0 and len(trades) >= 5:
            alerts.append(Alert(
                severity=AlertSeverity.WARNING,
                alert_type="data_quality",
                description=f"No closed trades ({len(trades)} open records)",
                threshold=1.0,
                actual=0.0,
            ))

        # Look for impossible prices (e.g., too close to zero for FX)
        suspicious = [t for t in trades if t.entry_price <= 0]
        if suspicious:
            alerts.append(Alert(
                severity=AlertSeverity.WARNING,
                alert_type="data_quality",
                description=f"{len(suspicious)} trades with non-positive entry price",
                threshold=0.0,
                actual=len(suspicious),
            ))

        # Duplicate trade IDs
        seen: set[str] = set()
        dupes: set[str] = set()
        for t in trades:
            if t.trade_id in seen:
                dupes.add(t.trade_id)
            seen.add(t.trade_id)
        if dupes:
            alerts.append(Alert(
                severity=AlertSeverity.CRITICAL,
                alert_type="data_quality",
                description=f"Duplicate trade IDs: {', '.join(sorted(dupes))}",
                threshold=0.0,
                actual=len(dupes),
            ))

        # Future timestamps
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        futures = [t for t in trades if t.entry_time and _aware(t.entry_time) > now]
        if futures:
            alerts.append(Alert(
                severity=AlertSeverity.ERROR if False else AlertSeverity.WARNING,
                alert_type="data_quality",
                description=f"{len(futures)} trades with future entry timestamps",
                threshold=0.0,
                actual=len(futures),
            ))

        return alerts

    def _check_anomalies(self, anomalies: list[dict]) -> list[Alert]:
        alerts = []
        for a in anomalies:
            severity = AlertSeverity.CRITICAL if a.get("severity") == "high" else AlertSeverity.WARNING
            alerts.append(Alert(
                severity=severity,
                alert_type=a.get("type", "anomaly"),
                description=a.get("description", ""),
                threshold=self.thresholds.get("max_anomaly_z_score", 2.5),
                actual=float(a.get("count", 1)),
            ))
        return alerts


def _aware(dt) -> object:
    from datetime import timezone

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
