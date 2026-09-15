"""Operational monitoring, alerting, and health checks.

Exposes a single ``OperationalMonitor`` that the worker and API share.  Every
metric is append-only and thread-safe so the async event loop and the
synchronous scan thread can update it concurrently.

Checked items from imp.txt:
  - stale prices / stale scans
  - database-lock retries with alert threshold
  - broker disconnects
  - unknown orders
  - risk-halt state
  - news-data freshness
  - clock-health monitoring
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from fxbot.operations import ClockHealth

log = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    STALE_PRICE = "stale_price"
    STALE_SCAN = "stale_scan"
    DB_LOCK = "db_lock"
    BROKER_DISCONNECT = "broker_disconnect"
    UNKNOWN_ORDER = "unknown_order"
    RISK_HALT = "risk_halt"
    NEWS_FRESHNESS = "news_freshness"
    CLOCK_SKEW = "clock_skew"
    OPERATOR_REVIEW = "operator_review"


@dataclass(frozen=True)
class Alert:
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MonitorSnapshot:
    """Serialisable point-in-time view for the API /status endpoint."""

    stale_prices: dict[str, float]
    stale_scan_age_seconds: float | None
    db_lock_retries: int
    db_lock_alert: bool
    broker_connected: bool
    unknown_orders: int
    risk_halted: bool
    risk_halt_reason: str | None
    news_data_age_seconds: float | None
    news_data_fresh: bool | None
    clock_health: dict[str, Any]
    recent_alerts: list[dict[str, Any]]
    uptime_seconds: float


class OperationalMonitor:
    def __init__(
        self,
        *,
        stale_price_threshold_seconds: float = 120.0,
        stale_scan_threshold_seconds: float = 300.0,
        db_lock_alert_threshold: int = 10,
        clock_max_skew_seconds: float = 300.0,
        news_data_max_age_seconds: float = 3600.0,
        max_alerts: int = 200,
    ) -> None:
        self._lock = threading.Lock()
        self._start_time = time.monotonic()
        # Prices: instrument -> last observed epoch
        self._price_times: dict[str, float] = {}
        # Scan heartbeat
        self._last_scan_epoch: float | None = None
        # DB lock
        self._db_lock_retries: int = 0
        self._db_lock_alert_threshold = db_lock_alert_threshold
        # Broker
        self._broker_connected: bool = True
        self._last_disconnect_epoch: float | None = None
        # Unknown orders: track distinct unresolved ids so the same order does
        # not re-emit a CRITICAL alert on every scan cycle.
        self._unknown_order_ids: set[str] = set()
        # Risk halt
        self._risk_halted: bool = False
        self._risk_halt_reason: str | None = None
        # News freshness
        self._news_data_age: float | None = None
        self._news_data_fresh: bool | None = None
        # Clock
        self._clock_health: ClockHealth | None = None
        # Alerts (ring buffer)
        self._max_alerts = max_alerts
        self._alerts: list[Alert] = []
        # Thresholds
        self._stale_price_threshold = stale_price_threshold_seconds
        self._stale_scan_threshold = stale_scan_threshold_seconds
        self._clock_max_skew = clock_max_skew_seconds
        self._news_max_age = news_data_max_age_seconds

    # ------------------------------------------------------------------
    # Record mutations (called from worker / API)
    # ------------------------------------------------------------------

    def record_price(self, instrument: str, observed_at: datetime) -> None:
        epoch = observed_at.timestamp() if observed_at.tzinfo is not None else observed_at.replace(tzinfo=timezone.utc).timestamp()
        with self._lock:
            self._price_times[instrument] = epoch

    def record_scan_heartbeat(self) -> None:
        with self._lock:
            self._last_scan_epoch = time.monotonic()

    def record_db_lock_retry(self) -> None:
        with self._lock:
            self._db_lock_retries += 1
            if self._db_lock_retries >= self._db_lock_alert_threshold:
                self._emit_alert(
                    AlertType.DB_LOCK,
                    AlertSeverity.WARNING,
                    f"database lock retries reached {self._db_lock_retries}",
                )

    def record_broker_disconnect(self, reason: str = "") -> None:
        with self._lock:
            self._broker_connected = False
            self._last_disconnect_epoch = time.monotonic()
            self._emit_alert(
                AlertType.BROKER_DISCONNECT,
                AlertSeverity.CRITICAL,
                f"broker disconnected: {reason}" if reason else "broker disconnected",
            )

    def record_broker_reconnect(self) -> None:
        with self._lock:
            self._broker_connected = True
            self._last_disconnect_epoch = None

    def record_unknown_order(self, client_order_id: str) -> None:
        with self._lock:
            if client_order_id in self._unknown_order_ids:
                return
            self._unknown_order_ids.add(client_order_id)
            self._emit_alert(
                AlertType.UNKNOWN_ORDER,
                AlertSeverity.CRITICAL,
                f"unknown order {client_order_id}",
                payload={"client_order_id": client_order_id},
            )

    def clear_unknown_order(self, client_order_id: str) -> None:
        """Forget an order once it reconciles/rejects so it can leave the alert set."""
        with self._lock:
            self._unknown_order_ids.discard(client_order_id)

    def record_risk_halt(self, reason: str) -> None:
        with self._lock:
            self._risk_halted = True
            self._risk_halt_reason = reason
            self._emit_alert(AlertType.RISK_HALT, AlertSeverity.WARNING, f"risk halt: {reason}")

    def clear_risk_halt(self) -> None:
        with self._lock:
            self._risk_halted = False
            self._risk_halt_reason = None

    def record_news_freshness(self, age_seconds: float | None, is_fresh: bool) -> None:
        with self._lock:
            self._news_data_age = age_seconds
            previous_fresh = self._news_data_fresh
            self._news_data_fresh = is_fresh
            if age_seconds is not None and not is_fresh and previous_fresh is not False:
                self._emit_alert(
                    AlertType.NEWS_FRESHNESS,
                    AlertSeverity.WARNING,
                    f"news data is stale: {age_seconds:.0f}s old (max {self._news_max_age:.0f}s)",
                )

    def record_clock_health(self, health: ClockHealth) -> None:
        with self._lock:
            self._clock_health = health
            if health.status != "healthy":
                self._emit_alert(
                    AlertType.CLOCK_SKEW,
                    AlertSeverity.WARNING,
                    f"broker-host clock skew: {health.skew_seconds:.1f}s ({health.reason})",
                )

    def record_operator_review(self, message: str, payload: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._emit_alert(
                AlertType.OPERATOR_REVIEW,
                AlertSeverity.CRITICAL,
                message,
                payload=payload or {},
            )

    def check_stale_prices(self, now: datetime | None = None) -> list[str]:
        """Return instruments whose price data is older than the threshold."""
        now_epoch = (now or datetime.now(timezone.utc)).timestamp()
        stale: list[str] = []
        with self._lock:
            for name, epoch in self._price_times.items():
                age = now_epoch - epoch
                if age > self._stale_price_threshold:
                    stale.append(name)
                    self._emit_alert(
                        AlertType.STALE_PRICE,
                        AlertSeverity.WARNING,
                        f"stale price for {name}: {age:.0f}s old",
                        payload={"instrument": name, "age_seconds": age},
                    )
        return stale

    def check_stale_scan(self, now: datetime | None = None) -> bool:
        now_mono = time.monotonic()
        with self._lock:
            if self._last_scan_epoch is None:
                return False
            age = now_mono - self._last_scan_epoch
            if age > self._stale_scan_threshold:
                self._emit_alert(
                    AlertType.STALE_SCAN,
                    AlertSeverity.WARNING,
                    f"scan heartbeat stale: {age:.0f}s since last scan",
                )
                return True
        return False

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self, now: datetime | None = None) -> MonitorSnapshot:
        now_epoch = (now or datetime.now(timezone.utc)).timestamp()
        with self._lock:
            stale_prices = {
                name: now_epoch - epoch
                for name, epoch in self._price_times.items()
                if (now_epoch - epoch) > self._stale_price_threshold
            }
            scan_age: float | None = None
            if self._last_scan_epoch is not None:
                scan_age = time.monotonic() - self._last_scan_epoch
            db_lock_alert = self._db_lock_retries >= self._db_lock_alert_threshold
            clock_dict: dict[str, Any] = {}
            if self._clock_health is not None:
                clock_dict = {
                    "status": self._clock_health.status,
                    "skew_seconds": self._clock_health.skew_seconds,
                    "observed_at": self._clock_health.observed_at,
                    "reason": self._clock_health.reason,
                }
            alerts = [self._alert_to_dict(a) for a in self._alerts[-50:]]
            return MonitorSnapshot(
                stale_prices=stale_prices,
                stale_scan_age_seconds=scan_age,
                db_lock_retries=self._db_lock_retries,
                db_lock_alert=db_lock_alert,
                broker_connected=self._broker_connected,
                unknown_orders=len(self._unknown_order_ids),
                risk_halted=self._risk_halted,
                risk_halt_reason=self._risk_halt_reason,
                news_data_age_seconds=self._news_data_age,
                news_data_fresh=self._news_data_fresh,
                clock_health=clock_dict,
                recent_alerts=alerts,
                uptime_seconds=time.monotonic() - self._start_time,
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _emit_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp=datetime.now(timezone.utc).isoformat(),
            payload=payload or {},
        )
        self._alerts.append(alert)
        if len(self._alerts) > self._max_alerts:
            self._alerts = self._alerts[-self._max_alerts :]
        if severity in (AlertSeverity.WARNING, AlertSeverity.CRITICAL):
            log_fn = log.warning if severity == AlertSeverity.WARNING else log.critical
            log_fn("[monitor] %s: %s", alert_type.value, message)

    @staticmethod
    def _alert_to_dict(alert: Alert) -> dict[str, Any]:
        return {
            "type": alert.alert_type.value,
            "severity": alert.severity.value,
            "message": alert.message,
            "timestamp": alert.timestamp,
            "payload": alert.payload,
        }
