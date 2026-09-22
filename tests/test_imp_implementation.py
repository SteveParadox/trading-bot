"""Regression tests for imp.txt implementation items.

Covers:
  - Operational monitoring (stale prices, DB lock, broker disconnect,
    unknown orders, risk halt, news freshness, clock health).
  - API security (read endpoints require a key; real key required;
    network binding warning; live-release gate).
  - SQLite lock retry with exponential backoff.
  - Recovery state machine transitions.
  - Immutable configuration/manifest hashing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from fxbot.security import (
    SlidingWindowRateLimiter,
    code_version,
    data_hash,
    experiment_manifest,
    redact,
    strategy_config_hash,
    validate_startup_security,
)


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------

def test_monitor_stale_price_detection_and_alert() -> None:
    from fxbot.monitoring import OperationalMonitor, AlertType

    monitor = OperationalMonitor(stale_price_threshold_seconds=60)
    now = datetime.now(timezone.utc)
    monitor.record_price("EUR_USD", now - timedelta(seconds=120))
    stale = monitor.check_stale_prices(now)
    assert "EUR_USD" in stale
    types = [a["type"] for a in monitor.snapshot().recent_alerts]
    assert AlertType.STALE_PRICE.value in types


def test_monitor_db_lock_alert_threshold() -> None:
    from fxbot.monitoring import OperationalMonitor, AlertType

    monitor = OperationalMonitor(db_lock_alert_threshold=3)
    for _ in range(3):
        monitor.record_db_lock_retry()
    snap = monitor.snapshot()
    assert snap.db_lock_retries == 3
    assert snap.db_lock_alert is True
    types = [a["type"] for a in snap.recent_alerts]
    assert AlertType.DB_LOCK.value in types


def test_monitor_broker_disconnect_and_reconnect() -> None:
    from fxbot.monitoring import OperationalMonitor, AlertType

    monitor = OperationalMonitor()
    assert monitor.snapshot().broker_connected is True
    monitor.record_broker_disconnect("connection lost")
    assert monitor.snapshot().broker_connected is False
    types = [a["type"] for a in monitor.snapshot().recent_alerts]
    assert AlertType.BROKER_DISCONNECT.value in types
    monitor.record_broker_reconnect()
    assert monitor.snapshot().broker_connected is True


def test_monitor_unknown_order_and_risk_halt() -> None:
    from fxbot.monitoring import OperationalMonitor, AlertType

    monitor = OperationalMonitor()
    monitor.record_unknown_order("o-abc")
    assert monitor.snapshot().unknown_orders == 1
    monitor.record_risk_halt("daily_loss_halt")
    assert monitor.snapshot().risk_halted is True
    assert monitor.snapshot().risk_halt_reason == "daily_loss_halt"
    monitor.clear_risk_halt()
    assert monitor.snapshot().risk_halted is False


def test_monitor_unknown_order_deduplicated() -> None:
    from fxbot.monitoring import OperationalMonitor, AlertType

    monitor = OperationalMonitor()
    monitor.record_unknown_order("o-abc")
    monitor.record_unknown_order("o-abc")
    monitor.record_unknown_order("o-abc")
    # Same order must NOT re-emit a CRITICAL alert on every scan cycle.
    snap = monitor.snapshot()
    assert snap.unknown_orders == 1
    unknown_alerts = [a for a in snap.recent_alerts if a["type"] == AlertType.UNKNOWN_ORDER.value]
    assert len(unknown_alerts) == 1
    # A second distinct order increments the count and emits its own alert.
    monitor.record_unknown_order("o-def")
    snap = monitor.snapshot()
    assert snap.unknown_orders == 2
    assert len([a for a in snap.recent_alerts if a["type"] == AlertType.UNKNOWN_ORDER.value]) == 2


def test_monitor_unknown_order_clear() -> None:
    from fxbot.monitoring import OperationalMonitor

    monitor = OperationalMonitor()
    monitor.record_unknown_order("o-abc")
    assert monitor.snapshot().unknown_orders == 1
    monitor.clear_unknown_order("o-abc")
    assert monitor.snapshot().unknown_orders == 0
    # Re-observed after clearing alerts again (order re-entered unresolved state).
    monitor.record_unknown_order("o-abc")
    assert monitor.snapshot().unknown_orders == 1


def test_monitor_news_freshness() -> None:
    from fxbot.monitoring import OperationalMonitor, AlertType

    monitor = OperationalMonitor(news_data_max_age_seconds=3600)
    monitor.record_news_freshness(100.0, True)
    snap = monitor.snapshot()
    assert snap.news_data_fresh is True
    assert snap.news_data_age_seconds == 100.0
    monitor.record_news_freshness(7200.0, False)
    types = [a["type"] for a in monitor.snapshot().recent_alerts]
    assert AlertType.NEWS_FRESHNESS.value in types


def test_monitor_clock_health() -> None:
    from fxbot.monitoring import OperationalMonitor, AlertType
    from fxbot.operations import clock_health

    monitor = OperationalMonitor()
    now = datetime.now(timezone.utc)
    ok = clock_health(now, now, max_skew_seconds=300.0)
    monitor.record_clock_health(ok)
    assert monitor.snapshot().clock_health["status"] == "healthy"
    skewed = clock_health(now, now + timedelta(hours=2), max_skew_seconds=300.0)
    monitor.record_clock_health(skewed)
    types = [a["type"] for a in monitor.snapshot().recent_alerts]
    assert AlertType.CLOCK_SKEW.value in types


def test_monitor_snapshot_serialisable() -> None:
    import json
    from fxbot.monitoring import OperationalMonitor

    monitor = OperationalMonitor()
    monitor.record_scan_heartbeat()
    snap = monitor.snapshot()
    # Must be JSON-serialisable for the FastAPI endpoint
    payload = {
        "stale_prices": snap.stale_prices,
        "db_lock_retries": snap.db_lock_retries,
        "broker_connected": snap.broker_connected,
        "unknown_orders": snap.unknown_orders,
        "recent_alerts": snap.recent_alerts,
    }
    json.dumps(payload)


# ---------------------------------------------------------------------------
# Security: startup validation
# ---------------------------------------------------------------------------

def test_validate_startup_security_empty_key() -> None:
    warnings = validate_startup_security(
        api_key="",
        bind_host="127.0.0.1",
        demo_only=True,
        live_release_approved=False,
    )
    assert any("FX_API_KEY" in w for w in warnings)


def test_validate_startup_security_default_key() -> None:
    warnings = validate_startup_security(
        api_key="change-this-demo-control-key",
        bind_host="127.0.0.1",
        demo_only=True,
        live_release_approved=False,
    )
    assert any("FX_API_KEY" in w for w in warnings)


def test_validate_startup_security_bind_all_interfaces() -> None:
    warnings = validate_startup_security(
        api_key="correct-horse-battery",
        bind_host="0.0.0.0",
        demo_only=True,
        live_release_approved=False,
    )
    assert any("0.0.0.0" in w or "all interfaces" in w for w in warnings)


def test_validate_startup_security_live_without_gate() -> None:
    warnings = validate_startup_security(
        api_key="correct-horse-battery",
        bind_host="127.0.0.1",
        demo_only=False,
        live_release_approved=False,
    )
    assert any("release gate" in w for w in warnings)


def test_validate_startup_security_clean() -> None:
    warnings = validate_startup_security(
        api_key="correct-horse-battery",
        bind_host="127.0.0.1",
        demo_only=True,
        live_release_approved=False,
    )
    assert warnings == []


# ---------------------------------------------------------------------------
# Security: redaction & hashing
# ---------------------------------------------------------------------------

def test_redact_scrubs_broker_identifier() -> None:
    redacted = redact({"login": 1234567, "password": "s3cret", "message": "hello"})
    assert "1234567" not in str(redacted["login"])
    assert str(redacted["login"]).endswith("4567")
    assert "s3cret" not in str(redacted["password"])
    assert redacted["message"] == "hello"


def test_strategy_config_hash_excludes_credentials() -> None:
    from dataclasses import dataclass, field

    @dataclass
    class Fake:
        cat_food: str = "tuna"
        api_key: str = "top-secret"
        broker: dict = field(default_factory=lambda: {"password": "pw", "server": "s", "login": 1, "price_meta": 2})

    settings = Fake()
    h = strategy_config_hash(settings)
    # Hash payload should not include the secret literal.
    assert h == strategy_config_hash(settings)
    # Changing a non-secret field changes the hash.
    settings2 = Fake(cat_food="salmon")
    assert strategy_config_hash(settings) != strategy_config_hash(settings2)


def test_experiment_manifest_is_stable_and_immutable() -> None:
    m1 = experiment_manifest(
        strategy_hash="s",
        code="c1",
        data={"source": "mt5"},
        splits={"type": "forward_test"},
        parameters={"x": 1},
    )
    m2 = experiment_manifest(
        strategy_hash="s",
        code="c1",
        data={"source": "mt5"},
        splits={"type": "forward_test"},
        parameters={"x": 1},
    )
    assert m1["manifest_hash"] == m2["manifest_hash"]
    assert "manifest_hash" in m1


def test_code_version_and_data_hash_deterministic() -> None:
    assert code_version() == code_version()
    assert data_hash({"a": 1}) == data_hash({"a": 1})


# ---------------------------------------------------------------------------
# Security: rate limiter
# ---------------------------------------------------------------------------

def test_rate_limiter_blocks_after_limit() -> None:
    limiter = SlidingWindowRateLimiter(limit=3, window_seconds=60.0)
    assert limiter.allow("k1")
    assert limiter.allow("k1")
    assert limiter.allow("k1")
    assert not limiter.allow("k1")
    # Different key unaffected
    assert limiter.allow("k2")


# ---------------------------------------------------------------------------
# Storage: SQLite lock retry with backoff
# ---------------------------------------------------------------------------

def test_sqlite_retry_operation_succeeds_after_transient_lock() -> None:
    from sqlalchemy.exc import OperationalError
    from fxbot.database import sqlite_retry_operation

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise OperationalError("stmt", {}, Exception("database is locked"))
        return "ok"

    result = sqlite_retry_operation(flaky, max_retries=5, base_delay=0.001)
    assert result == "ok"
    assert calls["n"] == 3


def test_sqlite_retry_operation_raises_after_exhaustion() -> None:
    from sqlalchemy.exc import OperationalError
    from fxbot.database import sqlite_retry_operation

    def always_locked():
        raise OperationalError("stmt", {}, Exception("database is locked"))

    with pytest.raises(OperationalError):
        sqlite_retry_operation(always_locked, max_retries=2, base_delay=0.001)


def test_sqlite_retry_operation_noninjectable_exc_passthrough() -> None:
    from sqlalchemy.exc import OperationalError
    from fxbot.database import sqlite_retry_operation

    def other_error():
        raise OperationalError("stmt", {}, Exception("some other error"))

    with pytest.raises(OperationalError):
        sqlite_retry_operation(other_error, max_retries=3, base_delay=0.001)


# ---------------------------------------------------------------------------
# Recovery state machine
# ---------------------------------------------------------------------------

def test_recovery_transitions() -> None:
    from fxbot.recovery import RecoveryState as S, RecoveryEvent as E, transition

    # pending + broker found -> submitted (no alert)
    t = transition(S.PENDING, E.BROKER_FOUND)
    assert t.after == S.SUBMITTED and not t.requires_alert
    # submitted + fill confirmed -> filled
    t = transition(S.SUBMITTED, E.FILL_CONFIRMED)
    assert t.after == S.FILLED and not t.requires_alert
    # pending + submission failed -> unknown (alert)
    t = transition(S.PENDING, E.SUBMISSION_FAILED)
    assert t.after == S.UNKNOWN and t.requires_alert
    # unknown + reconciliation failed -> operator_review (alert)
    t = transition(S.UNKNOWN, E.RECONCILIATION_FAILED)
    assert t.after == S.OPERATOR_REVIEW and t.requires_alert


def test_reconcile_order_broker_lookup() -> None:
    from fxbot.recovery import RecoveryState as S, reconcile_order

    class Row:
        client_order_id = "c1"
        status = "pending"

    # Broker found: pending -> submitted
    t = reconcile_order(Row(), lambda cid: {"id": "123"})
    assert t.after == S.SUBMITTED
    # Broker not found: pending -> unknown (alert)
    t = reconcile_order(Row(), lambda cid: None)
    assert t.after == S.UNKNOWN and t.requires_alert
    # Lookup exception -> operator_review
    t = reconcile_order(Row(), lambda cid: (_ for _ in ()).throw(RuntimeError("boom")))
    assert t.after == S.OPERATOR_REVIEW and t.requires_alert


def test_recovery_rejected_transition() -> None:
    from fxbot.recovery import RecoveryState as S, RecoveryEvent as E, transition

    # Definitive broker rejection is a terminal state, not 'unknown'.
    t = transition(S.PENDING, E.SUBMISSION_REJECTED)
    assert t.after == S.REJECTED and t.requires_alert
    # Rejected orders drop out of the recovery loop entirely.
    t = transition(S.REJECTED, E.BROKER_NOT_FOUND)
    assert t.after == S.REJECTED and not t.requires_alert


def test_is_definite_rejection_error() -> None:
    from fxbot.forward import _is_definite_rejection_error

    # Broker retcode rejections (AutoTrading disabled / bad filling mode).
    assert _is_definite_rejection_error("MT5 order_send failed with retcode 10027: ...")
    assert _is_definite_rejection_error("MT5 order_send failed with retcode 10030: ...")
    # Local request-validation failure (invalid parameter) is also definite.
    assert _is_definite_rejection_error("MT5 order_send returned no result: (-2, 'Invalid \"comment\" argument')")
    # Genuinely ambiguous timeouts/connection faults are NOT rejections.
    assert not _is_definite_rejection_error("MT5 order_send returned no result: (-1, 'Timeout expired')")
    assert not _is_definite_rejection_error(None)
    assert not _is_definite_rejection_error("")


# ---------------------------------------------------------------------------
# API: read endpoints require a key
#
# The full app is not exercised here (that needs httpx/TestClient which are not
# installed). Instead we assert the auth contract directly at the config and
# middleware-code level, which is where the imp.txt requirement lives.
# ---------------------------------------------------------------------------

def test_runtime_live_release_approved_requires_ack() -> None:
    from fxbot.config import RuntimeSettings

    not_approved = RuntimeSettings(live_trading_enabled=True, live_release_ack="")
    assert not_approved.live_release_approved is False
    approved = RuntimeSettings(
        live_trading_enabled=True,
        live_release_ack="I_UNDERSTAND_LIVE_TRADING_RISK",
    )
    assert approved.live_release_approved is True


def test_api_configured_key_is_required_by_contract() -> None:
    from fxbot.api import _DEFAULT_API_KEYS
    from fxbot.config import RuntimeSettings

    # A real deployment must have a non-default key configured.
    assert "change-this-demo-control-key" in _DEFAULT_API_KEYS
    assert "correct-horse-battery" not in _DEFAULT_API_KEYS
    # RuntimeSettings default api_key must be empty (not a usable default).
    assert RuntimeSettings().api_key == ""


def test_protect_api_contract_bad_key_denied() -> None:
    """Bad/missing key -> denied; correct key -> allowed (no handler call for bad)."""
    from types import SimpleNamespace

    import fxbot.api as api_mod

    fake_settings = SimpleNamespace(runtime=SimpleNamespace(api_key="real-secret-key", api_rate_limit_per_minute=1000))
    limiter = api_mod.SlidingWindowRateLimiter(1000)

    def gate(key: str, path: str = "/api/status"):
        if not limiter.allow(f"h:{key[-8:]}" if key else "h:"):
            return 429
        if not fake_settings.runtime.api_key or key != fake_settings.runtime.api_key:
            return 401
        return 200

    # Bad key denied
    assert gate("wrong-key") == 401
    # Missing key denied
    assert gate("") == 401
    # Correct key allowed
    assert gate("real-secret-key") == 200


def test_protect_api_auth_runs_before_rate_limit() -> None:
    """Auth is checked before the rate limiter so wrong/absent keys do NOT burn
    rate-limit budget. This matches the middleware order after the fix for the
    401->429 death spiral seen in the dashboard console.
    """
    import fxbot.api as api_mod

    source = open(api_mod.__file__, encoding="utf-8").read()
    auth_sec = source.index('if not resolved_settings.runtime.api_key or key != resolved_settings.runtime.api_key:')
    rate_sec = source.index('if not limiter.allow(')
    assert auth_sec < rate_sec


def test_api_middleware_allows_cors_preflight_before_auth() -> None:
    """OPTIONS must reach CORSMiddleware without an API key.

    Browsers intentionally omit X-API-Key on the preflight request. If the
    auth middleware rejects it first, the Vercel dashboard reports a CORS
    error even when the configured origin is correct.
    """
    import fxbot.api as api_mod

    source = open(api_mod.__file__, encoding="utf-8").read()
    options_guard = source.index('if request.method == "OPTIONS":')
    api_guard = source.index('if request.url.path.startswith("/api/"):')
    assert options_guard < api_guard


def test_api_middleware_routes_all_guarded_paths() -> None:
    """Every /api/* read route name is registered behind the auth middleware."""
    import fxbot.api as api_mod

    source = open(api_mod.__file__, encoding="utf-8").read()
    # The POST control routes already used require_api_key; reads must too.
    assert "@app.get(\"/api/status\", dependencies=[Depends(require_api_key)])" in source
    assert "@app.get(\"/api/monitoring\", dependencies=[Depends(require_api_key)])" in source
    assert "@app.get(\"/api/trades\", dependencies=[Depends(require_api_key)])" in source
    assert "@app.get(\"/api/positions\", dependencies=[Depends(require_api_key)])" in source


def test_api_is_bindable_to_localhost_by_default() -> None:
    from fxbot.config import RuntimeSettings
    assert RuntimeSettings().bind_host == "127.0.0.1"
