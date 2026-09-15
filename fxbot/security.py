"""Security, provenance, and operational-safety helpers.

These helpers deliberately have no broker or web-framework dependencies.  That
makes the safety rules usable from the worker, the API, and offline research
jobs without creating a second implementation of the same policy.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

log = logging.getLogger(__name__)

SENSITIVE_KEY_PARTS = (
    "password",
    "secret",
    "token",
    "api_key",
    "authorization",
    "account",
    "login",
    "broker_order_id",
    "broker_trade_id",
    "client_order_id",
)

# Keys that should never appear as a live API key in production.
_KNOWN_DEFAULT_KEYS = frozenset({"", "change-this-demo-control-key", "your_api_key_here", "changeme"})


def canonical_json(value: Any) -> str:
    """Return deterministic JSON suitable for hashing and manifests."""

    if is_dataclass(value):
        value = asdict(value)
    return json.dumps(value, default=str, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def strategy_config_hash(settings: Any) -> str:
    """Hash only strategy/risk/instrument settings, excluding credentials."""

    payload = asdict(settings) if is_dataclass(settings) else dict(settings)
    payload.pop("runtime", None)
    broker = payload.get("broker")
    if isinstance(broker, dict):
        for key in ("password", "login", "server", "terminal_path"):
            broker.pop(key, None)
    return sha256_json(payload)


def code_version(root: str | Path | None = None) -> str:
    """Return an explicit release id, or a stable source-tree fingerprint."""

    explicit = os.getenv("FX_CODE_VERSION", "").strip()
    if explicit:
        return explicit
    base = Path(root or Path(__file__).resolve().parents[1])
    digest = hashlib.sha256()
    for path in sorted(base.glob("fxbot/*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def data_hash(value: Any) -> str:
    """Hash a dataset descriptor or a file without loading it into memory."""

    if isinstance(value, (str, Path)) and Path(value).is_file():
        digest = hashlib.sha256()
        with Path(value).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    return sha256_json(value)


def experiment_manifest(*, strategy_hash: str, code: str, data: Any, splits: Any, parameters: Any) -> dict[str, Any]:
    manifest = {
        "strategy_hash": strategy_hash,
        "code_version": code,
        "data_hash": data_hash(data),
        "splits": splits,
        "parameters": parameters,
    }
    manifest["manifest_hash"] = sha256_json(manifest)
    return manifest


def redact_identifier(value: Any) -> str:
    """Keep enough suffix for correlation while avoiding account disclosure."""

    text = str(value or "")
    if not text:
        return ""
    return f"…{text[-4:]}" if len(text) > 4 else "[redacted]"


def redact(value: Any, *, key: str = "") -> Any:
    if isinstance(value, Mapping):
        result = {}
        for name, item in value.items():
            lowered = str(name).lower()
            if any(part in lowered for part in SENSITIVE_KEY_PARTS):
                result[str(name)] = redact_identifier(item)
            else:
                result[str(name)] = redact(item, key=lowered)
        return result
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value


def validate_startup_security(
    *,
    api_key: str,
    bind_host: str,
    demo_only: bool,
    live_release_approved: bool,
) -> list[str]:
    """Return a list of startup warnings.  Empty means all checks passed.

    This does NOT raise — the operator should read the warnings and decide.
    The goal is to make insecure defaults visible rather than silent.
    """
    warnings: list[str] = []
    if not api_key or api_key in _KNOWN_DEFAULT_KEYS:
        warnings.append(
            "FX_API_KEY is empty or a known default. "
            "All /api/* endpoints will reject requests. "
            "Set a strong, unique key before deploying."
        )
    if bind_host in {"0.0.0.0", "::"}:
        warnings.append(
            f"API is binding to {bind_host} (all interfaces). "
            "Use FX_API_HOST=127.0.0.1 unless you have a reverse proxy with TLS."
        )
    if not demo_only and not live_release_approved:
        warnings.append(
            "Live trading is enabled but the release gate "
            "(FX_LIVE_TRADING_ENABLED + FX_LIVE_RELEASE_ACK) is not approved. "
            "The worker will halt immediately on first scan."
        )
    for w in warnings:
        log.warning("[startup-security] %s", w)
    return warnings


class SlidingWindowRateLimiter:
    """Small process-local limiter for the single-process API deployment."""

    def __init__(self, limit: int = 120, window_seconds: float = 60.0) -> None:
        self.limit = max(1, int(limit))
        self.window_seconds = max(1.0, float(window_seconds))
        self._requests: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            bucket = self._requests.setdefault(key, deque())
            cutoff = now - self.window_seconds
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self.limit:
                return False
            bucket.append(now)
            if len(self._requests) > 2048:
                self._requests = {name: values for name, values in self._requests.items() if values}
            return True

