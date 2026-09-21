"""Provider-agnostic economic-calendar ingestion.

Architecture (news.txt spec):

    NewsProvider -> Normalizer -> Local Cache (SQLite) -> NewsService -> Risk/Strategy Engine

- NewsProvider is an abstraction over any calendar source (manual JSON file,
  HTTP API provider). The strategy is never hard-coded to one provider.
- All events are normalized into the canonical NewsEvent schema in config.py
  and deduplicated by event_id.
- Events are cached locally in SQLite so the trading engine does not depend on
  real-time API availability. The gateway refreshes the cache on a throttle.
- Fail-safe: an API outage must never be interpreted as "no news". When the
  cache has never been filled or is stale beyond a configured maximum age, the
  gateway reports stale=True so the risk engine can block new trades, log an
  explicit reason, and alert the operator.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
import urllib.request
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from zoneinfo import ZoneInfo

from fxbot.config import (
    NewsEvent,
    StrategySettings,
    _DATETIME_12H_FORMATS,
    _as_utc,
    _news_event_from_dict,
)
from fxbot.monitoring import OperationalMonitor


# Cached events are pruned once they are this far in the past. The FF thisweek
# feed spans a full week; keep recent history visible but bound table growth.
PRUNE_HORIZON = timedelta(hours=48)


def sqlite_path_from_url(database_url: str) -> str | None:
    """Return the filesystem path for a sqlite:/// URL, or None otherwise."""
    if not database_url.startswith("sqlite:///"):
        return None
    path = database_url.replace("sqlite:///", "", 1)
    return path if path else None


class NewsProvider(ABC):
    """Abstract economic-calendar provider. Implementations must be idempotent."""

    name: str = "generic"

    @abstractmethod
    def fetch(self) -> list[NewsEvent]:
        """Return normalized events for the configured horizon."""

    def validate_snapshot(self, events: list[NewsEvent], now: datetime) -> None:
        """Optionally reject expired coverage, even after a successful download."""


class NewsRateLimitError(RuntimeError):
    """A provider-requested cooldown; do not retry on every trading scan."""

    def __init__(self, retry_after_seconds: float) -> None:
        # Reject non-finite server values; bound hostile/accidental huge values
        # so datetime arithmetic cannot make the safety gate itself crash.
        if not math.isfinite(retry_after_seconds):
            retry_after_seconds = 300.0
        self.retry_after_seconds = min(7 * 86400.0, max(300.0, retry_after_seconds))
        super().__init__(f"news provider rate limited; retry after {self.retry_after_seconds:g}s")


class ManualJsonNewsProvider(NewsProvider):
    """Read a JSON list of events from a file (the manual/fallback provider)."""

    name = "manual-json"

    def __init__(self, path: str | Path, source_name: str | None = None) -> None:
        self.path = Path(path)
        self.source_name = source_name or f"file:{self.path.name}"

    def fetch(self) -> list[NewsEvent]:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{self.path} must contain a JSON list of news events")
        events = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                event = _news_event_from_dict(item)
            except ValueError:
                continue
            events.append(event)
        return events


class HttpNewsProvider(NewsProvider):
    """Generic HTTP economic-calendar provider backed by a JSON endpoint.

    Provider-agnostic: subclasses override ``extract_items`` to adapt a
    particular API's response shape to the canonical event dicts consumed by
    ``_news_event_from_dict``.
    """

    name = "http"
    strict_records = False
    minimum_retry_seconds = 0

    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str | None = None,
        api_key_header: str = "X-API-Key",
        timeout_seconds: int = 30,
        max_retries: int = 2,
    ) -> None:
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("news endpoint must be an absolute http(s) URL")
        if parsed.username or parsed.password:
            raise ValueError("news endpoint must not contain credentials")
        self.endpoint = endpoint
        self.api_key = api_key
        self.api_key_header = api_key_header
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, int(max_retries))

    def _request_headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers[self.api_key_header] = self.api_key
        return headers

    def fetch(self) -> list[NewsEvent]:
        request = urllib.request.Request(self.endpoint, headers=self._request_headers())
        payload: Any | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except HTTPError as exc:
                if exc.code == 429:
                    raw_delay = exc.headers.get("Retry-After", "300") if exc.headers else "300"
                    try:
                        delay = float(raw_delay)
                    except ValueError:
                        try:
                            delay = (parsedate_to_datetime(raw_delay) - datetime.now(timezone.utc)).total_seconds()
                        except (TypeError, ValueError, OverflowError):
                            delay = 300.0
                    raise NewsRateLimitError(delay) from exc
                # Permanent client errors fail immediately. Rate limits above
                # use the gateway cooldown rather than short in-request retries.
                if 400 <= exc.code < 500:
                    raise
                if attempt >= self.max_retries:
                    raise
                time.sleep(min(2.0, 0.25 * (2**attempt)))
            except (OSError, URLError, TimeoutError):
                if attempt >= self.max_retries:
                    raise
                time.sleep(min(2.0, 0.25 * (2**attempt)))
        if payload is None:
            raise ValueError(f"{self.name} provider returned no payload")
        row = self.extract_items(payload)
        if not isinstance(row, list):
            raise ValueError(f"{self.name} provider returned a non-list payload")
        events = []
        for item in row:
            if not isinstance(item, dict):
                if self.strict_records:
                    raise ValueError(f"{self.name} returned a non-object event")
                continue
            try:
                events.append(self.parse_item(item))
            except (ValueError, TypeError, AttributeError) as exc:
                if self.strict_records:
                    raise ValueError(f"{self.name} returned an invalid event") from exc
                continue
        return events

    def parse_item(self, item: dict[str, Any]) -> NewsEvent:
        return _news_event_from_dict(self.normalize_item(item))

    def extract_items(self, payload: Any) -> list[dict[str, Any]]:
        """Return the list of raw event dicts from the provider payload."""
        if isinstance(payload, dict):
            for key in ("events", "data", "list", "items", "results"):
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    return [c for c in candidate if isinstance(c, dict)]
        return [c for c in payload if isinstance(c, dict)] if isinstance(payload, list) else []

    def normalize_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Hook for adapting provider-specific field names to the canonical schema."""
        return {
            "name": item.get("name") or item.get("event_name") or item.get("title"),
            "currency": item.get("currency") or item.get("ccy"),
            "impact": item.get("impact") or item.get("importance") or "low",
            "starts_at": item.get("starts_at") or item.get("start") or item.get("date") or item.get("timestamp"),
            "ends_at": item.get("ends_at") or item.get("end") or item.get("starts_at") or item.get("start"),
            "event_id": item.get("event_id") or item.get("id"),
            "previous": item.get("previous"),
            "forecast": item.get("forecast") or item.get("consensus"),
            "actual": item.get("actual"),
            "description": item.get("description") or item.get("desc"),
            "source_url": item.get("source_url") or item.get("url"),
            "status": item.get("status") or "scheduled",
            "confidence": item.get("confidence"),
            "source": item.get("source") or self.name,
        }


class ForexFactoryProvider(HttpNewsProvider):
    """Economic-calendar provider that parses the public Forex Factory JSON.

    Source: ``https://nfs.faireconomy.media/ff_calendar_thisweek.json`` (and the
    ``..._nextweek.json`` variant). The public export is a flat list with
    ``title``, a currency in ``country``, and offset-aware ISO ``date`` values.
    Legacy nested ``days[].items[]`` exports remain supported. Neither an
    unsupported shape nor a partially invalid week is a known-empty calendar.
    This third-party calendar has no uptime/latency guarantee; demo use only.
    """

    name = "forexfactory"
    strict_records = True
    minimum_retry_seconds = 300

    def __init__(self, week_offset: int = 0, *, timeout_seconds: int = 30) -> None:
        week = "thisweek" if week_offset == 0 else ("nextweek" if week_offset > 0 else "lastweek")
        self.week_offset = 0 if week_offset == 0 else (1 if week_offset > 0 else -1)
        endpoint = f"https://nfs.faireconomy.media/ff_calendar_{week}.json"
        super().__init__(endpoint, timeout_seconds=timeout_seconds, max_retries=0)

    def extract_items(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict) and isinstance(payload.get("days"), list):
            items = []
            for day in payload["days"]:
                if not isinstance(day, dict) or not isinstance(day.get("items"), list):
                    raise ValueError("forexfactory returned an invalid day")
                items.extend(day["items"])
        else:
            raise ValueError("forexfactory returned an unsupported calendar shape")
        if not items or any(not isinstance(item, dict) for item in items):
            raise ValueError("forexfactory returned an empty or invalid weekly calendar")
        return items

    def parse_item(self, item: dict[str, Any]) -> NewsEvent:
        event = super().parse_item(item)
        if len(event.currency) != 3 or not event.currency.isalpha():
            raise ValueError("forexfactory event has an invalid currency")
        if event.impact not in {"high", "medium", "low", "holiday", "non-economic"}:
            raise ValueError("forexfactory event has an unknown impact")
        if not item.get("impact"):
            raise ValueError("forexfactory event is missing impact")
        raw_time = str(item.get("date") or "")
        if ":" not in raw_time:
            raise ValueError("forexfactory event is missing a release time")
        return event

    def validate_snapshot(self, events: list[NewsEvent], now: datetime) -> None:
        local = _as_utc(now).astimezone(_NY_TZ)
        start = local.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(
            days=(local.weekday() + 1) % 7
        ) + timedelta(weeks=self.week_offset)
        end = start + timedelta(days=7)
        if not any(
            event.source == self.name and start <= event.starts_at.astimezone(_NY_TZ) < end
            for event in events
        ):
            raise ValueError("forexfactory calendar does not cover the requested week")

    def normalize_item(self, item: dict[str, Any]) -> dict[str, Any]:
        country = item.get("country") or {}
        country_name = (
            country.get("name") if isinstance(country, dict) else str(country or "").strip()
        )
        currency = str(country.get("code") or "") if isinstance(country, dict) else str(country or "").strip()
        raw_time = str(item.get("date") or "").strip()
        event_id = str(item.get("id") or "").strip() or None
        if event_id:
            event_id = "".join(ch for ch in event_id if ch.isalnum() or ch in "-_")[:64]
        # Honor explicit offsets; legacy naive timestamps use New York time.
        starts_at_utc = _ff_time_to_utc_iso(raw_time)
        return {
            "name": item.get("title"),
            "currency": currency.upper(),
            "country": country_name,
            "impact": item.get("impact"),
            "starts_at": starts_at_utc,
            "ends_at": starts_at_utc,
            "event_id": event_id,
            "previous": _parse_pct_number(item.get("previous")),
            "forecast": _parse_pct_number(item.get("forecast")),
            "actual": _parse_pct_number(item.get("actual")),
            "description": item.get("description") or item.get("title") or "",
            "source_url": item.get("url") or "",
            "status": item.get("status") or "scheduled",
            "confidence": item.get("confidence"),
            "source": self.name,
        }


_SUFFIX_MULTIPLIERS = {"k": 1e3, "m": 1e6, "b": 1e9}

# The public export carries ISO offsets. Legacy 12-hour timestamps without
# an offset use America/New_York; never overwrite an explicit source offset.

_NY_TZ = ZoneInfo("America/New_York")


def _ff_time_to_utc_iso(raw_time: str) -> str:
    """Normalize an ISO-offset or legacy New York timestamp to UTC."""
    if not raw_time:
        return ""
    text = raw_time.strip()
    parsed: datetime | None = None
    for fmt in _DATETIME_12H_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
            break
        except (ValueError, TypeError):
            continue
    if parsed is None:
        try:
            parsed = datetime.fromisoformat(text)
        except (ValueError, TypeError):
            return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_NY_TZ)
    return parsed.astimezone(timezone.utc).isoformat()


def _parse_pct_number(value: Any) -> float | None:
    """Extract the numeric part of values like ``"3.1%"`` or ``"1.20k"``.

    Suffixes ``k`` (thousands), ``m`` (millions), ``b`` (billions) are
    multiplied out. A trailing ``%`` (with optional surrounding whitespace)
    is stripped without affecting the numeric value.
    """
    if value is None:
        return None
    # Numeric inputs pass straight through so a provider that already
    # delivers floats is never mangled by string round-tripping.
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "").lower()
    if not text or text in {"none", "n/a", "-"}:
        return None
    text = text.rstrip("%").strip()
    multiplier = 1.0
    if text and text[-1] in _SUFFIX_MULTIPLIERS:
        multiplier = _SUFFIX_MULTIPLIERS[text[-1]]
        text = text[:-1].strip()
    try:
        return float(text) * multiplier
    except (ValueError, TypeError):
        return None


# ── Step 4: impact scoring ──────────────────────────────────────────────────

# Base impact for well-known global macro events, 0-100. Providers that already
# tag HIGH/MEDIUM/LOW still get a minimum baseline so operators with no custom
# scoring can rely on the provider's own labels.
KNOWN_EVENT_BASE_SCORES: dict[str, int] = {
    "fomc": 100,
    "cpi": 95,
    "nfp": 95,
    "non-farm": 95,
    "nonfarm": 95,
    "rate decision": 90,
    "rba rate": 90,
    "ecb": 90,
    "boc": 90,
    "boj": 90,
    "m2 money supply": 40,
    "retail sales": 50,
    "unemployment": 85,
    "gdp": 70,
    "pmi": 50,
    "consumer confidence": 40,
    "trade balance": 40,
    "producer price": 45,
    "core cpi": 90,
    "initial claims": 60,
    "durable goods": 45,
    "manufacturing": 40,
    "housing starts": 40,
    "business inventories": 35,
    "existing home sales": 40,
    "new home sales": 35,
    "industrial production": 45,
    "capacity utilization": 35,
    "empire state": 35,
    "philadelphia fed": 40,
    "chicago pmi": 40,
}

CORE_IMPACT_BASE = {
    "high": 75,
    "medium": 40,
    "low": 10,
}

# Currencies whose macro events move the global FX market more than most.
MAJOR_CURRENCIES = {"USD", "EUR", "GBP", "JPY"}


@dataclass(frozen=True)
class ImpactScore:
    score: int
    level: str
    reason: str

    @property
    def high(self) -> bool:
        return self.score >= 71

    @property
    def medium(self) -> bool:
        return 31 <= self.score <= 70


def calculate_impact_score(event: NewsEvent) -> ImpactScore:
    """Score an event from 0-100 using event-name + currency importance.

    The provider's impact label supplies a baseline; a whitelist of well-known
    global macro releases overrides it with a bespoke score. The score is
    clamped to 0-100.
    """
    score = int(event.impact_score or 0)
    if score > 0:
        return _level_from_score(score, "provider")

    name_key = event.name.lower().removeprefix("us ").strip()
    known = next(
        (base for token, base in sorted(KNOWN_EVENT_BASE_SCORES.items(), key=lambda kv: (-kv[1], -len(kv[0]))) if token in name_key),
        None,
    )
    if known is not None:
        score = known
        reason = f"known_event:{event.name}"
    else:
        score = CORE_IMPACT_BASE.get(event.impact.lower(), 10)
        reason = f"provider_impact:{event.impact.lower()}"

    currency = event.currency.upper()
    if currency in MAJOR_CURRENCIES:
        score += 10
    score = max(0, min(100, score))
    return _level_from_score(score, reason)


def _level_from_score(score: int, reason: str) -> ImpactScore:
    if score >= 71:
        return ImpactScore(score, "HIGH", reason)
    if score >= 31:
        return ImpactScore(score, "MEDIUM", reason)
    return ImpactScore(score, "LOW", reason)


def score_events(events: list[NewsEvent]) -> list[NewsEvent]:
    """Attach a computed impact score to every event (in-place replacement)."""
    scored: list[NewsEvent] = []
    for event in events:
        computed = calculate_impact_score(event)
        scored.append(
            NewsEvent(
                name=event.name,
                currency=event.currency,
                impact=event.impact,
                starts_at=event.starts_at,
                ends_at=event.ends_at,
                event_id=event.event_id,
                country=event.country,
                previous=event.previous,
                forecast=event.forecast,
                actual=event.actual,
                source=event.source,
                created_at=event.created_at,
                updated_at=event.updated_at,
                impact_score=computed.score,
                description=event.description,
                source_url=event.source_url,
                status=event.status,
                confidence=event.confidence,
            )
        )
    return scored


def event_identity(event: NewsEvent) -> str:
    """Return a stable deduplication key even when a provider omits an ID."""

    return str(event.event_id or _event_fallback_key(event))


def deduplicate_events(events: list[NewsEvent]) -> list[NewsEvent]:
    """Deduplicate provider output before it reaches policy evaluation.

    A provider may repeat an event in overlapping pages or return the same
    release once with forecast data and once with actual data. The last record
    wins, which preserves the most recently normalized values while keeping the
    event set deterministic.
    """

    unique: dict[str, NewsEvent] = {}
    for event in events:
        unique[event_identity(event)] = event
    return sorted(unique.values(), key=lambda event: (_as_utc(event.starts_at), event_identity(event)))


class NewsCache:
    """Local SQLite cache of normalized news events, deduplicated by event_id.

    Uses its own ``news_cache_events`` table. The SQLAlchemy ``news_events``
    model in database.py is the canonical dashboard/journal table, so the raw
    cache table is named distinctly to avoid column collisions in the same file.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS news_cache_events (
        event_id   TEXT PRIMARY KEY,
        currency   TEXT NOT NULL,
        name       TEXT NOT NULL,
        impact     TEXT NOT NULL,
        starts_at  TEXT NOT NULL,
        ends_at    TEXT NOT NULL,
        country    TEXT NOT NULL DEFAULT '',
        impact_score INTEGER NOT NULL DEFAULT 0,
        description  TEXT NOT NULL DEFAULT '',
        source_url   TEXT NOT NULL DEFAULT '',
        status       TEXT NOT NULL DEFAULT 'scheduled',
        confidence   REAL,
        previous   REAL,
        forecast   REAL,
        actual     REAL,
        source     TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        fetched_at TEXT NOT NULL
    )
    """

    META_SCHEMA = """
    CREATE TABLE IF NOT EXISTS news_cache_meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """

    _EXTRA_COLUMNS = (
        "country",
        "impact_score",
        "description",
        "source_url",
        "status",
        "confidence",
    )

    def __init__(self, database_url: str, timeout_seconds: int = 30) -> None:
        path = sqlite_path_from_url(database_url)
        if path is None:
            raise ValueError(f"NewsCache requires a sqlite URL, got {database_url!r}")
        self.database_url = database_url
        self._path = path
        self._timeout_seconds = timeout_seconds
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            self._ensure_schema(connection)
        finally:
            connection.close()

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute(self.SCHEMA)
        connection.execute(self.META_SCHEMA)
        rows = connection.execute("PRAGMA table_info(news_cache_events)").fetchall()
        existing = {str(row[1]) for row in rows}
        for column in self._EXTRA_COLUMNS:
            if column in existing:
                continue
            if column == "country":
                connection.execute("ALTER TABLE news_cache_events ADD COLUMN country TEXT NOT NULL DEFAULT ''")
            elif column == "impact_score":
                connection.execute("ALTER TABLE news_cache_events ADD COLUMN impact_score INTEGER NOT NULL DEFAULT 0")
            elif column == "description":
                connection.execute("ALTER TABLE news_cache_events ADD COLUMN description TEXT NOT NULL DEFAULT ''")
            elif column == "source_url":
                connection.execute("ALTER TABLE news_cache_events ADD COLUMN source_url TEXT NOT NULL DEFAULT ''")
            elif column == "status":
                connection.execute("ALTER TABLE news_cache_events ADD COLUMN status TEXT NOT NULL DEFAULT 'scheduled'")
            elif column == "confidence":
                connection.execute("ALTER TABLE news_cache_events ADD COLUMN confidence REAL")
        connection.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, timeout=self._timeout_seconds)

    @contextmanager
    def _connection(self) -> Any:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def upsert_many(self, events: list[NewsEvent], fetched_at: datetime | None = None) -> int:
        """Insert-or-replace events; deduplication is by provider ID or fallback key."""
        now = _as_utc(fetched_at or datetime.now(timezone.utc))
        stamp = now.isoformat()
        events = deduplicate_events(events)
        with self._connection() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO news_cache_meta (key, value) VALUES ('last_fetched', ?)",
                (stamp,),
            )
            for event in events:
                connection.execute(
                    "INSERT OR REPLACE INTO news_cache_events "
                    "(event_id, currency, name, impact, starts_at, ends_at, country, impact_score, description, source_url, status, confidence, previous, forecast, actual, source, created_at, updated_at, fetched_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        event_identity(event),
                        event.currency,
                        event.name,
                        event.impact,
                        event.starts_at.isoformat(),
                        event.ends_at.isoformat(),
                        event.country or "",
                        event.impact_score or 0,
                        event.description,
                        event.source_url,
                        event.status,
                        event.confidence,
                        event.previous,
                        event.forecast,
                        event.actual,
                        event.source,
                        _as_utc(event.created_at or now).isoformat(),
                        _as_utc(event.updated_at or now).isoformat(),
                        stamp,
                    ),
                )
        return len(events)

    def all(self) -> list[NewsEvent]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT event_id, currency, name, impact, starts_at, ends_at, country, impact_score, description, source_url, status, confidence, previous, forecast, actual, source, created_at, updated_at FROM news_cache_events"
            ).fetchall()
        events = []
        for row in rows:
            try:
                events.append(
                    NewsEvent(
                        event_id=row[0],
                        currency=row[1],
                        name=row[2],
                        impact=row[3],
                        starts_at=datetime.fromisoformat(row[4]),
                        ends_at=datetime.fromisoformat(row[5]),
                        country=row[6],
                        impact_score=int(row[7] or 0),
                        description=row[8] or "",
                        source_url=row[9] or "",
                        status=row[10] or "scheduled",
                        confidence=None if row[11] is None else float(row[11]),
                        previous=row[12],
                        forecast=row[13],
                        actual=row[14],
                        source=row[15],
                        created_at=datetime.fromisoformat(row[16]),
                        updated_at=datetime.fromisoformat(row[17]),
                    )
                )
            except ValueError:
                continue
        return events

    def last_fetched(self) -> datetime | None:
        with self._connection() as connection:
            meta = connection.execute(
                "SELECT value FROM news_cache_meta WHERE key = 'last_fetched'"
            ).fetchone()
            row = connection.execute(
                "SELECT MAX(fetched_at) FROM news_cache_events"
            ).fetchone()
        raw = None
        if meta is not None and meta[0]:
            raw = str(meta[0])
        elif row is not None and row[0]:
            raw = str(row[0])
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None

    def count(self) -> int:
        with self._connection() as connection:
            row = connection.execute("SELECT COUNT(*) FROM news_cache_events").fetchone()
        return int(row[0] or 0)

    def clear(self) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM news_cache_events")
            connection.execute("DELETE FROM news_cache_meta")

    def prune(self, before: datetime | None = None) -> int:
        """Delete cached events that ended before *before* (default: 48h past).

        Keeps the table bounded so the blackout scan and /api/news listing do
        not grow without limit, while still retaining recent history.
        """
        cutoff = _as_utc(before or (datetime.now(timezone.utc) - PRUNE_HORIZON)).isoformat()
        with self._connection() as connection:
            cursor = connection.execute("DELETE FROM news_cache_events WHERE ends_at < ?", (cutoff,))
            return max(0, cursor.rowcount)


def _event_fallback_key(event: NewsEvent) -> str:
    return f"{event.currency}:{event.name}:{event.starts_at.isoformat()}"


@dataclass(frozen=True)
class NewsSnapshot:
    """Point-in-time view of news state handed to the risk/strategy engine."""

    events: list[NewsEvent] = field(default_factory=list)
    stale: bool = False
    last_updated: datetime | None = None
    age_seconds: float | None = None
    source: str = "empty"
    last_error: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.events) and not self.stale

    @property
    def known_state(self) -> str:
        if self.stale:
            return "UNKNOWN"
        return "FRESH" if self.events else "EMPTY"


class NewsGateway:
    """Orchestrates provider refresh, caching, staleness and operator alerts."""

    def __init__(
        self,
        *,
        settings: StrategySettings,
        provider: NewsProvider | None = None,
        cache: NewsCache | None = None,
        monitor: OperationalMonitor | None = None,
        static_events: list[NewsEvent] | None = None,
        refresh_throttle_seconds: int = 60,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.cache = cache
        self.monitor = monitor
        # Static events (manual JSON / config) are scored too so the blackout
        # threshold behaves identically regardless of the data source.
        self.static_events = score_events(list(static_events or []))
        self.refresh_throttle_seconds = max(1, refresh_throttle_seconds)
        self._lock = threading.Lock()
        self._last_refresh_at: datetime | None = None
        self._last_error: str | None = None
        self._retry_not_before: datetime | None = None
        # Respect a provider's minimum retry spacing; generic providers retain
        # the existing short recovery backoff.
        self._failure_backoff = max(
            min(30, self.refresh_throttle_seconds // 2),
            getattr(self.provider, "minimum_retry_seconds", 0),
        )

    @property
    def max_age_seconds(self) -> int:
        return self.settings.news_data_max_age_seconds

    def _age_seconds(self, last_updated: datetime, now: datetime) -> float:
        return max(0.0, (_as_utc(now) - _as_utc(last_updated)).total_seconds())

    def _stale(self, last_updated: datetime | None, now: datetime) -> bool:
        if last_updated is None:
            return True
        return self._age_seconds(last_updated, now) > self.max_age_seconds

    def _snapshot(self, now: datetime) -> NewsSnapshot:
        last_updated = self.cache.last_fetched() if self.cache is not None else None
        if last_updated is not None:
            # A successfully fetched cache -- even one whose current window
            # legitimately holds no events -- reports freshness by age, never as
            # an unexplained outage. Only a never-fetched cache is UNKNOWN.
            events = deduplicate_events(self.cache.all() if self.cache is not None else [])
            stale = self._stale(last_updated, now)
            error = self._last_error
            try:
                self._validate_snapshot(events, now)
            except ValueError as exc:
                stale = True
                error = str(exc)
            return NewsSnapshot(
                events=events,
                stale=stale,
                last_updated=last_updated,
                age_seconds=self._age_seconds(last_updated, now),
                source=self.provider.name if self.provider is not None else "cache",
                last_error=error,
            )
        if self.static_events and self.provider is None:
            # No persistent cache populated: the configured static snapshot is the
            # operator-provided source of truth only when no provider is wired
            # in. Static rows must not mask a live provider's initial failure.
            return NewsSnapshot(
                events=deduplicate_events(self.static_events),
                stale=False,
                last_updated=_as_utc(now),
                age_seconds=0.0,
                source="static",
                last_error=None,
            )
        return NewsSnapshot(
            events=[], stale=True,
            source=self.provider.name if self.provider is not None else "empty",
            last_error=self._last_error,
        )

    def _validate_snapshot(self, events: list[NewsEvent], now: datetime) -> None:
        validator = getattr(self.provider, "validate_snapshot", None)
        if validator is not None:
            validator(events, now)

    def _refresh_due(self, now: datetime) -> bool:
        if self.provider is None or self.cache is None:
            return False
        if self._retry_not_before is not None and _as_utc(now) < self._retry_not_before:
            return False
        if self._last_refresh_at is None:
            return True
        return (_as_utc(now) - self._last_refresh_at).total_seconds() >= self.refresh_throttle_seconds

    def ensure_current(self, now: datetime | None = None) -> NewsSnapshot:
        """Refresh the cache if due, then return the authoritative snapshot."""
        current = _as_utc(now or datetime.now(timezone.utc))
        with self._lock:
            if self._refresh_due(current):
                self._refresh_from_provider(current)
            snapshot = self._snapshot(current)
        self._record_monitor_state(snapshot)
        return snapshot

    def _refresh_from_provider(self, now: datetime) -> bool:
        """Attempt a provider refresh. Returns True when the cache was updated.

        Success advances the fetch clock. Failures leave cache freshness
        unchanged and schedule a provider-aware retry cooldown.
        """
        if self.provider is None or self.cache is None:
            return False
        if self._retry_not_before is not None and _as_utc(now) < self._retry_not_before:
            return False
        try:
            events = deduplicate_events(self.provider.fetch())
            self._validate_snapshot(events, now)
            # Step 4 (impact scoring): always compute our own impact score before
            # caching, so downstream consumers never depend on provider labels.
            self.cache.upsert_many(score_events(events), fetched_at=now)
            self.cache.prune(_as_utc(now) - PRUNE_HORIZON)
            self._last_refresh_at = _as_utc(now)
            self._last_error = None
            self._retry_not_before = None
        except Exception as exc:
            # Fail-safe: an API outage (or a malformed/unsupported payload) must
            # never be interpreted as "no news" and must never break the scan
            # cycle. Keep the last known events; the snapshot propagates the
            # UNKNOWN state via staleness, and the engine continues
            # monitoring/recovery. Re-arm the throttle so the next cycle retries
            # after a short backoff instead of the full throttle interval.
            current = _as_utc(now)
            backoff = self._failure_backoff
            if isinstance(exc, NewsRateLimitError):
                backoff = max(backoff, exc.retry_after_seconds)
            self._retry_not_before = current + timedelta(seconds=backoff)
            self._last_refresh_at = current - timedelta(seconds=self.refresh_throttle_seconds)
            self._last_error = f"{type(exc).__name__}: {exc}"
            if self.monitor is not None:
                last_updated = self.cache.last_fetched()
                stale = self._stale(last_updated, current)
                self.monitor.record_news_freshness(
                    None if last_updated is None else self._age_seconds(last_updated, current),
                    is_fresh=not stale,
                )
            return False
        if self.monitor is not None:
            last_updated = self.cache.last_fetched()
            self.monitor.record_news_freshness(
                None if last_updated is None else self._age_seconds(last_updated, now),
                is_fresh=True,
            )
        return True

    def _record_monitor_state(self, snapshot: NewsSnapshot) -> None:
        if self.monitor is None:
            return
        # An authoritative empty calendar is still fresh. ``available`` means
        # "contains at least one event" and must not be used as a freshness
        # signal, otherwise operators see a false stale alert on quiet days.
        self.monitor.record_news_freshness(snapshot.age_seconds, not snapshot.stale)

    def sync_upcoming(self, now: datetime | None = None) -> NewsSnapshot:
        """Force a provider refresh (regardless of throttle) and return the
        snapshot. Provider error/rate-limit cooldowns still apply."""
        current = _as_utc(now or datetime.now(timezone.utc))
        with self._lock:
            self._refresh_from_provider(current)
            snapshot = self._snapshot(current)
        self._record_monitor_state(snapshot)
        return snapshot


def build_news_gateway(
    settings: StrategySettings,
    *,
    database_url: str | None = None,
    monitor: OperationalMonitor | None = None,
    static_events: list[NewsEvent] | None = None,
) -> NewsGateway:
    """Construct a gateway from strategy settings, preferring the live provider.

    Provider precedence: HTTP API (if endpoint configured) -> manual JSON file
    (if file configured) -> ForexFactory (if opted in) -> static events
    snapshot passed by the caller.

    ``FX_USE_FOREX_FACTORY=false`` (default) -- the bot does not depend on the
    community Forex Factory JSON feed out of the box. Operators who want that
    free feed can opt in with ``FX_USE_FOREX_FACTORY=true``; anyone needing a
    licensed source should point ``FX_NEWS_API_ENDPOINT`` at it instead (that
    endpoint always takes precedence).
    """
    provider: NewsProvider | None = None
    cache: NewsCache | None = None
    if settings.news_api_endpoint:
        provider = HttpNewsProvider(
            settings.news_api_endpoint,
            api_key=settings.news_api_key or None,
            timeout_seconds=max(5, settings.news_http_timeout_seconds),
        )
    elif settings.news_events_file:
        provider = ManualJsonNewsProvider(settings.news_events_file)
    elif settings.news_use_forex_factory:
        provider = ForexFactoryProvider(timeout_seconds=max(5, settings.news_http_timeout_seconds))
    if database_url and sqlite_path_from_url(database_url):
        if provider is not None:
            cache = NewsCache(database_url)
    return NewsGateway(
        settings=settings,
        provider=provider,
        cache=cache,
        monitor=monitor,
        static_events=static_events,
        refresh_throttle_seconds=settings.news_sync_interval_seconds,
    )


# ── Step 7: Structured LLM sentiment analysis ───────────────────────────────


@dataclass(frozen=True)
class SentimentResult:
    """Structured output from LLM sentiment analysis of a news event.

    The LLM never trades directly -- its output becomes a feature consumed by
    the trading strategy through the risk manager.  For central-bank events the
    ``hawkish_score`` / ``dovish_score`` pair is typically more actionable than
    a simple bullish/bearish label.
    """

    event_id: str
    sentiment: str  # "BULLISH", "BEARISH", "NEUTRAL"
    confidence: float  # 0.0 - 1.0
    currency: str
    hawkish_score: float = 0.0  # 0.0 - 1.0
    dovish_score: float = 0.0  # 0.0 - 1.0
    relevance: float = 0.0  # 0.0 - 1.0
    summary: str = ""
    model: str = ""
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if self.hawkish_score < 0.0 or self.hawkish_score > 1.0:
            raise ValueError("hawkish_score must be between 0.0 and 1.0")
        if self.dovish_score < 0.0 or self.dovish_score > 1.0:
            raise ValueError("dovish_score must be between 0.0 and 1.0")
        if self.relevance < 0.0 or self.relevance > 1.0:
            raise ValueError("relevance must be between 0.0 and 1.0")
        valid_sentiments = {"BULLISH", "BEARISH", "NEUTRAL"}
        if self.sentiment.upper() not in valid_sentiments:
            raise ValueError(f"sentiment must be one of {valid_sentiments}")
        object.__setattr__(self, "sentiment", self.sentiment.upper())
        object.__setattr__(self, "currency", self.currency.upper())


# ── Step 8: Sentiment cache (SQLite) ────────────────────────────────────────


class SentimentCache:
    """Cache for LLM sentiment analysis results keyed by event_id.

    Prevents redundant LLM calls for the same news article and reduces both
    latency and cost.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS news_sentiment (
        event_id      TEXT PRIMARY KEY,
        sentiment     TEXT NOT NULL,
        confidence    NUMERIC NOT NULL,
        currency      TEXT NOT NULL DEFAULT '',
        hawkish_score NUMERIC NOT NULL DEFAULT 0,
        dovish_score  NUMERIC NOT NULL DEFAULT 0,
        relevance     NUMERIC NOT NULL DEFAULT 0,
        analysis      TEXT NOT NULL DEFAULT '',
        model         TEXT NOT NULL DEFAULT '',
        created_at    TIMESTAMP NOT NULL
    )
    """

    def __init__(self, database_url: str, timeout_seconds: int = 30) -> None:
        path = sqlite_path_from_url(database_url)
        if path is None:
            raise ValueError(f"SentimentCache requires a sqlite URL, got {database_url!r}")
        self._path = path
        self._timeout_seconds = timeout_seconds
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            connection.execute(self.SCHEMA)
            rows = connection.execute("PRAGMA table_info(news_sentiment)").fetchall()
            existing = {str(row[1]) for row in rows}
            if "currency" not in existing:
                connection.execute("ALTER TABLE news_sentiment ADD COLUMN currency TEXT NOT NULL DEFAULT ''")
            connection.commit()
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, timeout=self._timeout_seconds)

    @contextmanager
    def _connection(self) -> Any:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def get(self, event_id: str) -> SentimentResult | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT event_id, sentiment, confidence, currency, hawkish_score, "
                "dovish_score, relevance, analysis, model, created_at "
                "FROM news_sentiment WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        if row is None:
            return None
        return SentimentResult(
            event_id=row[0],
            sentiment=row[1],
            confidence=float(row[2]),
            currency=row[3] or "",
            hawkish_score=float(row[4]),
            dovish_score=float(row[5]),
            relevance=float(row[6]),
            summary=row[7],
            model=row[8],
            created_at=datetime.fromisoformat(row[9]) if row[9] else None,
        )

    def upsert(self, result: SentimentResult) -> None:
        now = _as_utc(result.created_at or datetime.now(timezone.utc))
        with self._connection() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO news_sentiment "
                "(event_id, sentiment, confidence, currency, hawkish_score, dovish_score, "
                "relevance, analysis, model, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    result.event_id,
                    result.sentiment,
                    result.confidence,
                    result.currency,
                    result.hawkish_score,
                    result.dovish_score,
                    result.relevance,
                    result.summary,
                    result.model,
                    now.isoformat(),
                ),
            )

    def has(self, event_id: str) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM news_sentiment WHERE event_id = ?", (event_id,)
            ).fetchone()
        return row is not None

    def count(self) -> int:
        with self._connection() as connection:
            row = connection.execute("SELECT COUNT(*) FROM news_sentiment").fetchone()
        return int(row[0] or 0)

    def clear(self) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM news_sentiment")


# ── Step 9: Scheduled vs unscheduled news classifier ────────────────────────

# Events whose names contain these tokens are considered scheduled calendar
# releases.  Everything else (war, emergency decisions, bank collapses, etc.)
# is treated as unscheduled / surprise news.
_SCHEDULED_KEYWORDS: frozenset[str] = frozenset({
    "cpi", "nfp", "non-farm", "nonfarm", "fomc", "gdp",
    "rate decision", "interest rate", "retail sales", "pmi",
    "unemployment", "consumer confidence", "trade balance",
    "durable goods", "housing starts", "existing home sales",
    "new home sales", "industrial production", "capacity utilization",
    "empire state", "philadelphia fed", "chicago pmi",
    "producer price", "core cpi", "initial claims",
    "m2 money supply", "ecb", "boc", "boj", "rba",
    "boe rate", "snb", "retail sales",
})


def _is_scheduled_event(event: NewsEvent) -> bool:
    """Determine whether an event is a known scheduled calendar release."""
    lower_name = event.name.lower()
    # "Emergency rate decision" / "Emergency rate cut" are unscheduled surprises,
    # even though they contain the word "rate decision".
    if "emergency" in lower_name:
        return False
    return any(token in lower_name for token in _SCHEDULED_KEYWORDS)


@dataclass(frozen=True)
class NewsClassification:
    """Classification result for a single news event."""

    event: NewsEvent
    scheduled: bool
    category: str  # "scheduled" or "unscheduled"


class NewsClassifier:
    """Separates events into scheduled calendar releases and unscheduled surprises.

    Scheduled events (CPI, NFP, FOMC, ...) can be handled via the calendar ->
    risk-manager path.  Unscheduled events (war, emergency rate decisions,
    political announcements) require a news-feed -> classifier -> sentiment ->
    risk-manager path.
    """

    def classify(self, events: list[NewsEvent]) -> list[NewsClassification]:
        results: list[NewsClassification] = []
        for event in events:
            scheduled = _is_scheduled_event(event)
            results.append(
                NewsClassification(
                    event=event,
                    scheduled=scheduled,
                    category="scheduled" if scheduled else "unscheduled",
                )
            )
        return results

    def scheduled_only(self, events: list[NewsEvent]) -> list[NewsEvent]:
        return [c.event for c in self.classify(events) if c.scheduled]

    def unscheduled_only(self, events: list[NewsEvent]) -> list[NewsEvent]:
        return [c.event for c in self.classify(events) if not c.scheduled]


# ── Step 10: Post-news volatility detection ─────────────────────────────────

VOLATILITY_NORMAL = "normal"
VOLATILITY_ELEVATED = "elevated"
VOLATILITY_EXTREME = "extreme"

_VOLATILITY_LABELS = {
    VOLATILITY_NORMAL,
    VOLATILITY_ELEVATED,
    VOLATILITY_EXTREME,
}


@dataclass(frozen=True)
class VolatilityReading:
    """Point-in-time volatility measurement relative to a baseline ATR."""

    ratio: float  # current_atr / baseline_atr
    level: str  # "normal", "elevated", "extreme"
    baseline_atr: float
    current_atr: float

    @property
    def is_normal(self) -> bool:
        return self.level == VOLATILITY_NORMAL

    @property
    def is_elevated(self) -> bool:
        return self.level == VOLATILITY_ELEVATED

    @property
    def is_extreme(self) -> bool:
        return self.level == VOLATILITY_EXTREME


class VolatilityDetector:
    """Measure post-news volatility by comparing current ATR to a baseline.

    Thresholds (from the spec):
        ratio < 1.5   -> normal
        1.5 - 2.5     -> elevated
        > 2.5          -> extreme
    """

    def __init__(
        self,
        *,
        elevated_threshold: float = 1.5,
        extreme_threshold: float = 2.5,
    ) -> None:
        if elevated_threshold <= 0:
            raise ValueError("elevated_threshold must be positive")
        if extreme_threshold <= elevated_threshold:
            raise ValueError("extreme_threshold must exceed elevated_threshold")
        self._elevated = elevated_threshold
        self._extreme = extreme_threshold

    def measure(self, baseline_atr: float, current_atr: float) -> VolatilityReading:
        """Compute the volatility ratio and classify it."""
        if baseline_atr <= 0:
            raise ValueError("baseline_atr must be positive")
        ratio = current_atr / baseline_atr
        if ratio >= self._extreme:
            level = VOLATILITY_EXTREME
        elif ratio >= self._elevated:
            level = VOLATILITY_ELEVATED
        else:
            level = VOLATILITY_NORMAL
        return VolatilityReading(
            ratio=ratio,
            level=level,
            baseline_atr=baseline_atr,
            current_atr=current_atr,
        )

    def is_safe_to_trade(self, baseline_atr: float, current_atr: float) -> bool:
        """Return True when volatility has normalized enough to trade."""
        return self.measure(baseline_atr, current_atr).is_normal


# ── Step 11: Post-news trading state machine ────────────────────────────────

from enum import Enum


class NewsTradingState(str, Enum):
    """States in the post-news trading lifecycle.

    NORMAL -> PRE_NEWS -> NEWS_LOCK -> POST_NEWS -> TRADE -> NORMAL
                                       ^                      |
                                       |______________________|
    """

    NORMAL = "NORMAL"
    PRE_NEWS = "PRE_NEWS"
    NEWS_LOCK = "NEWS_LOCK"
    POST_NEWS = "POST_NEWS"
    TRADE = "TRADE"


@dataclass
class NewsTradingStateMachine:
    """Drives the trading state through the post-news lifecycle.

    Transitions:
        NORMAL   -- 15 min before news --> PRE_NEWS
        PRE_NEWS -- news occurs       --> NEWS_LOCK
        NEWS_LOCK -- volatility drops  --> POST_NEWS
        POST_NEWS -- setup confirmed   --> TRADE
        TRADE    -- time passes        --> NORMAL
    """

    pre_news_minutes: float = 15.0
    min_lock_minutes: float = 1.0
    max_lock_minutes: float = 5.0
    _state: NewsTradingState = field(default=NewsTradingState.NORMAL, init=False)
    _news_start: datetime | None = field(default=None, init=False)
    _state_entered_at: datetime | None = field(default=None, init=False)

    @property
    def state(self) -> NewsTradingState:
        return self._state

    @property
    def seconds_in_state(self) -> float | None:
        if self._state_entered_at is None:
            return None
        now = datetime.now(timezone.utc)
        return (now - _as_utc(self._state_entered_at)).total_seconds()

    def _transition(self, new_state: NewsTradingState, now: datetime) -> None:
        self._state = new_state
        self._state_entered_at = _as_utc(now)

    def check_transition(
        self,
        now: datetime,
        *,
        upcoming_event_start: datetime | None = None,
        volatility: VolatilityReading | None = None,
        setup_confirmed: bool = False,
    ) -> NewsTradingState:
        """Evaluate whether a state transition should occur and apply it.

        Call this on each scan cycle with the current market context.
        Returns the (possibly updated) state after evaluation.
        """
        current = _as_utc(now)

        if self._state == NewsTradingState.NORMAL:
            if upcoming_event_start is not None:
                minutes_until = (upcoming_event_start - current).total_seconds() / 60.0
                if 0 < minutes_until <= self.pre_news_minutes:
                    self._transition(NewsTradingState.PRE_NEWS, current)
                    return self._state

        elif self._state == NewsTradingState.PRE_NEWS:
            if upcoming_event_start is not None and current >= upcoming_event_start:
                self._transition(NewsTradingState.NEWS_LOCK, current)
                self._news_start = current
                return self._state
            if upcoming_event_start is None:
                # Event disappeared (cancelled / not found) -- revert to NORMAL.
                self._transition(NewsTradingState.NORMAL, current)
                return self._state
            minutes_until = (upcoming_event_start - current).total_seconds() / 60.0
            if minutes_until > self.pre_news_minutes:
                self._transition(NewsTradingState.NORMAL, current)
                return self._state

        elif self._state == NewsTradingState.NEWS_LOCK:
            lock_elapsed = self._minutes_since_state_entered(current)
            if lock_elapsed < self.min_lock_minutes:
                return self._state
            if volatility is not None and volatility.is_normal:
                self._transition(NewsTradingState.POST_NEWS, current)
                return self._state
            if lock_elapsed >= self.max_lock_minutes:
                self._transition(NewsTradingState.POST_NEWS, current)
                return self._state

        elif self._state == NewsTradingState.POST_NEWS:
            if setup_confirmed:
                self._transition(NewsTradingState.TRADE, current)
                return self._state

        elif self._state == NewsTradingState.TRADE:
            self._transition(NewsTradingState.NORMAL, current)
            return self._state

        return self._state

    def force_lock(self, now: datetime) -> None:
        """Force the state machine into NEWS_LOCK (e.g. for surprise news)."""
        self._transition(NewsTradingState.NEWS_LOCK, _as_utc(now))
        self._news_start = _as_utc(now)

    def reset(self, now: datetime) -> None:
        """Reset to NORMAL state."""
        self._transition(NewsTradingState.NORMAL, _as_utc(now))
        self._news_start = None

    def _minutes_since_state_entered(self, now: datetime) -> float:
        if self._state_entered_at is None:
            return float("inf")
        return (now - _as_utc(self._state_entered_at)).total_seconds() / 60.0

    def _as_dict(self) -> dict[str, Any]:
        """Serialisable snapshot for monitoring / API."""
        return {
            "state": self._state.value,
            "news_start": self._news_start.isoformat() if self._news_start else None,
            "state_entered_at": self._state_entered_at.isoformat() if self._state_entered_at else None,
            "seconds_in_state": self.seconds_in_state,
        }


# ── Step 6 + 12: Centralised NewsRiskManager ────────────────────────────────


@dataclass(frozen=True)
class NewsRiskDecision:
    """Structured decision returned by NewsRiskManager.evaluate()."""

    blocked: bool
    reason: str
    events: list[NewsEvent] = field(default_factory=list)
    state: NewsTradingState = NewsTradingState.NORMAL
    volatility: VolatilityReading | None = None

    def __bool__(self) -> bool:
        return self.blocked


class NewsRiskManager:
    """Single entry-point for all news-related risk decisions.

    Replaces scattered blackout conditions throughout the strategy with one
    clean call::

        decision = news_risk.evaluate("EURUSD", now, volatility=..., setup_confirmed=...)
        if decision.blocked:
            return None

    Combines:
      - Blackout window evaluation (before/after high-impact events)
      - Currency-pair relevance filtering
      - Post-news state machine (no immediate trading after announcements)
      - Volatility guard (don't trade when spreads are blown out)
      - Minimum post-announcement cooldown
    """

    def __init__(
        self,
        *,
        blackout_before_minutes: int = 30,
        blackout_after_minutes: int = 30,
        impact_score_min: int = 71,
        min_post_news_minutes: float = 3.0,
        max_spread_pips: float = 3.0,
        fail_closed: bool = False,
        volatility_detector: VolatilityDetector | None = None,
        state_machine: NewsTradingStateMachine | None = None,
        classifier: NewsClassifier | None = None,
    ) -> None:
        self.blackout_before_minutes = blackout_before_minutes
        self.blackout_after_minutes = blackout_after_minutes
        self.impact_score_min = impact_score_min
        self.min_post_news_minutes = min_post_news_minutes
        self.max_spread_pips = max_spread_pips
        # Fail-closed kill switch (Step 17): when the calendar status is
        # UNKNOWN the bot must not open new positions, rather than assuming
        # "no news" and trading normally.
        self.fail_closed = fail_closed
        self.volatility_detector = volatility_detector or VolatilityDetector()
        self.state_machine = state_machine or NewsTradingStateMachine()
        self.classifier = classifier or NewsClassifier()

    def get_relevant_events(
        self,
        symbol: str,
        events: list[NewsEvent],
    ) -> list[NewsEvent]:
        """Return events that affect the given symbol within the blackout window."""
        from fxbot.instruments import split_instrument_name

        base, quote = split_instrument_name(symbol)
        currencies = {base, quote}
        relevant: list[NewsEvent] = []
        for event in events:
            score = event.impact_score or _fallback_score(event.impact)
            if score < self.impact_score_min:
                continue
            if event.currency.upper() not in currencies:
                continue
            relevant.append(event)
        return relevant

    def evaluate(
        self,
        symbol: str,
        now: datetime,
        events: list[NewsEvent] | None = None,
        *,
        news_snapshot: NewsSnapshot | None = None,
        current_spread_pips: float | None = None,
        baseline_atr: float | None = None,
        current_atr: float | None = None,
        setup_confirmed: bool = False,
    ) -> NewsRiskDecision:
        """Evaluate whether trading is blocked for *symbol* at *now*.

        ``news_snapshot`` supplies the fail-closed status: when ``fail_closed``
        is enabled and the snapshot is UNKNOWN this returns blocked with reason
        ``news_status_unknown`` so a calendar outage is never read as "no news".

        Returns a ``NewsRiskDecision`` with ``blocked=True`` and a human-readable
        ``reason`` when any guard fires, or ``blocked=False`` when trading is
        permitted.
        """
        current = _as_utc(now)
        events = events or []
        relevant = self.get_relevant_events(symbol, events)

        # --- Fail-closed kill switch (Step 17) ---
        # Deterministic order: an unknown calendar blocks first, regardless of
        # what the potentially-empty event list otherwise implies.
        status = news_status_from_snapshot(news_snapshot)
        if self.fail_closed and status == NEWS_STATUS_UNKNOWN:
            return NewsRiskDecision(
                blocked=True,
                reason="news_status_unknown",
                events=relevant,
                state=self.state_machine.state,
            )

        # --- Blackout window check ---
        blackout = self._check_blackout(symbol, relevant, current)
        if blackout is not None:
            return NewsRiskDecision(
                blocked=True,
                reason=blackout,
                events=relevant,
                state=self.state_machine.state,
            )

        # --- Post-news cooldown (Step 12) ---
        cooldown = self._check_post_news_cooldown(relevant, current)
        if cooldown is not None:
            return NewsRiskDecision(
                blocked=True,
                reason=cooldown,
                events=relevant,
                state=self.state_machine.state,
            )

        # --- Spread check ---
        if current_spread_pips is not None and current_spread_pips > self.max_spread_pips:
            return NewsRiskDecision(
                blocked=True,
                reason=f"spread_too_wide:{current_spread_pips:.1f}>{self.max_spread_pips:.1f}",
                events=relevant,
                state=self.state_machine.state,
            )

        # --- Volatility check ---
        volatility: VolatilityReading | None = None
        if baseline_atr is not None and current_atr is not None and baseline_atr > 0:
            volatility = self.volatility_detector.measure(baseline_atr, current_atr)
            if volatility.is_extreme:
                return NewsRiskDecision(
                    blocked=True,
                    reason=f"volatility_extreme:ratio={volatility.ratio:.1f}",
                    events=relevant,
                    state=self.state_machine.state,
                    volatility=volatility,
                )

        # --- State machine transition ---
        upcoming = self._next_upcoming_event(relevant, current)
        upcoming_start = upcoming.starts_at if upcoming else None
        self.state_machine.check_transition(
            current,
            upcoming_event_start=upcoming_start,
            volatility=volatility,
            setup_confirmed=setup_confirmed,
        )

        if self.state_machine.state == NewsTradingState.NEWS_LOCK:
            return NewsRiskDecision(
                blocked=True,
                reason="news_lock",
                events=relevant,
                state=self.state_machine.state,
                volatility=volatility,
            )

        return NewsRiskDecision(
            blocked=False,
            reason="ok",
            events=relevant,
            state=self.state_machine.state,
            volatility=volatility,
        )

    def _check_blackout(
        self,
        symbol: str,
        relevant_events: list[NewsEvent],
        now: datetime,
    ) -> str | None:
        """Check if now falls within a blackout window for any relevant event."""
        before = timedelta(minutes=max(0, self.blackout_before_minutes))
        after = timedelta(minutes=max(0, self.blackout_after_minutes))
        for event in relevant_events:
            score = event.impact_score or _fallback_score(event.impact)
            if score < self.impact_score_min:
                continue
            start = _as_utc(event.starts_at) - before
            end = _as_utc(event.ends_at) + after
            if start <= now <= end:
                return f"HIGH_IMPACT_NEWS:{event.currency}:{event.name}:score={score}"
        return None

    def _check_post_news_cooldown(
        self,
        relevant_events: list[NewsEvent],
        now: datetime,
    ) -> str | None:
        """Don't trade in the first N minutes after a high-impact announcement.

        This prevents entering during wild spread/liquidity conditions right
        after the release.
        """
        cooldown = timedelta(minutes=max(0, self.min_post_news_minutes))
        for event in relevant_events:
            event_end = _as_utc(event.ends_at)
            if event_end <= now < event_end + cooldown:
                minutes_since = (now - event_end).total_seconds() / 60.0
                return (
                    f"post_news_cooldown:{event.currency}:{event.name}:"
                    f"{minutes_since:.1f}min<{self.min_post_news_minutes:.1f}min"
                )
        return None

    def _next_upcoming_event(
        self,
        relevant_events: list[NewsEvent],
        now: datetime,
    ) -> NewsEvent | None:
        """Return the next relevant event that hasn't started yet."""
        upcoming = [e for e in relevant_events if _as_utc(e.starts_at) > now]
        if not upcoming:
            return None
        return min(upcoming, key=lambda e: _as_utc(e.starts_at))


# ── Step 17: Fail-closed kill switch ─────────────────────────────────────────

# News state drawn from a NewsSnapshot / NewsGateway.  Mirrors the spec's
# news_status taxonomy:
#   "FRESH"   -- calendar data has been fetched recently enough to be trusted.
#   "EMPTY"   -- fetched recently but the window legitimately holds no events.
#   "UNKNOWN" -- the calendar has never been fetched, or is stale beyond the
#                configured max age, or the provider is down.
NEWS_STATUS_FRESH = "FRESH"
NEWS_STATUS_EMPTY = "EMPTY"
NEWS_STATUS_UNKNOWN = "UNKNOWN"


def news_status_from_snapshot(snapshot: NewsSnapshot | None) -> str:
    """Classify an authoritative news snapshot into a fail-closed status."""
    if snapshot is None:
        return NEWS_STATUS_UNKNOWN
    if snapshot.stale:
        return NEWS_STATUS_UNKNOWN
    return NEWS_STATUS_FRESH if snapshot.events else NEWS_STATUS_EMPTY


def news_known(status: str) -> bool:
    """Return True when the calendar is in a known state (FRESH or EMPTY)."""
    return status != NEWS_STATUS_UNKNOWN


# ── Step 13: News as a feature for the existing strategy ─────────────────────

_BULLISH = "BULLISH"
_BEARISH = "BEARISH"
_NEUTRAL = "NEUTRAL"


def sentiment_contribution(sentiment: str) -> int:
    """Map a sentiment label onto a +1/0/-1 feature contribution.

    Per the spec: BULLISH -> +1, BEARISH -> -1, NEUTRAL -> 0.  This is the
    signal-level contribution added to the technical/trend/momentum score --
    it must always be backtested before being assigned weights in production.
    """
    label = (sentiment or "").upper()
    if label == _BULLISH:
        return 1
    if label == _BEARISH:
        return -1
    return 0


@dataclass(frozen=True)
class NewsFeatureResult:
    """Outcome of applying news sentiment as a feature to a symbol."""

    contribution: int  # -1, 0, +1
    sentiment: str | None
    confidence: float
    currency: str | None
    applied: bool  # True when a sentiment result matched the symbol
    reason: str


class NewsFeatureEngine:
    """Turn LLM sentiment into a composable feature for the existing strategy.

    The spec's signal combination becomes::

        score = technical_score + trend_score + momentum_score
        feature = news_feature.apply("EUR_USD", sentiment_results)
        score += feature.contribution

        if news_risk.blocked:
            return None
        if score >= 4:
            return BUY
        if score <= -4:
            return SELL

    The contribution is deliberately tiny (1 point) so one LLM label can never
    dominate the technical score; its real weight belongs in a backtest.
    """

    def apply(
        self,
        symbol: str,
        sentiment_results: list[SentimentResult],
    ) -> NewsFeatureResult:
        from fxbot.instruments import split_instrument_name

        base, quote = split_instrument_name(symbol)
        currencies = {base, quote}
        best: SentimentResult | None = None
        for result in sentiment_results:
            if result.currency.upper() not in currencies:
                continue
            if best is None or result.confidence > best.confidence:
                best = result
        if best is None:
            return NewsFeatureResult(
                contribution=0,
                sentiment=None,
                confidence=0.0,
                currency=None,
                applied=False,
                reason=f"no_sentiment_for_symbol:{symbol}",
            )
        contribution = sentiment_contribution(best.sentiment)
        return NewsFeatureResult(
            contribution=contribution,
            sentiment=best.sentiment,
            confidence=best.confidence,
            currency=best.currency,
            applied=True,
            reason=f"sentiment:{best.sentiment}:confidence={best.confidence:.2f}",
        )


# ── Step 18: News audit logging ──────────────────────────────────────────────

# Structured event-type prefixes for the news audit trail (Step 18 spec).
NEWS_AUDIT_EVENT = "news_audit.event"
NEWS_AUDIT_FORECAST = "news_audit.forecast"
NEWS_AUDIT_ACTUAL = "news_audit.actual"
NEWS_AUDIT_BLOCKED = "news_audit.blocked"
NEWS_AUDIT_UNLOCKED = "news_audit.unlocked"
NEWS_AUDIT_STATE = "news_audit.state"
NEWS_AUDIT_SENTIMENT = "news_audit.sentiment"
NEWS_AUDIT_SIGNAL = "news_audit.signal"
NEWS_AUDIT_ORDER = "news_audit.order"
NEWS_AUDIT_KILL_SWITCH = "news_audit.kill_switch"


class NewsAuditRecorder:
    """Structured audit logger for the full news lifecycle.

    Wraps a ``StructuredJournal`` (or anything with ``log_event``) and emits the
    timeline described in Step 18 -- event announced, trading blocked, actual
    released, spread/ATR measurements, unlock, final combined score, and order
    execution -- so debugging is far easier.
    """

    def __init__(self, journal: Any) -> None:
        self.journal = journal

    def log_event_released(self, event: NewsEvent, *, scheduled: bool = True) -> None:
        self.journal.log_event(
            NEWS_AUDIT_EVENT,
            f"NEWS EVENT {event.name} ({event.currency}), impact={event.impact}, scheduled={scheduled}",
            payload={
                "name": event.name,
                "currency": event.currency,
                "impact": event.impact,
                "impact_score": event.impact_score,
                "starts_at": _as_utc(event.starts_at).isoformat(),
                "ends_at": _as_utc(event.ends_at).isoformat(),
                "source": event.source,
                "scheduled": scheduled,
            },
        )

    def log_forecast(self, event: NewsEvent) -> None:
        if event.forecast is None:
            return
        self.journal.log_event(
            NEWS_AUDIT_FORECAST,
            f"{event.name} forecast: {event.forecast}",
            payload={"currency": event.currency, "forecast": event.forecast},
        )

    def log_previous(self, event: NewsEvent) -> None:
        if event.previous is None:
            return
        self.journal.log_event(
            NEWS_AUDIT_FORECAST,
            f"{event.name} previous: {event.previous}",
            payload={"currency": event.currency, "previous": event.previous},
        )

    def log_actual(self, event: NewsEvent) -> None:
        if event.actual is None:
            return
        self.journal.log_event(
            NEWS_AUDIT_ACTUAL,
            f"{event.name} actual: {event.actual}",
            payload={"currency": event.currency, "actual": event.actual},
        )

    def log_blocked(
        self,
        symbol: str,
        reason: str,
        *,
        event: NewsEvent | None = None,
        additional: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {"symbol": symbol, "reason": reason}
        if event is not None:
            payload["currency"] = event.currency
            payload["event"] = event.name
        if additional:
            payload.update(additional)
        self.journal.log_event(
            NEWS_AUDIT_BLOCKED,
            f"{symbol} trading blocked: {reason}",
            payload=payload,
        )

    def log_unlocked(
        self,
        symbol: str,
        *,
        spread_pips: float | None = None,
        atr_ratio: float | None = None,
        reason: str = "conditions_normal",
        state: NewsTradingState | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "symbol": symbol,
            "reason": reason,
            "spread_pips": spread_pips,
            "atr_ratio": atr_ratio,
            "state": state.value if state else None,
        }
        self.journal.log_event(
            NEWS_AUDIT_UNLOCKED,
            f"{symbol} trading unlocked: {reason}",
            payload=payload,
        )

    def log_state_transition(self, symbol: str, to_state: NewsTradingState, from_state: NewsTradingState | None = None) -> None:
        self.journal.log_event(
            NEWS_AUDIT_STATE,
            f"{symbol} news state -> {to_state.value}",
            payload={
                "symbol": symbol,
                "from": from_state.value if from_state else None,
                "to": to_state.value,
            },
        )

    def log_sentiment(self, result: SentimentResult) -> None:
        self.journal.log_event(
            NEWS_AUDIT_SENTIMENT,
            f"{result.currency} news sentiment: {result.sentiment} (confidence={result.confidence:.2f})",
            payload={
                "event_id": result.event_id,
                "sentiment": result.sentiment,
                "confidence": result.confidence,
                "hawkish_score": result.hawkish_score,
                "dovish_score": result.dovish_score,
                "relevance": result.relevance,
                "summary": result.summary,
                "model": result.model,
            },
        )

    def log_final_signal(
        self,
        symbol: str,
        *,
        technical_signal: str,
        news_contribution: int,
        final_score: int,
        side: str | None,
        threshold: int = 4,
    ) -> None:
        self.journal.log_event(
            NEWS_AUDIT_SIGNAL,
            f"{symbol} technical={technical_signal} news_contribution={news_contribution:+d} final_score={final_score}",
            payload={
                "symbol": symbol,
                "technical_signal": technical_signal,
                "news_contribution": news_contribution,
                "final_score": final_score,
                "side": side,
                "threshold": threshold,
            },
        )

    def log_order_executed(
        self,
        symbol: str,
        *,
        side: str,
        units: float,
        entry_price: float,
        client_order_id: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "units": units,
            "entry_price": entry_price,
            "client_order_id": client_order_id,
        }
        self.journal.log_event(
            NEWS_AUDIT_ORDER,
            f"ORDER EXECUTED {symbol} {side} {units} units @ {entry_price}",
            payload=payload,
        )
