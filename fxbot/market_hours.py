"""FX session, weekend, rollover, and news blackout guards."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from fxbot.config import NewsEvent, StrategySettings
from fxbot.instruments import split_instrument_name

NEW_YORK = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class SessionWindow:
    name: str
    start_utc: time
    end_utc: time


DEFAULT_SESSIONS = (
    SessionWindow("asian", time(0, 0), time(8, 0)),
    SessionWindow("london", time(7, 0), time(16, 0)),
    SessionWindow("new_york", time(12, 0), time(21, 0)),
    SessionWindow("overlap", time(12, 0), time(16, 0)),
)


def coerce_utc(value: datetime | None = None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        return current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc)


def active_sessions(now: datetime | None = None, windows: tuple[SessionWindow, ...] = DEFAULT_SESSIONS) -> set[str]:
    current = coerce_utc(now).time()
    return {window.name for window in windows if _time_in_window(current, window.start_utc, window.end_utc)}


def is_fx_market_open(now: datetime | None = None) -> bool:
    """Return False during the normal Friday 5pm NY to Sunday 5pm NY close."""

    ny_time = coerce_utc(now).astimezone(NEW_YORK)
    weekday = ny_time.weekday()
    clock = ny_time.time()
    if weekday == 5:
        return False
    if weekday == 4 and clock >= time(17, 0):
        return False
    if weekday == 6 and clock < time(17, 0):
        return False
    return True


def is_rollover_window(now: datetime | None = None, avoid_minutes: int = 15) -> bool:
    ny_time = coerce_utc(now).astimezone(NEW_YORK)
    rollover = ny_time.replace(hour=17, minute=0, second=0, microsecond=0)
    return abs(ny_time - rollover) <= timedelta(minutes=max(0, avoid_minutes))


def too_close_to_weekend(now: datetime | None = None, close_before_minutes: int = 60) -> bool:
    ny_time = coerce_utc(now).astimezone(NEW_YORK)
    if ny_time.weekday() != 4:
        return False
    cutoff = ny_time.replace(hour=17, minute=0, second=0, microsecond=0) - timedelta(
        minutes=max(0, close_before_minutes)
    )
    return ny_time >= cutoff


def news_blackout_reason(
    instrument: str,
    events: list[NewsEvent],
    now: datetime | None = None,
    before_minutes: int = 30,
    after_minutes: int = 30,
) -> str | None:
    base, quote = split_instrument_name(instrument)
    currencies = {base, quote}
    current = coerce_utc(now)
    before = timedelta(minutes=max(0, before_minutes))
    after = timedelta(minutes=max(0, after_minutes))
    for event in events:
        if event.impact.lower() != "high" or event.currency.upper() not in currencies:
            continue
        start = coerce_utc(event.starts_at) - before
        end = coerce_utc(event.ends_at) + after
        if start <= current <= end:
            return f"news_blackout:{event.currency.upper()}:{event.name}"
    return None


def trading_allowed_now(
    instrument: str,
    settings: StrategySettings,
    events: list[NewsEvent],
    now: datetime | None = None,
) -> tuple[bool, str]:
    if not is_fx_market_open(now):
        return False, "market_closed"
    if too_close_to_weekend(now, settings.close_before_weekend_minutes):
        return False, "weekend_cutoff"
    if is_rollover_window(now, settings.avoid_rollover_minutes):
        return False, "rollover_window"
    allowed_sessions = {session.lower() for session in settings.trade_sessions_utc}
    if allowed_sessions:
        current_sessions = active_sessions(now)
        if current_sessions.isdisjoint(allowed_sessions):
            return False, "session_filter"
    blackout = news_blackout_reason(
        instrument,
        events,
        now,
        settings.news_blackout_before_minutes,
        settings.news_blackout_after_minutes,
    )
    if blackout:
        return False, blackout
    return True, "ok"


def _time_in_window(value: time, start: time, end: time) -> bool:
    if start <= end:
        return start <= value < end
    return value >= start or value < end

