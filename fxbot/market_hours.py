"""FX session, weekend, rollover, and news blackout guards."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Any
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
    impact_score_min: int = 71,
) -> str | None:
    """Return a reason string when *instrument* is affected by a blackout window.

    An event triggers a blackout only when all three conditions hold:
      1. Its ``impact_score`` >= ``impact_score_min`` (default 71 = HIGH).
         If ``impact_score`` is 0 (unset), the provider's ``impact`` label is
         used as a fallback (``high`` → score 70, ``medium`` → 40, ``low`` → 10).
      2. The event's currency matches one of the instrument's two currencies.
      3. The current time falls inside [event.starts_at - before, event.ends_at + after].
    """
    base, quote = split_instrument_name(instrument)
    currencies = {base, quote}
    current = coerce_utc(now)
    before = timedelta(minutes=max(0, before_minutes))
    after = timedelta(minutes=max(0, after_minutes))
    for event in events:
        score = event.impact_score or _fallback_score(event.impact)
        if score < impact_score_min:
            continue
        if event.currency.upper() not in currencies:
            continue
        start = coerce_utc(event.starts_at) - before
        end = coerce_utc(event.ends_at) + after
        if start <= current <= end:
            return f"news_blackout:{event.currency.upper()}:{event.name}:score={score}"
    return None


def _fallback_score(impact_label: str) -> int:
    """Map a provider impact label to a numeric score when impact_score is 0.

    Kept conservative: a bare ``high`` label maps to 75 so it clears the default
    71 cut-off even when the event has never been run through the impact scorer.
    """
    label = impact_label.lower()
    if label == "high":
        return 75
    if label == "medium":
        return 40
    return 10


@dataclass(frozen=True)
class TradePermissionDecision:
    """Explainable, deterministic result of the entry permission gate."""

    allowed: bool
    reason: str
    matching_event: NewsEvent | None = None
    impact: str | None = None
    time_until_event_seconds: float | None = None
    time_since_event_seconds: float | None = None
    applicable_restriction: str | None = None
    news_stale: bool = False

    @property
    def event_id(self) -> str | None:
        return self.matching_event.event_id if self.matching_event else None

    def as_dict(self) -> dict[str, Any]:
        event = self.matching_event
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "event_id": self.event_id,
            "event_name": event.name if event else None,
            "currency": event.currency if event else None,
            "impact": self.impact,
            "time_until_event_seconds": self.time_until_event_seconds,
            "time_since_event_seconds": self.time_since_event_seconds,
            "applicable_restriction": self.applicable_restriction,
            "news_stale": self.news_stale,
        }


def _news_state_parts(
    news_state: Any,
    *,
    events: list[NewsEvent] | None,
    news_stale: bool,
) -> tuple[list[NewsEvent], bool, bool]:
    """Extract events, freshness, and whether the state is authoritative."""

    if news_state is None:
        return list(events or []), bool(news_stale), bool(events) and not news_stale
    if isinstance(news_state, list):
        return list(news_state), False, True
    state_events = list(getattr(news_state, "events", []) or [])
    state_stale = bool(getattr(news_state, "stale", False))
    return state_events, state_stale, True


def _event_override(event: NewsEvent, settings: StrategySettings) -> str | None:
    keys = [event.name.strip().lower()]
    if event.event_id:
        keys.insert(0, event.event_id.strip().lower())
    for key in keys:
        if key in settings.news_event_overrides:
            return settings.news_event_overrides[key]
    return None


def _decision_for_event(
    event: NewsEvent,
    now: datetime,
    *,
    before_minutes: int,
    after_minutes: int,
    impact_score_min: int,
    medium_enabled: bool,
    restricted: bool,
    restriction_name: str | None,
    override: str | None,
) -> TradePermissionDecision | None:
    status = event.status.strip().lower()
    if status in {"cancelled", "canceled", "invalid"} or override == "allow":
        return None
    score = event.impact_score or _fallback_score(event.impact)
    minimum = 31 if medium_enabled else impact_score_min
    if override != "block" and not restricted and score < minimum:
        return None
    if override != "block" and restricted and score < 31:
        return None
    start = coerce_utc(event.starts_at) - timedelta(minutes=max(0, before_minutes))
    end = coerce_utc(event.ends_at) + timedelta(minutes=max(0, after_minutes))
    if not start <= now <= end:
        return None
    until = max(0.0, (coerce_utc(event.starts_at) - now).total_seconds()) if now < coerce_utc(event.starts_at) else 0.0
    since = max(0.0, (now - coerce_utc(event.ends_at)).total_seconds()) if now > coerce_utc(event.ends_at) else 0.0
    if override == "block":
        restriction = "event_override:block"
    elif restricted and restriction_name:
        restriction = restriction_name
    elif restricted:
        restriction = "instrument_restriction"
    elif score >= 71:
        restriction = "high_impact_news"
    else:
        restriction = "medium_impact_news"
    reason_prefix = "news_blackout" if score >= 71 else "news_blackout_medium"
    return TradePermissionDecision(
        allowed=False,
        reason=f"{reason_prefix}:{event.currency.upper()}:{event.name}:score={score}",
        matching_event=event,
        impact="HIGH" if score >= 71 else "MEDIUM" if score >= 31 else "LOW",
        time_until_event_seconds=until,
        time_since_event_seconds=since,
        applicable_restriction=restriction,
    )


def can_trade(
    instrument: str,
    timestamp: datetime,
    strategy_signal: Any = None,
    account_state: Any = None,
    news_state: Any = None,
    *,
    settings: StrategySettings | None = None,
    events: list[NewsEvent] | None = None,
    news_stale: bool = False,
    manual_override: str | None = None,
    emergency_kill_switch: bool | None = None,
) -> TradePermissionDecision:
    """Return the single deterministic entry permission decision.

    This gate owns calendar, session, weekend, manual override, and emergency
    kill-switch policy. Strategy output remains an input, never an authority;
    hard risk sizing and broker checks still run after this function.
    """

    del strategy_signal  # Reserved for audit correlation; policy is deterministic.
    resolved = settings or StrategySettings()
    current = coerce_utc(timestamp)
    kill = resolved.news_emergency_kill_switch if emergency_kill_switch is None else emergency_kill_switch
    override = resolved.news_manual_override if manual_override is None else manual_override.lower()

    if kill:
        return TradePermissionDecision(False, "news_emergency_kill_switch", applicable_restriction="emergency_kill_switch")
    if override == "block":
        return TradePermissionDecision(False, "news_manual_block", applicable_restriction="manual_override")

    state_events, stale, authoritative = _news_state_parts(
        news_state,
        events=events,
        news_stale=news_stale,
    )
    if resolved.require_news_data and stale:
        return TradePermissionDecision(False, "news_data_stale", news_stale=True, applicable_restriction="stale_data")
    if resolved.require_news_data and not authoritative:
        return TradePermissionDecision(False, "news_data_unavailable", applicable_restriction="missing_data")

    # An explicit account halt is honored here as a second defensive boundary;
    # detailed position sizing and exposure calculations remain in FxRiskManager.
    if isinstance(account_state, dict):
        account_halted = bool(account_state.get("trading_halted") or account_state.get("kill_switch"))
    else:
        account_halted = bool(getattr(account_state, "trading_halted", False) or getattr(account_state, "kill_switch", False))
    if account_halted:
        return TradePermissionDecision(False, "account_trading_halted", applicable_restriction="account_state")

    if not is_fx_market_open(current):
        return TradePermissionDecision(False, "market_closed")
    if too_close_to_weekend(current, resolved.close_before_weekend_minutes):
        return TradePermissionDecision(False, "weekend_cutoff")
    if is_rollover_window(current, resolved.avoid_rollover_minutes):
        return TradePermissionDecision(False, "rollover_window")
    allowed_sessions = {session.lower() for session in resolved.trade_sessions_utc}
    if allowed_sessions and active_sessions(current).isdisjoint(allowed_sessions):
        return TradePermissionDecision(False, "session_filter")

    base, quote = split_instrument_name(instrument)
    currencies = {base, quote}
    restricted_instrument = instrument.upper() in {item.upper() for item in resolved.news_restricted_instruments}
    restricted_currencies = {item.upper() for item in resolved.news_restricted_currencies}
    matches: list[TradePermissionDecision] = []
    for event in state_events:
        currency = event.currency.upper()
        if currency not in currencies:
            continue
        event_override = _event_override(event, resolved)
        restricted = restricted_instrument or currency in restricted_currencies
        restriction_name = "instrument_restriction" if restricted_instrument else "currency_restriction" if currency in restricted_currencies else None
        decision = _decision_for_event(
            event,
            current,
            before_minutes=resolved.news_blackout_before_minutes,
            after_minutes=resolved.news_blackout_after_minutes,
            impact_score_min=resolved.news_blackout_impact_score_min,
            medium_enabled=resolved.news_medium_impact_enabled,
            restricted=restricted,
            restriction_name=restriction_name,
            override=event_override,
        )
        if decision is not None:
            matches.append(decision)
    if matches and override != "allow":
        return min(
            matches,
            key=lambda decision: abs(decision.time_until_event_seconds or 0.0) + abs(decision.time_since_event_seconds or 0.0),
        )
    return TradePermissionDecision(True, "ok", news_stale=stale)


def trading_allowed_now(
    instrument: str,
    settings: StrategySettings,
    events: list[NewsEvent],
    now: datetime | None = None,
    news_stale: bool = False,
) -> tuple[bool, str]:
    """Backward-compatible tuple wrapper around :func:`can_trade`."""

    decision = can_trade(
        instrument,
        coerce_utc(now),
        news_state=None,
        settings=settings,
        events=events,
        news_stale=news_stale,
    )
    return decision.allowed, decision.reason


def _time_in_window(value: time, start: time, end: time) -> bool:
    if start <= end:
        return start <= value < end
    return value >= start or value < end
