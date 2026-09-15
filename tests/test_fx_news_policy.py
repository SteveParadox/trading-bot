from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fxbot.api import _config_payload
from fxbot.config import FxBotSettings, NewsEvent, StrategySettings
from fxbot.market_hours import can_trade
from fxbot.news import NewsSnapshot, deduplicate_events


NOW = datetime(2026, 9, 15, 14, 0, tzinfo=timezone.utc)


def event(
    *,
    name: str = "US CPI",
    currency: str = "USD",
    score: int = 95,
    event_id: str | None = "cpi-1",
    start: datetime = NOW + timedelta(minutes=10),
) -> NewsEvent:
    return NewsEvent(
        name=name,
        currency=currency,
        impact="high" if score >= 71 else "medium",
        starts_at=start,
        ends_at=start + timedelta(minutes=15),
        event_id=event_id,
        impact_score=score,
    )


def test_can_trade_returns_explainable_high_impact_decision() -> None:
    decision = can_trade(
        "EUR_USD",
        NOW,
        news_state=NewsSnapshot(events=[event()], stale=False),
        settings=StrategySettings(trade_sessions_utc=()),
    )

    assert decision.allowed is False
    assert decision.matching_event is not None
    assert decision.event_id == "cpi-1"
    assert decision.impact == "HIGH"
    assert decision.applicable_restriction == "high_impact_news"
    assert decision.time_until_event_seconds == 600


def test_medium_news_is_optional_but_high_news_remains_blocked() -> None:
    medium = event(name="Retail Sales", score=50)
    permissive = can_trade(
        "EUR_USD",
        NOW,
        news_state=NewsSnapshot(events=[medium], stale=False),
        settings=StrategySettings(trade_sessions_utc=()),
    )
    restrictive = can_trade(
        "EUR_USD",
        NOW,
        news_state=NewsSnapshot(events=[medium], stale=False),
        settings=StrategySettings(trade_sessions_utc=(), news_medium_impact_enabled=True),
    )

    assert permissive.allowed is True
    assert restrictive.allowed is False
    assert restrictive.applicable_restriction == "medium_impact_news"


def test_currency_and_instrument_restrictions_and_event_override() -> None:
    medium = event(name="Retail Sales", score=50)
    settings = StrategySettings(
        trade_sessions_utc=(),
        news_restricted_currencies=("USD",),
        news_event_overrides={"retail sales": "allow"},
    )
    allowed = can_trade(
        "EUR_USD",
        NOW,
        news_state=NewsSnapshot(events=[medium], stale=False),
        settings=settings,
    )

    blocked = can_trade(
        "EUR_USD",
        NOW,
        news_state=NewsSnapshot(events=[medium], stale=False),
        settings=StrategySettings(trade_sessions_utc=(), news_restricted_currencies=("USD",)),
    )

    assert allowed.allowed is True
    assert blocked.allowed is False
    assert blocked.applicable_restriction == "currency_restriction"


def test_stale_data_fails_closed_but_authoritative_empty_calendar_is_known() -> None:
    settings = StrategySettings(trade_sessions_utc=(), require_news_data=True)
    stale = can_trade(
        "EUR_USD",
        NOW,
        news_state=NewsSnapshot(events=[], stale=True),
        settings=settings,
    )
    empty = can_trade(
        "EUR_USD",
        NOW,
        news_state=NewsSnapshot(events=[], stale=False),
        settings=settings,
    )

    assert stale.reason == "news_data_stale"
    assert empty.allowed is True


def test_manual_allow_cannot_bypass_emergency_kill_switch_or_stale_data() -> None:
    event_now = event(start=NOW - timedelta(minutes=5))
    allowed = can_trade(
        "EUR_USD",
        NOW,
        news_state=NewsSnapshot(events=[event_now], stale=False),
        settings=StrategySettings(trade_sessions_utc=(), news_manual_override="allow"),
    )
    killed = can_trade(
        "EUR_USD",
        NOW,
        news_state=NewsSnapshot(events=[event_now], stale=False),
        settings=StrategySettings(trade_sessions_utc=(), news_manual_override="allow", news_emergency_kill_switch=True),
    )

    assert allowed.allowed is True
    assert killed.reason == "news_emergency_kill_switch"


def test_provider_duplicates_are_deduplicated_by_id() -> None:
    first = event()
    second = event()
    deduped = deduplicate_events([first, second])

    assert len(deduped) == 1
    assert deduped[0].event_id == "cpi-1"


def test_config_endpoint_does_not_expose_news_provider_key() -> None:
    payload = _config_payload(
        FxBotSettings(strategy=StrategySettings(news_api_key="calendar-secret"))
    )

    assert payload["strategy"]["news_api_key_configured"] is True
    assert "calendar-secret" not in str(payload)
    assert "news_api_key" not in payload["strategy"]
