"""Offline regressions for the public weekly export and its safety boundary."""

from __future__ import annotations

import io
import json
import os
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from email.message import Message
from email.utils import format_datetime
from pathlib import Path
from urllib.error import HTTPError

import pytest
from dotenv import dotenv_values

from fxbot.config import BrokerSettings, FxBotSettings, RuntimeSettings, StrategySettings, settings_from_env
from fxbot.market_hours import can_trade
from fxbot.news import ForexFactoryProvider, NewsRateLimitError, build_news_gateway
from fxbot.news_check import main as check_news


NOW = datetime(2026, 9, 10, 12, 10, tzinfo=timezone.utc)


@pytest.fixture
def payload():
    # Same flat shape/offset-bearing dates as the public thisweek export.
    # Synthetic fixed dates keep tests independent of the network and clock.
    return [
        {"title": "CPI m/m", "country": "USD", "date": "2026-09-10T08:30:00-04:00", "impact": "High", "forecast": "0.3%", "previous": "0.2%"},
        {"title": "Employment Change", "country": "AUD", "date": "2026-09-11T08:30:00-04:00", "impact": "High", "forecast": "20.9K", "previous": "-15.8K"},
        {"title": "Bank Holiday", "country": "JPY", "date": "2026-09-09T19:00:00-04:00", "impact": "Holiday", "forecast": "", "previous": ""},
    ]


@pytest.fixture
def http(monkeypatch):
    calls = []
    state = {}

    def respond(request, timeout):
        calls.append(request)
        if "error" in state:
            raise state["error"]
        return io.BytesIO(json.dumps(state["payload"]).encode())

    monkeypatch.setattr("fxbot.news.urllib.request.urlopen", respond)
    return state, calls


def gateway(tmp_path):
    settings = StrategySettings(require_news_data=True, news_use_forex_factory=True, trade_sessions_utc=())
    return build_news_gateway(settings, database_url=f"sqlite:///{tmp_path / 'news.db'}")


def test_flat_feed_fetch_normalizes_and_does_not_invent_actuals(payload, http):
    state, calls = http
    state["payload"] = payload
    events = ForexFactoryProvider().fetch()
    assert len(events) == 3
    assert calls[0].full_url == "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    assert calls[0].get_header("X-api-key") is None
    assert events[0].starts_at == datetime(2026, 9, 10, 12, 30, tzinfo=timezone.utc)
    assert events[0].currency == "USD"
    assert events[0].impact == "high"
    assert events[0].forecast == 0.3
    assert events[0].previous == 0.2
    assert events[0].actual is None
    assert events[0].source == "forexfactory"
    assert events[1].forecast == 20900
    assert events[1].previous == -15800
    assert events[2].impact == "holiday"


def test_legacy_nested_feed_still_fetches(payload, http):
    item = {**payload[0], "country": {"code": "USD", "name": "United States"}, "date": "2026-09-10 08:30am"}
    http[0]["payload"] = {"days": [{"items": [item]}]}
    event = ForexFactoryProvider().fetch()[0]
    assert event.starts_at.hour == 12
    assert event.country == "United States"


@pytest.mark.parametrize(("date", "hour"), [
    ("2026-01-09T08:30:00-05:00", 13),
    ("2026-07-09T08:30:00-04:00", 12),
    ("2026-07-09T08:30:00Z", 8),
])
def test_explicit_offsets_are_authoritative(payload, http, date, hour):
    http[0]["payload"] = [{**payload[0], "date": date}]
    assert ForexFactoryProvider().fetch()[0].starts_at.hour == hour


@pytest.mark.parametrize("bad", [{}, {"error": "rate limited"}, [], {"days": []}, {"days": [{}]}, [None], "html"])
def test_unknown_or_empty_week_is_not_success(bad, http):
    http[0]["payload"] = bad
    with pytest.raises(ValueError):
        ForexFactoryProvider().fetch()


@pytest.mark.parametrize("change", [
    {"title": ""}, {"country": ""}, {"country": "United States"},
    {"date": "bad date"}, {"date": "2026-09-10"},
    {"impact": ""}, {"impact": "unexpected"},
])
def test_one_bad_record_invalidates_whole_week(payload, http, change):
    http[0]["payload"] = [payload[0], {**payload[1], **change}]
    with pytest.raises(ValueError, match="invalid event"):
        ForexFactoryProvider().fetch()


def test_throttle_dedup_and_pair_aware_blackouts(tmp_path, payload, http):
    state, calls = http
    state["payload"] = payload + [payload[0]]
    service = gateway(tmp_path)
    snapshot = service.ensure_current(NOW)
    assert snapshot.known_state == "FRESH"
    assert len(snapshot.events) == 3
    assert not can_trade("EUR_USD", NOW, news_state=snapshot, settings=service.settings).allowed
    assert can_trade("EUR_GBP", NOW, news_state=snapshot, settings=service.settings).allowed
    event_time = NOW + timedelta(minutes=20)
    for delta in (-30, 0, 30):
        assert not can_trade("EUR_USD", event_time + timedelta(minutes=delta), news_state=snapshot, settings=service.settings).allowed
    assert can_trade("EUR_USD", event_time + timedelta(minutes=31), news_state=snapshot, settings=service.settings).allowed
    service.ensure_current(NOW + timedelta(seconds=299))
    assert len(calls) == 1
    service.ensure_current(NOW + timedelta(seconds=300))
    assert len(calls) == 2
    assert service.cache.count() == 3


def test_failure_keeps_last_good_cache_but_does_not_extend_freshness(tmp_path, payload, http):
    state, calls = http
    state["payload"] = payload
    service = gateway(tmp_path)
    service.ensure_current(NOW)
    state["payload"] = {"error": "maintenance"}
    cached = service.ensure_current(NOW + timedelta(seconds=300))
    assert len(cached.events) == 3
    assert not cached.stale
    assert cached.last_error
    assert service.cache.last_fetched() == NOW
    service.ensure_current(NOW + timedelta(seconds=301))
    assert len(calls) == 2  # no rapid retry on every trading scan
    stale = service.ensure_current(NOW + timedelta(seconds=3601))
    assert stale.known_state == "UNKNOWN"
    decision = can_trade("EUR_GBP", NOW + timedelta(seconds=3601), news_state=stale, settings=service.settings)
    assert not decision.allowed
    assert decision.reason == "news_data_stale"


def test_initial_outage_cannot_fall_back_to_static_always_fresh_rows(tmp_path, payload, http):
    state, _ = http
    state["payload"] = payload
    static = ForexFactoryProvider().fetch()
    state["error"] = OSError("offline")
    service = gateway(tmp_path)
    service.static_events = static
    snapshot = service.ensure_current(NOW)
    assert snapshot.stale
    assert snapshot.source == "forexfactory"
    assert not can_trade("EUR_GBP", NOW, news_state=snapshot, settings=service.settings).allowed


def test_previously_cached_empty_feed_is_unknown(tmp_path, http):
    service = gateway(tmp_path)
    service.cache.upsert_many([], fetched_at=NOW)
    http[0]["error"] = OSError("offline")
    assert service.ensure_current(NOW).stale


def test_previous_week_cannot_be_refreshed_as_current(tmp_path, payload, http):
    http[0]["payload"] = payload
    service = gateway(tmp_path)
    snapshot = service.ensure_current(NOW + timedelta(days=7))
    assert snapshot.stale
    assert "requested week" in snapshot.last_error
    assert service.cache.last_fetched() is None


def test_week_rollover_expires_cache_even_inside_ttl(tmp_path, payload, http):
    http[0]["payload"] = payload
    service = gateway(tmp_path)
    # Sunday begins at 04:00 UTC in New York in September.
    saturday = datetime(2026, 9, 13, 3, 59, tzinfo=timezone.utc)
    assert not service.ensure_current(saturday).stale
    assert service.ensure_current(saturday + timedelta(minutes=2)).stale
    assert len(http[1]) == 1


def test_rate_limit_respects_retry_after_even_on_forced_sync(tmp_path, http):
    headers = Message()
    headers["Retry-After"] = "1800"
    http[0]["error"] = HTTPError("https://calendar.invalid", 429, "Too many requests", headers, None)
    service = gateway(tmp_path)
    assert service.ensure_current(NOW).stale
    service.ensure_current(NOW + timedelta(seconds=300))
    service.sync_upcoming(NOW + timedelta(seconds=1799))
    assert len(http[1]) == 1
    service.ensure_current(NOW + timedelta(seconds=1800))
    assert len(http[1]) == 2


@pytest.mark.parametrize("header", ["bad", "0", "-30", "inf", "nan"])
def test_invalid_or_short_retry_after_has_safe_minimum(header, http):
    headers = Message()
    headers["Retry-After"] = header
    http[0]["error"] = HTTPError("https://calendar.invalid", 429, "Too many requests", headers, None)
    with pytest.raises(NewsRateLimitError) as caught:
        ForexFactoryProvider().fetch()
    assert caught.value.retry_after_seconds == 300
    assert len(http[1]) == 1


def test_http_date_retry_after_and_large_values(http, monkeypatch):
    monkeypatch.setattr("fxbot.news.datetime", type("Clock", (), {"now": staticmethod(lambda tz: NOW)}))
    headers = Message()
    headers["Retry-After"] = format_datetime(NOW + timedelta(minutes=20), usegmt=True)
    http[0]["error"] = HTTPError("https://calendar.invalid", 429, "Too many requests", headers, None)
    with pytest.raises(NewsRateLimitError) as caught:
        ForexFactoryProvider().fetch()
    assert caught.value.retry_after_seconds == 1200
    assert NewsRateLimitError(1e300).retry_after_seconds == 7 * 86400


def test_demo_profile_loads_and_selects_free_feed(tmp_path, monkeypatch):
    for key in tuple(os.environ):
        if key.startswith(("FX_", "MT5_", "AI_DELIBERATION_")):
            monkeypatch.delenv(key)
    profile = Path(__file__).parents[1] / "configs/fx_forexfactory_demo.env.example"
    for key, value in dotenv_values(profile).items():
        monkeypatch.setenv(key, value)
    settings = settings_from_env()
    service = build_news_gateway(settings.strategy, database_url=f"sqlite:///{tmp_path / 'profile.db'}")
    assert isinstance(service.provider, ForexFactoryProvider)
    assert service.refresh_throttle_seconds == 300
    assert service.max_age_seconds == 3600
    assert settings.broker.demo_only
    assert not settings.runtime.live_trading_enabled
    assert settings.strategy.require_news_data


def test_free_feed_rejects_live_or_fail_open_settings():
    strategy = StrategySettings(news_use_forex_factory=True, require_news_data=True)
    with pytest.raises(ValueError, match="demo/forward-test only"):
        FxBotSettings(strategy=strategy, broker=BrokerSettings(demo_only=False))
    with pytest.raises(ValueError, match="demo/forward-test only"):
        FxBotSettings(strategy=strategy, runtime=RuntimeSettings(live_trading_enabled=True))
    with pytest.raises(ValueError, match="FX_REQUIRE_NEWS_DATA=true"):
        FxBotSettings(strategy=replace(strategy, require_news_data=False))
    # A separately configured HTTP provider keeps its existing precedence.
    FxBotSettings(strategy=replace(strategy, news_api_endpoint="https://calendar.invalid"), broker=BrokerSettings(demo_only=False))


def test_check_command_does_not_open_runtime_database(tmp_path, payload, http, monkeypatch, capsys):
    runtime_db = tmp_path / "must-not-be-created.db"
    settings = FxBotSettings(
        strategy=StrategySettings(news_use_forex_factory=True, require_news_data=True),
        runtime=RuntimeSettings(database_url=f"sqlite:///{runtime_db}"),
    )
    monkeypatch.setattr("fxbot.news_check.settings_from_env", lambda: settings)
    monkeypatch.setattr("fxbot.news_check.datetime", type("Clock", (), {"now": staticmethod(lambda tz: NOW)}))
    http[0]["payload"] = payload
    assert check_news([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["event_count"] == 3
    assert report["state"] == "FRESH"
    assert not runtime_db.exists()
    http[0]["error"] = OSError("offline")
    assert check_news([]) == 1
    assert json.loads(capsys.readouterr().out)["state"] == "UNKNOWN"
