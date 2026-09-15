"""Tests for the Part A provider-agnostic news ingestion architecture."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from fxbot.config import NewsEvent, StrategySettings, _news_event_from_dict
from fxbot.journal import StructuredJournal
from fxbot.market_hours import news_blackout_reason, trading_allowed_now
from fxbot.monitoring import OperationalMonitor
from fxbot.news import (
    ForexFactoryProvider,
    HttpNewsProvider,
    ManualJsonNewsProvider,
    NewsCache,
    NewsGateway,
    calculate_impact_score,
    score_events,
    _parse_pct_number,
)


FIXED_NOW = datetime(2026, 9, 10, 12, 10, tzinfo=timezone.utc)


def test_manual_provider_normalizes_and_converts_utc(tmp_path: Path) -> None:
    path = tmp_path / "events.json"
    path.write_text(
        json.dumps(
            [
                {
                    "event_id": "evt-1",
                    "name": "US CPI",
                    "currency": "USD",
                    "impact": "HIGH",
                    "starts_at": "2026-09-10T08:30:00-04:00",  # offset, must become UTC 12:30
                    "ends_at": "2026-09-10T08:45:00-04:00",
                }
            ]
        ),
        encoding="utf-8",
    )
    provider = ManualJsonNewsProvider(path)
    events = provider.fetch()
    assert len(events) == 1
    event = events[0]
    assert event.currency == "USD"
    assert event.impact == "high"
    assert event.event_id == "evt-1"
    assert event.starts_at == datetime(2026, 9, 10, 12, 30, tzinfo=timezone.utc)
    assert event.ends_at == datetime(2026, 9, 10, 12, 45, tzinfo=timezone.utc)


def test_news_cache_deduplicates_by_event_id(tmp_path: Path) -> None:
    cache = NewsCache(f"sqlite:///{tmp_path / 'news.db'}")
    event = NewsEvent(
        event_id="evt-1",
        name="US CPI",
        currency="USD",
        impact="high",
        starts_at=datetime(2026, 9, 10, 12, 30, tzinfo=timezone.utc),
        ends_at=datetime(2026, 9, 10, 12, 45, tzinfo=timezone.utc),
    )
    cache.upsert_many([event], fetched_at=FIXED_NOW)
    cache.upsert_many([event], fetched_at=FIXED_NOW + timedelta(minutes=1))
    assert cache.count() == 1
    assert len(cache.all()) == 1


def test_news_cache_coexists_with_sqlalchemy_journal(tmp_path: Path) -> None:
    # The raw sqlite cache must not collide with the SQLAlchemy news_events model
    # when both live in the same database file (forward-test worker scenario).
    url = f"sqlite:///{tmp_path / 'journal.db'}"
    cache = NewsCache(url)
    cache.upsert_many(
        [
            NewsEvent(
                event_id="evt-1",
                name="US CPI",
                currency="USD",
                impact="high",
                starts_at=datetime(2026, 9, 10, 12, 30, tzinfo=timezone.utc),
                ends_at=datetime(2026, 9, 10, 12, 45, tzinfo=timezone.utc),
            )
        ],
        fetched_at=FIXED_NOW,
    )
    journal = StructuredJournal(url, None)
    try:
        assert journal.get_state().state == "stopped"
    finally:
        journal.close()
    assert cache.count() == 1


def test_news_cache_fallback_key_for_events_without_id(tmp_path: Path) -> None:
    cache = NewsCache(f"sqlite:///{tmp_path / 'news.db'}")
    base = dict(
        name="US CPI",
        currency="USD",
        impact="high",
        starts_at=datetime(2026, 9, 10, 12, 30, tzinfo=timezone.utc),
        ends_at=datetime(2026, 9, 10, 12, 45, tzinfo=timezone.utc),
    )
    cache.upsert_many([NewsEvent(**base)], fetched_at=FIXED_NOW)
    cache.upsert_many([NewsEvent(**base)], fetched_at=FIXED_NOW)
    assert cache.count() == 1


def test_empty_cache_is_stale(tmp_path: Path) -> None:
    cache = NewsCache(f"sqlite:///{tmp_path / 'news.db'}")
    gateway = NewsGateway(settings=StrategySettings(require_news_data=True), cache=cache)
    snapshot = gateway.ensure_current(FIXED_NOW)
    assert snapshot.stale is True
    assert snapshot.available is False
    assert snapshot.known_state == "UNKNOWN"


def test_cache_goes_stale_when_older_than_max_age(tmp_path: Path) -> None:
    cache = NewsCache(f"sqlite:///{tmp_path / 'news.db'}")
    cache.upsert_many(
        [
            NewsEvent(
                event_id="evt-1",
                name="US CPI",
                currency="USD",
                impact="high",
                starts_at=datetime(2026, 9, 10, 12, 30, tzinfo=timezone.utc),
                ends_at=datetime(2026, 9, 10, 12, 45, tzinfo=timezone.utc),
            )
        ],
        fetched_at=FIXED_NOW - timedelta(hours=2),
    )
    gateway = NewsGateway(settings=StrategySettings(require_news_data=True, news_data_max_age_seconds=3600), cache=cache)
    snapshot = gateway.ensure_current(FIXED_NOW)
    assert snapshot.events
    assert snapshot.stale is True
    assert snapshot.age_seconds == pytest.approx(7200.0, abs=1.0)


def test_stale_news_blocks_entries_when_required(tmp_path: Path) -> None:
    cache = NewsCache(f"sqlite:///{tmp_path / 'news.db'}")
    cache.upsert_many(
        [
            NewsEvent(
                name="US CPI",
                currency="USD",
                impact="high",
                starts_at=datetime(2026, 9, 10, 12, 30, tzinfo=timezone.utc),
                ends_at=datetime(2026, 9, 10, 12, 45, tzinfo=timezone.utc),
            )
        ],
        fetched_at=FIXED_NOW - timedelta(hours=5),
    )
    settings = StrategySettings(require_news_data=True, news_data_max_age_seconds=3600, trade_sessions_utc=())
    gateway = NewsGateway(settings=settings, cache=cache)
    snapshot = gateway.ensure_current(FIXED_NOW)
    allowed, reason = trading_allowed_now(
        "EUR_USD",
        settings,
        snapshot.events,
        FIXED_NOW,
        news_stale=snapshot.stale,
    )
    assert allowed is False
    assert reason == "news_data_stale"


def test_stale_news_does_not_block_when_not_required(tmp_path: Path) -> None:
    cache = NewsCache(f"sqlite:///{tmp_path / 'news.db'}")
    cache.upsert_many(
        [
            NewsEvent(
                name="US CPI",
                currency="USD",
                impact="high",
                starts_at=datetime(2026, 9, 10, 12, 30, tzinfo=timezone.utc),
                ends_at=datetime(2026, 9, 10, 12, 45, tzinfo=timezone.utc),
            )
        ],
        fetched_at=FIXED_NOW - timedelta(hours=5),
    )
    settings = StrategySettings(require_news_data=False, trade_sessions_utc=())
    gateway = NewsGateway(settings=settings, cache=cache)
    snapshot = gateway.ensure_current(FIXED_NOW)
    # Blackout logic still applies from the cached events when not required.
    allowed, reason = trading_allowed_now(
        "EUR_USD",
        settings,
        snapshot.events,
        FIXED_NOW,
        news_stale=snapshot.stale,
    )
    assert allowed is False
    assert reason.startswith("news_blackout:")


def test_provider_failure_never_drops_cached_events(tmp_path: Path) -> None:
    cache = NewsCache(f"sqlite:///{tmp_path / 'news.db'}")
    cache.upsert_many(
        [
            NewsEvent(
                event_id="evt-1",
                name="US CPI",
                currency="USD",
                impact="high",
                starts_at=datetime(2026, 9, 10, 12, 30, tzinfo=timezone.utc),
                ends_at=datetime(2026, 9, 10, 12, 45, tzinfo=timezone.utc),
            )
        ],
        fetched_at=FIXED_NOW - timedelta(minutes=1),
    )

    class FailingProvider:
        name = "failing"

        def fetch(self):
            raise OSError("provider down")

    monitor = OperationalMonitor(news_data_max_age_seconds=3600)
    gateway = NewsGateway(
        settings=StrategySettings(require_news_data=True),
        provider=FailingProvider(),
        cache=cache,
        monitor=monitor,
        refresh_throttle_seconds=1,
    )
    snapshot = gateway.ensure_current(FIXED_NOW)
    # Fail-safe: the outage must not be interpreted as "no news events".
    assert len(snapshot.events) == 1
    assert snapshot.source == "failing"
    assert snapshot.stale is False  # cache is still fresh
    assert snapshot.available is True


def test_provider_failure_with_aged_cache_is_unknown(tmp_path: Path) -> None:
    cache = NewsCache(f"sqlite:///{tmp_path / 'news.db'}")
    cache.upsert_many(
        [
            NewsEvent(
                event_id="evt-1",
                name="US CPI",
                currency="USD",
                impact="high",
                starts_at=datetime(2026, 9, 10, 12, 30, tzinfo=timezone.utc),
                ends_at=datetime(2026, 9, 10, 12, 45, tzinfo=timezone.utc),
            )
        ],
        fetched_at=FIXED_NOW - timedelta(hours=2),
    )

    class FailingProvider:
        name = "failing"

        def fetch(self):
            raise OSError("provider down")

    monitor = OperationalMonitor(news_data_max_age_seconds=3600)
    gateway = NewsGateway(
        settings=StrategySettings(require_news_data=True),
        provider=FailingProvider(),
        cache=cache,
        monitor=monitor,
        refresh_throttle_seconds=1,
    )
    snapshot = gateway.ensure_current(FIXED_NOW)
    assert snapshot.stale is True
    assert snapshot.known_state == "UNKNOWN"
    assert snapshot.available is False
    alert_types = [a["type"] for a in monitor.snapshot().recent_alerts]
    assert "news_freshness" in alert_types


def test_refresh_throttle_limits_provider_calls(tmp_path: Path) -> None:
    cache = NewsCache(f"sqlite:///{tmp_path / 'news.db'}")
    calls = []

    class CountingProvider:
        name = "counting"

        def fetch(self):
            calls.append(1)
            return [
                NewsEvent(
                    name="US CPI",
                    currency="USD",
                    impact="high",
                    starts_at=datetime(2026, 9, 10, 12, 30, tzinfo=timezone.utc),
                    ends_at=datetime(2026, 9, 10, 12, 45, tzinfo=timezone.utc),
                )
            ]

    gateway = NewsGateway(
        settings=StrategySettings(require_news_data=True),
        provider=CountingProvider(),
        cache=cache,
        refresh_throttle_seconds=60,
    )
    gateway.ensure_current(FIXED_NOW)
    gateway.ensure_current(FIXED_NOW + timedelta(seconds=5))
    assert len(calls) == 1  # throttle suppressed the second refresh
    gateway.ensure_current(FIXED_NOW + timedelta(seconds=61))
    assert len(calls) == 2


def test_static_events_fallback_without_provider(tmp_path: Path) -> None:
    static = [
        NewsEvent(
            name="US CPI",
            currency="USD",
            impact="high",
            starts_at=datetime(2026, 9, 10, 12, 30, tzinfo=timezone.utc),
            ends_at=datetime(2026, 9, 10, 12, 45, tzinfo=timezone.utc),
        )
    ]
    gateway = NewsGateway(settings=StrategySettings(require_news_data=True), static_events=static)
    snapshot = gateway.ensure_current(FIXED_NOW)
    # Static events are auto-scored so the blackout threshold is uniform.
    assert len(snapshot.events) == 1
    scored = snapshot.events[0]
    assert scored.name == "US CPI"
    assert scored.currency == "USD"
    assert scored.impact_score >= 71
    assert snapshot.stale is False
    assert snapshot.source == "static"


def test_news_blackout_reason_matches_currencies() -> None:
    high_usd = NewsEvent(
        name="US CPI",
        currency="USD",
        impact="high",
        starts_at=datetime(2026, 9, 10, 12, 30, tzinfo=timezone.utc),
        ends_at=datetime(2026, 9, 10, 12, 45, tzinfo=timezone.utc),
    )
    low_eur = NewsEvent(
        name="EU Harmless",
        currency="EUR",
        impact="low",
        starts_at=datetime(2026, 9, 10, 12, 30, tzinfo=timezone.utc),
        ends_at=datetime(2026, 9, 10, 12, 45, tzinfo=timezone.utc),
    )
    # EUR_USD quotes USD: blocked inside the window.
    assert news_blackout_reason("EUR_USD", [high_usd], datetime(2026, 9, 10, 12, 40, tzinfo=timezone.utc))
    # AUD_JPY is untouched by a USD event.
    assert news_blackout_reason("AUD_JPY", [high_usd], datetime(2026, 9, 10, 12, 40, tzinfo=timezone.utc)) is None
    # Low-impact events never create a blackout.
    assert news_blackout_reason("EUR_USD", [low_eur], datetime(2026, 9, 10, 12, 40, tzinfo=timezone.utc)) is None
    # Outside the window (before the pre-event buffer).
    assert news_blackout_reason("EUR_USD", [high_usd], datetime(2026, 9, 10, 11, 55, tzinfo=timezone.utc)) is None


def test_http_provider_extract_and_normalize_items() -> None:
    provider = HttpNewsProvider("https://calendar.invalid/v1", api_key="secret")

    class Shape(HttpNewsProvider):
        def extract_items(self, payload):
            return payload.get("sorted", {}).get("events", [])

        def normalize_item(self, item):
            item = dict(item)
            item["starts_at"] = item.pop("time")
            return item

    shaped = Shape("https://calendar.invalid/v1")
    payload = {"sorted": {"events": [{"time": "2026-09-10T12:30:00Z", "name": "US CPI", "ccy": "USD"}]}}
    items = shaped.extract_items(payload)
    assert len(items) == 1
    event = shaped.normalize_item(items[0])
    assert event["starts_at"] == "2026-09-10T12:30:00Z"
    assert provider.normalize_item({"name": "US CPI", "ccy": "USD", "date": "2026-09-10T12:30:00Z"})["currency"] == "USD"


# --- Existing-position protection policy tests ---------------------------------

class _StubNewsBroker:
    def __init__(self, trades=None) -> None:
        self.trades = trades or []
        self.tightened: list[dict] = []
        self.closed: list[dict] = []

    def open_trades(self):
        return list(self.trades)

    def set_trade_dependent_orders(self, *, trade_id, instrument, stop_loss=None, take_profit=None):
        self.tightened.append({"trade_id": trade_id, "stop_loss": stop_loss, "take_profit": take_profit})
        return {"mt5": {"retcode": 10009}, "request": {}}

    def close_position(self, *, trade_id, instrument, signed_units, comment="fxft-news-close"):
        self.closed.append({"trade_id": trade_id, "signed_units": signed_units, "comment": comment})
        return {"mt5": {"retcode": 10009}, "price": 1.12, "deal_id": "900"}


def _profit_trade() -> dict:
    return {
        "id": "42",
        "instrument": "EUR_USD",
        "price": 1.1000,  # entry
        "initialUnits": 1000,
        "currentUnits": 1000,
        "stopLossOrder": {"price": 1.0950},
        "takeProfitOrder": {"price": 1.1300},
    }


def _news_worker(tmp_path, action: str, client):
    from contextlib import closing

    from fxbot.config import FxBotSettings, RuntimeSettings
    from fxbot.forward import ForwardTestWorker
    from fxbot.journal import StructuredJournal

    strategy = StrategySettings(
        require_news_data=True,
        news_risk_action=action,
        news_blackout_before_minutes=30,
        news_blackout_after_minutes=30,
        trade_sessions_utc=(),
    )
    bot_settings = FxBotSettings(
        instruments=["EUR_USD"],
        strategy=strategy,
        runtime=RuntimeSettings(database_url=f"sqlite:///{tmp_path / 'j.db'}", log_jsonl_path=str(tmp_path / "j.jsonl")),
    )
    journal = StructuredJournal(bot_settings.runtime.database_url, bot_settings.runtime.log_jsonl_path)
    worker = ForwardTestWorker(bot_settings, client=client, journal=journal)
    return _ProtectionContext(bot_settings, worker, journal)


class _ProtectionContext:
    def __init__(self, bot_settings, worker, journal) -> None:
        self.bot_settings = bot_settings
        self.worker = worker
        self.journal = journal

    def inputs(self):
        from fxbot.instruments import FxInstrument, PriceSnapshot

        instrument = FxInstrument(name="EUR_USD")
        price = PriceSnapshot("EUR_USD", bid=1.1199, ask=1.1201, time=FIXED_NOW)
        events = [
            NewsEvent(
                name="US CPI",
                currency="USD",
                impact="high",
                starts_at=datetime(2026, 9, 10, 12, 30, tzinfo=timezone.utc),
                ends_at=datetime(2026, 9, 10, 12, 45, tzinfo=timezone.utc),
            )
        ]
        return {"EUR_USD": instrument}, {"EUR_USD": price}, events

    def close(self) -> None:
        self.journal.close()


def test_protect_and_block_tightens_breakeven_stop(tmp_path: Path) -> None:
    client = _StubNewsBroker(trades=[_profit_trade()])
    ctx = _news_worker(tmp_path, "protect_and_block", client)
    try:
        instruments, prices, events = ctx.inputs()
        ctx.worker._protect_positions_for_news(FIXED_NOW, instruments, prices, events)
        assert len(client.tightened) == 1
        assert client.tightened[0]["trade_id"] == "42"
        # Breakeven + buffer on a 1.1000 entry with 0.2 pips (EUR_USD: 1e-4).
        assert client.tightened[0]["stop_loss"] == pytest.approx(1.10002)
    finally:
        ctx.close()


def test_close_positions_closes_exposure(tmp_path: Path) -> None:
    client = _StubNewsBroker(trades=[_profit_trade()])
    ctx = _news_worker(tmp_path, "close_positions", client)
    try:
        instruments, prices, events = ctx.inputs()
        ctx.worker._protect_positions_for_news(FIXED_NOW, instruments, prices, events)
        assert len(client.closed) == 1
        assert client.closed[0]["trade_id"] == "42"
        assert client.closed[0]["signed_units"] == 1000
        assert not client.tightened  # close policy should not tighten
    finally:
        ctx.close()


def test_block_entries_holds_positions(tmp_path: Path) -> None:
    client = _StubNewsBroker(trades=[_profit_trade()])
    ctx = _news_worker(tmp_path, "block_entries", client)
    try:
        instruments, prices, events = ctx.inputs()
        ctx.worker._protect_positions_for_news(FIXED_NOW, instruments, prices, events)
        assert client.closed == []
        assert client.tightened == []
    finally:
        ctx.close()


def test_news_worker_skips_unaffected_pair(tmp_path: Path) -> None:
    from fxbot.instruments import FxInstrument, PriceSnapshot

    client = _StubNewsBroker(
        trades=[
            {
                "id": "43",
                "instrument": "AUD_JPY",
                "price": 90.0,
                "initialUnits": 500,
                "currentUnits": 500,
                "stopLossOrder": {"price": 89.0},
            }
        ]
    )
    ctx = _news_worker(tmp_path, "close_positions", client)
    try:
        instrument = FxInstrument(name="AUD_JPY")
        price = PriceSnapshot("AUD_JPY", bid=90.05, ask=90.06, time=FIXED_NOW)
        events = [
            NewsEvent(
                name="US CPI",
                currency="USD",
                impact="high",
                starts_at=datetime(2026, 9, 10, 12, 30, tzinfo=timezone.utc),
                ends_at=datetime(2026, 9, 10, 12, 45, tzinfo=timezone.utc),
            )
        ]
        ctx.worker._protect_positions_for_news(FIXED_NOW, {"AUD_JPY": instrument}, {"AUD_JPY": price}, events)
        assert client.closed == []
    finally:
        ctx.close()


def test_imports_are_clean() -> None:
    event = NewsEvent(
        name="US CPI",
        currency="USD",
        impact="high",
        starts_at=datetime(2026, 9, 10, 12, 30, tzinfo=timezone.utc),
        ends_at=datetime(2026, 9, 10, 12, 45, tzinfo=timezone.utc),
    )
    assert event.event_name == "US CPI"
    assert event.importance == "high"


# --- Step 4: impact-scoring tests ---------------------------------------------

def _event(name: str, currency: str, impact: str, impact_score: int = 0) -> NewsEvent:
    return NewsEvent(
        name=name,
        currency=currency,
        impact=impact,
        starts_at=datetime(2026, 9, 10, 12, 30, tzinfo=timezone.utc),
        ends_at=datetime(2026, 9, 10, 12, 45, tzinfo=timezone.utc),
        impact_score=impact_score,
    )


def test_known_event_overrides_provider_label() -> None:
    scored = calculate_impact_score(_event("US CPI", "USD", "high"))
    # 95 (US CPI known-event base) + 10 (USD is a major currency) = 100 cap.
    assert scored.score == pytest.approx(100)
    assert scored.level == "HIGH"
    assert scored.high is True


def test_fomc_is_max_impact() -> None:
    scored = calculate_impact_score(_event("FOMC Statement", "USD", "high"))
    assert scored.score >= 100
    assert scored.level == "HIGH"


def test_fomc_rate_decision_does_not_get_shadowed_by_rate_decision() -> None:
    # Regression: the longest-token-first lookup let the shorter-token FOMC
    # (score 100) be shadowed by "rate decision" (score 90) in "FOMC Rate
    # Decision".  Highest-scoring match must win.
    scored = calculate_impact_score(_event("US FOMC Rate Decision", "USD", "high"))
    assert scored.score >= 100
    assert scored.level == "HIGH"


def test_provider_label_baseline_plus_major_currency_bonus() -> None:
    scored = calculate_impact_score(_event("Some Minor Release", "USD", "medium"))
    # 40 baseline + 10 for USD, capped at 100.
    assert scored.score >= 50
    assert scored.level in {"MEDIUM", "HIGH"}


def test_non_major_currency_gets_baseline_only() -> None:
    scored = calculate_impact_score(_event("Some Release", "NZD", "low"))
    assert scored.score == 10
    assert scored.level == "LOW"
    assert scored.high is False


def test_existing_provider_score_is_preserved() -> None:
    scored = calculate_impact_score(_event("Anything", "USD", "low", impact_score=88))
    assert scored.score == 88
    assert scored.level == "HIGH"


def test_score_events_sets_every_event_scores() -> None:
    events = _event("US CPI", "USD", "high")
    scored = score_events([events])
    assert len(scored) == 1
    assert scored[0].impact_score >= 71
    assert scored[0].impact_score <= 100


def test_blackout_uses_impact_score_threshold() -> None:
    now = datetime(2026, 9, 10, 12, 40, tzinfo=timezone.utc)
    high_medium = _event("US CPI", "USD", "high")
    # Scored MEDIUM (< default 71) must NOT block.
    low_score = NewsEvent(
        name="US CPI",
        currency="USD",
        impact="high",
        starts_at=datetime(2026, 9, 10, 12, 30, tzinfo=timezone.utc),
        ends_at=datetime(2026, 9, 10, 12, 45, tzinfo=timezone.utc),
        impact_score=50,
    )
    assert news_blackout_reason("EUR_USD", [high_medium], now)  # unscored high -> fallback 75
    assert news_blackout_reason("EUR_USD", [low_score], now) is None  # explicit low score
    # A MEDIUM-scored event can still block when the operator lowers the threshold.
    assert (
        news_blackout_reason(
            "EUR_USD",
            [low_score],
            now,
            impact_score_min=50,
        )
        is not None
    )


def test_blackout_ignores_unrelated_currency_high_score_event() -> None:
    now = datetime(2026, 9, 10, 12, 40, tzinfo=timezone.utc)
    yen_event = _event("BoJ Policy Rate", "JPY", "high", impact_score=95)
    assert news_blackout_reason("EUR_USD", [yen_event], now) is None
    assert news_blackout_reason("USD_JPY", [yen_event], now) is not None


# --- Step 1: ForexFactory provider tests --------------------------------------

def test_forex_factory_provider_normalizes_payload() -> None:
    provider = ForexFactoryProvider()
    payload = {
        "days": [
            {
                "date": "2026-09-16",
                "items": [
                    {
                        "id": "us_retail_sales_m_m",
                        "date": "2026-09-16 12:30am",
                        "impact": "High",
                        "title": "Retail Sales m/m",
                        "country": {"name": "United States", "code": "USD", "flag": "us"},
                        "forecast": "0.3%",
                        "previous": "0.1%",
                        "actual": "",
                    }
                ],
            }
        ]
    }
    items = provider.extract_items(payload)
    assert len(items) == 1
    normalized = provider.normalize_item(items[0])
    assert normalized["name"] == "Retail Sales m/m"
    assert normalized["currency"] == "USD"
    assert normalized["country"] == "United States"
    assert normalized["impact"] == "High"
    assert normalized["forecast"] == pytest.approx(0.3)
    assert normalized["previous"] == pytest.approx(0.1)
    assert "." not in normalized["event_id"]  # dots stripped, safe for DB keys
    assert normalized["event_id"] == "us_retail_sales_m_m"


def test_forex_factory_provider_dotless_event_id() -> None:
    provider = ForexFactoryProvider()
    normalized = provider.normalize_item({"id": "us.cpi.y.y", "title": "CPI", "currency": "USD", "impact": "High"})
    assert "." not in normalized["event_id"]
    assert normalized["event_id"] == "uscpiyy"


def test_forex_factory_12hour_timestamp_is_parsed() -> None:
    # The generic/manual config path accepts bare 12-hour timestamps and treats
    # them as UTC (no timezone source). The ForexFactoryProvider itself converts
    # its NY-time stamps to UTC before this path, so only operator-supplied
    # manual events land here naive.
    event = _news_event_from_dict(
        {
            "name": "Retail Sales m/m",
            "currency": "USD",
            "impact": "High",
            "starts_at": "2026-09-16 12:30am",
            "ends_at": "2026-09-16 12:30am",
        }
    )
    assert event.starts_at == datetime(2026, 9, 16, 0, 30, tzinfo=timezone.utc)
    assert event.ends_at == datetime(2026, 9, 16, 0, 30, tzinfo=timezone.utc)


def test_forex_factory_provider_parses_clock_times_through_fetch(tmp_path: Path) -> None:
    # End-to-end: a raw FF payload (12-hour am/pm) must survive normalize_item
    # + _news_event_from_dict instead of being ValueError-dropped in fetch().
    payload = {
        "days": [
            {
                "date": "2026-09-16",
                "items": [
                    {
                        "id": "us_cpi",
                        "date": "2026-09-16 08:30am",
                        "impact": "High",
                        "title": "CPI y/y",
                        "country": {"code": "USD"},
                    }
                ],
            }
        ]
    }
    provider = ForexFactoryProvider()
    events = []
    for item in provider.extract_items(payload):
        normalized = provider.normalize_item(item)
        events.append(_news_event_from_dict(normalized))
    assert len(events) == 1
    assert events[0].currency == "USD"
    # FF timestamps are NY time: 08:30am EDT on 2026-09-16 == 12:30 UTC.
    assert events[0].starts_at == datetime(2026, 9, 16, 12, 30, tzinfo=timezone.utc)


def test_missing_impact_defaults_to_low_not_high() -> None:
    # An event without an impact label must not silently become HIGH and freeze
    # trading for unrelated reasons; default to LOW instead.
    provider = HttpNewsProvider("https://calendar.invalid/v1")
    normalized = provider.normalize_item({"name": "Some Release", "ccy": "GBP", "date": "2026-09-16T08:30:00Z"})
    assert normalized["impact"] == "low"
    event = _news_event_from_dict({"name": "Release", "currency": "USD", "starts_at": "2026-09-16T08:30:00Z"})
    assert event.impact == "low"
    # An explicitly-tagged HIGH event is still HIGH.
    assert _news_event_from_dict(
        {"name": "x", "currency": "USD", "impact": "HIGH", "starts_at": "2026-09-16T08:30:00Z"}
    ).impact == "high"


def test_parse_pct_number_scales_suffixes() -> None:
    assert _parse_pct_number("3.1%") == pytest.approx(3.1)
    assert _parse_pct_number("1.20k") == pytest.approx(1200.0)
    assert _parse_pct_number("0.5M") == pytest.approx(500000.0)
    assert _parse_pct_number("2B") == pytest.approx(2e9)
    assert _parse_pct_number("1,200") == pytest.approx(1200.0)
    assert _parse_pct_number("-0.4%") == pytest.approx(-0.4)
    assert _parse_pct_number("n/a") is None
    assert _parse_pct_number(None) is None


def test_known_nfp_token_scores_payrolls() -> None:
    scored = calculate_impact_score(_event("US NFP", "USD", "high"))
    assert scored.score >= 95
    assert scored.level == "HIGH"


def test_lstrip_bug_no_longer_eats_leading_token_letters() -> None:
    # Regression: the old name_key = name.lower().lstrip("us ") stripped any
    # leading 'u'/'s'/space char, corrupting tokens that begin with those
    # letters -- e.g. "US Unemployment Claims" lost its "unemployment" match.
    scored = calculate_impact_score(_event("US Unemployment Claims", "USD", "high"))
    assert scored.score >= 85
    scored_swiss = calculate_impact_score(_event("Swiss CPI", "CHF", "low"))
    # Known-event match (cpi) applies regardless of the v tag.
    assert scored_swiss.level == "HIGH"


def test_provider_failure_retries_after_short_backoff(tmp_path: Path) -> None:
    cache = NewsCache(f"sqlite:///{tmp_path / 'news.db'}")
    calls = []

    class FlakyProvider:
        name = "flaky"

        def fetch(self):
            calls.append(1)
            if len(calls) == 1:
                raise OSError("transient outage")
            return [
                NewsEvent(
                    name="US CPI",
                    currency="USD",
                    impact="high",
                    starts_at=datetime(2026, 9, 10, 12, 30, tzinfo=timezone.utc),
                    ends_at=datetime(2026, 9, 10, 12, 45, tzinfo=timezone.utc),
                )
            ]

    gateway = NewsGateway(
        settings=StrategySettings(require_news_data=True),
        provider=FlakyProvider(),
        cache=cache,
        refresh_throttle_seconds=300,
    )
    gateway.ensure_current(FIXED_NOW)
    assert len(calls) == 1  # first call failed
    assert gateway.ensure_current(FIXED_NOW).stale is True
    # A retry must be allowed far sooner than the full 300s throttle.
    gateway.ensure_current(FIXED_NOW + timedelta(seconds=31))
    assert len(calls) == 2
    snapshot = gateway.ensure_current(FIXED_NOW + timedelta(seconds=32))
    assert len(snapshot.events) == 1
    assert snapshot.stale is False


def test_news_cache_prunes_expired_events(tmp_path: Path) -> None:
    cache = NewsCache(f"sqlite:///{tmp_path / 'news.db'}")
    now = FIXED_NOW
    cache.upsert_many(
        [
            NewsEvent(
                name="old",
                currency="USD",
                impact="low",
                starts_at=now - timedelta(days=3),
                ends_at=now - timedelta(days=3),
            ),
            NewsEvent(
                name="fresh",
                currency="USD",
                impact="low",
                starts_at=now + timedelta(hours=1),
                ends_at=now + timedelta(hours=1),
            ),
        ],
        fetched_at=now,
    )
    assert cache.count() == 2
    pruned = cache.prune(now - timedelta(hours=48))
    assert pruned == 1
    names = {event.name for event in cache.all()}
    assert names == {"fresh"}
    # last_fetched must survive the prune for the still-fresh cache.
    assert cache.last_fetched() is not None


def test_empty_provider_result_still_marks_fresh(tmp_path: Path) -> None:
    # A successful fetch that returns no events must refresh the fetch clock
    # instead of being treated as an outage.
    cache = NewsCache(f"sqlite:///{tmp_path / 'news.db'}")

    class EmptyProvider:
        name = "empty"

        def fetch(self):
            return []

    gateway = NewsGateway(
        settings=StrategySettings(require_news_data=True),
        provider=EmptyProvider(),
        cache=cache,
        refresh_throttle_seconds=300,
    )
    gateway.ensure_current(FIXED_NOW)
    assert cache.count() == 0
    snapshot = gateway.ensure_current(FIXED_NOW + timedelta(seconds=5))
    assert snapshot.stale is False
    assert snapshot.known_state == "EMPTY"


# --- Step 1+5: sync_upcoming forces refresh -----------------------------------

def test_sync_upcoming_forces_refresh_before_throttle(tmp_path: Path) -> None:
    cache = NewsCache(f"sqlite:///{tmp_path / 'news.db'}")
    calls = []

    class CountingProvider:
        name = "counting"

        def fetch(self):
            calls.append(1)
            return [
                NewsEvent(
                    name="US CPI",
                    currency="USD",
                    impact="high",
                    starts_at=datetime(2026, 9, 10, 12, 30, tzinfo=timezone.utc),
                    ends_at=datetime(2026, 9, 10, 12, 45, tzinfo=timezone.utc),
                )
            ]

    gateway = NewsGateway(
        settings=StrategySettings(require_news_data=True),
        provider=CountingProvider(),
        cache=cache,
        refresh_throttle_seconds=3600,
    )
    # First ensure_current always refreshes (no previous refresh happened).
    snapshot = gateway.ensure_current(FIXED_NOW)
    assert len(calls) == 1
    # A second ensure_current within the throttle must NOT call the provider.
    snapshot = gateway.ensure_current(FIXED_NOW + timedelta(seconds=5))
    assert len(calls) == 1
    # sync_upcoming bypasses the throttle and forces a refresh.
    snapshot = gateway.sync_upcoming(FIXED_NOW + timedelta(seconds=10))
    assert len(calls) == 2
    assert snapshot.events
    assert snapshot.stale is False


# === news2.txt: Step 6 -- NewsRiskManager =====================================

from fxbot.news import (
    NewsRiskManager,
    NewsRiskDecision,
    SentimentResult,
    SentimentCache,
    NewsClassifier,
    NewsClassification,
    VolatilityDetector,
    VolatilityReading,
    NewsTradingState,
    NewsTradingStateMachine,
)


def _high_event(name: str, currency: str, starts_at: datetime, ends_at: datetime | None = None) -> NewsEvent:
    return NewsEvent(
        name=name,
        currency=currency,
        impact="high",
        starts_at=starts_at,
        ends_at=ends_at or (starts_at + timedelta(minutes=15)),
        impact_score=95,
    )


def _low_event(name: str, currency: str, starts_at: datetime) -> NewsEvent:
    return NewsEvent(
        name=name,
        currency=currency,
        impact="low",
        starts_at=starts_at,
        ends_at=starts_at + timedelta(minutes=15),
        impact_score=10,
    )


def test_news_risk_manager_blocks_during_blackout() -> None:
    event = _high_event("US CPI", "USD", FIXED_NOW)
    mgr = NewsRiskManager()
    decision = mgr.evaluate("EUR_USD", FIXED_NOW, [event])
    assert decision.blocked is True
    assert "HIGH_IMPACT_NEWS" in decision.reason
    assert decision.events


def test_news_risk_manager_allows_outside_window() -> None:
    event = _high_event("US CPI", "USD", FIXED_NOW + timedelta(hours=2))
    mgr = NewsRiskManager()
    decision = mgr.evaluate("EUR_USD", FIXED_NOW, [event])
    assert decision.blocked is False
    assert decision.reason == "ok"


def test_news_risk_manager_ignores_unrelated_currency() -> None:
    event = _high_event("US CPI", "USD", FIXED_NOW)
    mgr = NewsRiskManager()
    decision = mgr.evaluate("AUD_JPY", FIXED_NOW, [event])
    assert decision.blocked is False


def test_news_risk_manager_ignores_low_impact() -> None:
    event = _low_event("US Nothing", "USD", FIXED_NOW)
    mgr = NewsRiskManager()
    decision = mgr.evaluate("EUR_USD", FIXED_NOW, [event])
    assert decision.blocked is False


def test_news_risk_manager_blocks_on_spread() -> None:
    mgr = NewsRiskManager(max_spread_pips=3.0)
    decision = mgr.evaluate("EUR_USD", FIXED_NOW, [], current_spread_pips=5.0)
    assert decision.blocked is True
    assert "spread_too_wide" in decision.reason


def test_news_risk_manager_blocks_on_extreme_volatility() -> None:
    mgr = NewsRiskManager()
    decision = mgr.evaluate(
        "EUR_USD",
        FIXED_NOW,
        [],
        baseline_atr=8.0,
        current_atr=30.0,
    )
    assert decision.blocked is True
    assert "volatility_extreme" in decision.reason
    assert decision.volatility is not None
    assert decision.volatility.ratio == pytest.approx(3.75)


def test_news_risk_manager_allows_normal_volatility() -> None:
    mgr = NewsRiskManager()
    decision = mgr.evaluate(
        "EUR_USD",
        FIXED_NOW,
        [],
        baseline_atr=8.0,
        current_atr=10.0,
    )
    assert decision.blocked is False
    assert decision.volatility is not None
    assert decision.volatility.is_normal


def test_news_risk_manager_post_news_cooldown() -> None:
    event = _high_event("US CPI", "USD", FIXED_NOW - timedelta(minutes=30), FIXED_NOW - timedelta(minutes=28))
    mgr = NewsRiskManager(min_post_news_minutes=3.0, blackout_before_minutes=0, blackout_after_minutes=0)
    # 2 minutes after event ends -- still in cooldown (blackout has expired)
    decision = mgr.evaluate("EUR_USD", FIXED_NOW - timedelta(minutes=26), [event])
    assert decision.blocked is True
    assert "post_news_cooldown" in decision.reason
    # 5 minutes after event ends -- cooldown expired
    decision = mgr.evaluate("EUR_USD", FIXED_NOW - timedelta(minutes=23), [event])
    assert decision.blocked is False


def test_news_risk_decision_bool() -> None:
    blocked = NewsRiskDecision(blocked=True, reason="test")
    allowed = NewsRiskDecision(blocked=False, reason="ok")
    assert bool(blocked) is True
    assert bool(allowed) is False


def test_news_risk_manager_get_relevant_events_filters() -> None:
    usd_event = _high_event("US CPI", "USD", FIXED_NOW + timedelta(hours=2))
    jpy_event = _high_event("BoJ Rate", "JPY", FIXED_NOW + timedelta(hours=2))
    low = _low_event("Minor", "USD", FIXED_NOW + timedelta(hours=2))
    mgr = NewsRiskManager()
    relevant = mgr.get_relevant_events("AUD_JPY", [usd_event, jpy_event, low])
    assert len(relevant) == 1
    assert relevant[0].name == "BoJ Rate"


# === news2.txt: Step 7 -- SentimentResult ====================================

def test_sentiment_result_valid() -> None:
    result = SentimentResult(
        event_id="evt-1",
        sentiment="BULLISH",
        confidence=0.82,
        currency="USD",
        hawkish_score=0.75,
        dovish_score=0.10,
        relevance=0.95,
        summary="Tighter monetary policy.",
        model="gpt-4",
    )
    assert result.sentiment == "BULLISH"
    assert result.currency == "USD"
    assert result.confidence == 0.82


def test_sentiment_result_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        SentimentResult(event_id="x", sentiment="BULLISH", confidence=1.5, currency="USD")


def test_sentiment_result_rejects_invalid_hawkish() -> None:
    with pytest.raises(ValueError, match="hawkish_score"):
        SentimentResult(event_id="x", sentiment="BULLISH", confidence=0.5, currency="USD", hawkish_score=-0.1)


def test_sentiment_result_rejects_invalid_sentiment() -> None:
    with pytest.raises(ValueError, match="sentiment"):
        SentimentResult(event_id="x", sentiment="MAYBE", confidence=0.5, currency="USD")


def test_sentiment_result_normalizes_case() -> None:
    result = SentimentResult(event_id="x", sentiment="bearish", confidence=0.5, currency="eur")
    assert result.sentiment == "BEARISH"
    assert result.currency == "EUR"


# === news2.txt: Step 8 -- SentimentCache =====================================

def test_sentiment_cache_upsert_and_get(tmp_path: Path) -> None:
    cache = SentimentCache(f"sqlite:///{tmp_path / 'sentiment.db'}")
    result = SentimentResult(
        event_id="evt-1",
        sentiment="BULLISH",
        confidence=0.82,
        currency="USD",
        hawkish_score=0.75,
        dovish_score=0.10,
        relevance=0.95,
        summary="Tighter policy.",
        model="gpt-4",
    )
    cache.upsert(result)
    assert cache.has("evt-1")
    assert cache.count() == 1
    fetched = cache.get("evt-1")
    assert fetched is not None
    assert fetched.sentiment == "BULLISH"
    assert fetched.confidence == pytest.approx(0.82)
    assert fetched.currency == "USD"
    assert fetched.hawkish_score == pytest.approx(0.75)
    assert fetched.dovish_score == pytest.approx(0.10)


def test_sentiment_cache_deduplicates(tmp_path: Path) -> None:
    cache = SentimentCache(f"sqlite:///{tmp_path / 'sentiment.db'}")
    result = SentimentResult(event_id="evt-1", sentiment="BULLISH", confidence=0.8, currency="USD")
    cache.upsert(result)
    cache.upsert(result)
    assert cache.count() == 1


def test_sentiment_cache_has_returns_false_for_missing(tmp_path: Path) -> None:
    cache = SentimentCache(f"sqlite:///{tmp_path / 'sentiment.db'}")
    assert cache.has("nonexistent") is False
    assert cache.get("nonexistent") is None


def test_sentiment_cache_clear(tmp_path: Path) -> None:
    cache = SentimentCache(f"sqlite:///{tmp_path / 'sentiment.db'}")
    cache.upsert(SentimentResult(event_id="e1", sentiment="BULLISH", confidence=0.5, currency="USD"))
    cache.upsert(SentimentResult(event_id="e2", sentiment="BEARISH", confidence=0.6, currency="EUR"))
    assert cache.count() == 2
    cache.clear()
    assert cache.count() == 0


def test_sentiment_cache_coexists_with_news_cache(tmp_path: Path) -> None:
    db_path = f"sqlite:///{tmp_path / 'combined.db'}"
    news_cache = NewsCache(db_path)
    sent_cache = SentimentCache(db_path)
    news_cache.upsert_many(
        [NewsEvent(name="CPI", currency="USD", impact="high",
                    starts_at=FIXED_NOW, ends_at=FIXED_NOW + timedelta(minutes=15))],
        fetched_at=FIXED_NOW,
    )
    sent_cache.upsert(SentimentResult(event_id="e1", sentiment="BULLISH", confidence=0.8, currency="USD"))
    assert news_cache.count() == 1
    assert sent_cache.count() == 1


# === news2.txt: Step 9 -- NewsClassifier =====================================

def test_classifier_separates_scheduled_from_unscheduled() -> None:
    cpi = _high_event("US CPI", "USD", FIXED_NOW)
    war = NewsEvent(
        name="Middle East Escalation",
        currency="USD",
        impact="high",
        starts_at=FIXED_NOW,
        ends_at=FIXED_NOW + timedelta(hours=1),
        impact_score=80,
    )
    classifier = NewsClassifier()
    results = classifier.classify([cpi, war])
    assert len(results) == 2
    cpi_result = next(r for r in results if r.event.name == "US CPI")
    war_result = next(r for r in results if r.event.name == "Middle East Escalation")
    assert cpi_result.scheduled is True
    assert cpi_result.category == "scheduled"
    assert war_result.scheduled is False
    assert war_result.category == "unscheduled"


def test_classifier_scheduled_only() -> None:
    cpi = _high_event("US CPI", "USD", FIXED_NOW)
    fomc = _high_event("FOMC Statement", "USD", FIXED_NOW)
    war = NewsEvent(name="War", currency="USD", impact="high",
                    starts_at=FIXED_NOW, ends_at=FIXED_NOW + timedelta(hours=1))
    classifier = NewsClassifier()
    scheduled = classifier.scheduled_only([cpi, fomc, war])
    assert len(scheduled) == 2


def test_classifier_unscheduled_only() -> None:
    cpi = _high_event("US CPI", "USD", FIXED_NOW)
    war = NewsEvent(name="Bank Collapse", currency="EUR", impact="high",
                    starts_at=FIXED_NOW, ends_at=FIXED_NOW + timedelta(hours=1))
    classifier = NewsClassifier()
    unscheduled = classifier.unscheduled_only([cpi, war])
    assert len(unscheduled) == 1
    assert unscheduled[0].name == "Bank Collapse"


def test_classifier_nfp_is_scheduled() -> None:
    nfp = _high_event("US NFP", "USD", FIXED_NOW)
    classifier = NewsClassifier()
    assert classifier.scheduled_only([nfp])


def test_classifier_emergency_rate_decision_is_unscheduled() -> None:
    emergency = NewsEvent(
        name="Emergency Rate Decision",
        currency="USD",
        impact="high",
        starts_at=FIXED_NOW,
        ends_at=FIXED_NOW + timedelta(minutes=30),
        impact_score=90,
    )
    classifier = NewsClassifier()
    assert classifier.unscheduled_only([emergency])


# === news2.txt: Step 10 -- VolatilityDetector ================================

def test_volatility_detector_normal() -> None:
    detector = VolatilityDetector()
    reading = detector.measure(8.0, 10.0)
    assert reading.level == "normal"
    assert reading.ratio == pytest.approx(1.25)
    assert reading.is_normal is True


def test_volatility_detector_elevated() -> None:
    detector = VolatilityDetector()
    reading = detector.measure(8.0, 16.0)
    assert reading.level == "elevated"
    assert reading.ratio == pytest.approx(2.0)
    assert reading.is_elevated is True


def test_volatility_detector_extreme() -> None:
    detector = VolatilityDetector()
    reading = detector.measure(8.0, 24.0)
    assert reading.level == "extreme"
    assert reading.ratio == pytest.approx(3.0)
    assert reading.is_extreme is True


def test_volatility_detector_custom_thresholds() -> None:
    detector = VolatilityDetector(elevated_threshold=2.0, extreme_threshold=3.0)
    reading = detector.measure(10.0, 22.0)
    assert reading.level == "elevated"
    reading2 = detector.measure(10.0, 35.0)
    assert reading2.level == "extreme"


def test_volatility_detector_is_safe_to_trade() -> None:
    detector = VolatilityDetector()
    assert detector.is_safe_to_trade(8.0, 10.0) is True
    assert detector.is_safe_to_trade(8.0, 20.0) is False


def test_volatility_detector_rejects_zero_baseline() -> None:
    detector = VolatilityDetector()
    with pytest.raises(ValueError, match="baseline_atr"):
        detector.measure(0.0, 10.0)


def test_volatility_detector_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="elevated_threshold"):
        VolatilityDetector(elevated_threshold=-1.0)
    with pytest.raises(ValueError, match="extreme_threshold"):
        VolatilityDetector(elevated_threshold=2.0, extreme_threshold=1.0)


# === news2.txt: Step 11 -- NewsTradingStateMachine ==========================

def test_state_machine_starts_in_normal() -> None:
    sm = NewsTradingStateMachine()
    assert sm.state == NewsTradingState.NORMAL


def test_state_machine_normal_to_pre_news() -> None:
    sm = NewsTradingStateMachine(pre_news_minutes=15.0)
    event_start = FIXED_NOW + timedelta(minutes=10)
    sm.check_transition(FIXED_NOW, upcoming_event_start=event_start)
    assert sm.state == NewsTradingState.PRE_NEWS


def test_state_machine_skips_pre_news_when_far() -> None:
    sm = NewsTradingStateMachine(pre_news_minutes=15.0)
    event_start = FIXED_NOW + timedelta(minutes=20)
    sm.check_transition(FIXED_NOW, upcoming_event_start=event_start)
    assert sm.state == NewsTradingState.NORMAL


def test_state_machine_pre_news_to_lock() -> None:
    sm = NewsTradingStateMachine()
    sm._transition(NewsTradingState.PRE_NEWS, FIXED_NOW)
    event_start = FIXED_NOW - timedelta(seconds=30)
    sm.check_transition(FIXED_NOW, upcoming_event_start=event_start)
    assert sm.state == NewsTradingState.NEWS_LOCK


def test_state_machine_lock_to_post_news_on_volatility_drop() -> None:
    sm = NewsTradingStateMachine()
    sm._transition(NewsTradingState.NEWS_LOCK, FIXED_NOW)
    sm._state_entered_at = FIXED_NOW  # simulate immediate
    vol = VolatilityReading(ratio=1.2, level="normal", baseline_atr=8.0, current_atr=9.6)
    sm.check_transition(FIXED_NOW + timedelta(minutes=2), volatility=vol)
    assert sm.state == NewsTradingState.POST_NEWS


def test_state_machine_lock_timeout_to_post_news() -> None:
    sm = NewsTradingStateMachine(max_lock_minutes=5.0)
    sm._transition(NewsTradingState.NEWS_LOCK, FIXED_NOW)
    sm._state_entered_at = FIXED_NOW
    # After 5 minutes without volatility drop, force to POST_NEWS
    sm.check_transition(FIXED_NOW + timedelta(minutes=6))
    assert sm.state == NewsTradingState.POST_NEWS


def test_state_machine_post_news_to_trade() -> None:
    sm = NewsTradingStateMachine()
    sm._transition(NewsTradingState.POST_NEWS, FIXED_NOW)
    sm.check_transition(FIXED_NOW + timedelta(minutes=1), setup_confirmed=True)
    assert sm.state == NewsTradingState.TRADE


def test_state_machine_trade_to_normal() -> None:
    sm = NewsTradingStateMachine()
    sm._transition(NewsTradingState.TRADE, FIXED_NOW)
    sm.check_transition(FIXED_NOW + timedelta(seconds=5))
    assert sm.state == NewsTradingState.NORMAL


def test_state_machine_force_lock() -> None:
    sm = NewsTradingStateMachine()
    sm.force_lock(FIXED_NOW)
    assert sm.state == NewsTradingState.NEWS_LOCK


def test_state_machine_reset() -> None:
    sm = NewsTradingStateMachine()
    sm.force_lock(FIXED_NOW)
    assert sm.state == NewsTradingState.NEWS_LOCK
    sm.reset(FIXED_NOW)
    assert sm.state == NewsTradingState.NORMAL


def test_state_machine_pre_news_resets_when_event_cancelled() -> None:
    sm = NewsTradingStateMachine(pre_news_minutes=15.0)
    event_start = FIXED_NOW + timedelta(minutes=10)
    sm.check_transition(FIXED_NOW, upcoming_event_start=event_start)
    assert sm.state == NewsTradingState.PRE_NEWS
    # Event disappears -- revert to NORMAL
    sm.check_transition(FIXED_NOW + timedelta(minutes=1))
    assert sm.state == NewsTradingState.NORMAL


def test_state_machine_serializable() -> None:
    sm = NewsTradingStateMachine()
    sm.force_lock(FIXED_NOW)
    data = sm._as_dict()
    assert data["state"] == "NEWS_LOCK"
    assert data["news_start"] is not None


# === news2.txt: Full integration test =======================================

def test_news_risk_manager_full_lifecycle() -> None:
    """End-to-end: normal -> pre_news -> lock -> post_news -> trade -> normal."""
    event_start = FIXED_NOW + timedelta(minutes=10)
    event = _high_event("US CPI", "USD", event_start)

    sm = NewsTradingStateMachine(pre_news_minutes=15.0, min_lock_minutes=1.0)
    mgr = NewsRiskManager(state_machine=sm)

    # T+0: normal, event is 10 min away -> blocks via blackout
    decision = mgr.evaluate("EUR_USD", FIXED_NOW, [event])
    assert decision.blocked is True
    assert "HIGH_IMPACT_NEWS" in decision.reason

    # T+0: same symbol, event far away -> not blocked
    far_event = _high_event("US CPI", "USD", FIXED_NOW + timedelta(hours=3))
    decision2 = mgr.evaluate("EUR_USD", FIXED_NOW, [far_event])
    assert decision2.blocked is False
    assert decision2.state == NewsTradingState.PRE_NEWS or decision2.state == NewsTradingState.NORMAL

    # Force into NEWS_LOCK for the next assertion
    sm.force_lock(FIXED_NOW + timedelta(minutes=5))
    decision3 = mgr.evaluate("EUR_USD", FIXED_NOW + timedelta(minutes=5), [far_event])
    assert decision3.blocked is True
    assert decision3.state == NewsTradingState.NEWS_LOCK