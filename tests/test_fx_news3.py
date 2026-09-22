"""Tests for news3.txt: news-as-feature, backtesting, kill switch, audit logging."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from fxbot.config import NewsEvent
from fxbot.instruments import split_instrument_name
from fxbot.journal import StructuredJournal
from fxbot.news import (
    NewsAuditRecorder,
    NewsFeatureEngine,
    NewsRiskDecision,
    NewsRiskManager,
    NewsSnapshot,
    NewsTradingState,
    SentimentResult,
    news_known,
    news_status_from_snapshot,
    sentiment_contribution,
)
from fxbot.news_backtest import (
    BacktestTradeRecord,
    BlackoutWindowResult,
    DEFAULT_BLACKOUT_WINDOWS,
    ExperimentMetrics,
    MarketContext,
    StrategyExperiment,
    apply_blackout_window,
    compare_blackout_windows,
    compare_strategy_experiments,
    compute_group_stats,
    default_experiment_set,
    default_extreme_at_entry,
    group_news_stats,
    run_experiment,
    sentiment_confirm,
    symbol_currencies,
    tag_trades_with_events,
    trade_within_window,
)

FIXED_NOW = datetime(2026, 9, 10, 12, 10, tzinfo=timezone.utc)


def _event(name: str, currency: str, start: datetime) -> NewsEvent:
    return NewsEvent(
        name=name,
        currency=currency,
        impact="high",
        starts_at=start,
        ends_at=start + timedelta(minutes=15),
        impact_score=95,
    )


def _trade(
    trade_id: str,
    symbol: str,
    entry_time: datetime,
    side: str = "BUY",
    r_multiple: float = 0.5,
    spread_pips: float | None = None,
    mae: float = 0.0,
    mfe: float = 1.0,
) -> BacktestTradeRecord:
    context = MarketContext(price=1.1000, spread_pips=spread_pips or 0.5, atr_pips=10.0) if spread_pips else None
    return BacktestTradeRecord(
        trade_id=trade_id,
        symbol=symbol,
        entry_time=entry_time,
        side=side,
        r_multiple=r_multiple,
        mae=mae,
        mfe=mfe,
        context=context,
    )


# === Step 13: news as a feature ==============================================


def test_sentiment_contribution_map() -> None:
    assert sentiment_contribution("BULLISH") == 1
    assert sentiment_contribution("bearish") == -1
    assert sentiment_contribution("NEUTRAL") == 0
    assert sentiment_contribution("") == 0
    assert sentiment_contribution(None) == 0


def test_news_feature_engine_applies_symbol_sentiment() -> None:
    engine = NewsFeatureEngine()
    result = SentimentResult(event_id="e1", sentiment="BULLISH", confidence=0.82, currency="USD")
    feature = engine.apply("EUR_USD", [result])
    assert feature.applied is True
    assert feature.contribution == 1
    assert feature.sentiment == "BULLISH"
    assert feature.currency == "USD"


def test_news_feature_engine_ignores_unrelated_currency() -> None:
    engine = NewsFeatureEngine()
    result = SentimentResult(event_id="e1", sentiment="BULLISH", confidence=0.82, currency="JPY")
    feature = engine.apply("EUR_USD", [result])
    assert feature.applied is False
    assert feature.contribution == 0


def test_news_feature_engine_picks_highest_confidence() -> None:
    engine = NewsFeatureEngine()
    low = SentimentResult(event_id="e1", sentiment="BULLISH", confidence=0.3, currency="EUR")
    high = SentimentResult(event_id="e2", sentiment="BEARISH", confidence=0.9, currency="EUR")
    feature = engine.apply("EUR_USD", [low, high])
    assert feature.sentiment == "BEARISH"
    assert feature.contribution == -1


def test_news_feature_contribution_is_small_relative_to_threshold() -> None:
    # Spec: score >= 4 => BUY; a single sentiment point must not dominate.
    technical = 3
    engine = NewsFeatureEngine()
    result = SentimentResult(event_id="e", sentiment="BULLISH", confidence=0.9, currency="USD")
    contributed = engine.apply("EUR_USD", [result])
    assert technical + contributed.contribution == 4
    assert technical + contributed.contribution >= 4


# === Step 17: fail-closed kill switch ========================================


def test_news_status_from_fresh_snapshot() -> None:
    snapshot = NewsSnapshot(events=[_event("CPI", "USD", FIXED_NOW)], stale=False)
    assert news_status_from_snapshot(snapshot) == "FRESH"
    assert news_known("FRESH") is True


def test_news_status_from_empty_snapshot() -> None:
    snapshot = NewsSnapshot(events=[], stale=False)
    assert news_status_from_snapshot(snapshot) == "EMPTY"
    assert news_known("EMPTY") is True


def test_news_status_from_stale_snapshot_is_unknown() -> None:
    snapshot = NewsSnapshot(events=[], stale=True)
    assert news_status_from_snapshot(snapshot) == "UNKNOWN"
    assert news_known("UNKNOWN") is False


def test_news_status_none_is_unknown() -> None:
    assert news_status_from_snapshot(None) == "UNKNOWN"


def test_fail_closed_blocks_on_unknown_calendar() -> None:
    stale_snapshot = NewsSnapshot(events=[], stale=True)
    mgr = NewsRiskManager(fail_closed=True)
    decision = mgr.evaluate("EUR_USD", FIXED_NOW, [], news_snapshot=stale_snapshot)
    assert decision.blocked is True
    assert decision.reason == "news_status_unknown"


def test_fail_closed_allows_fresh_calendar() -> None:
    fresh_snapshot = NewsSnapshot(events=[_event("CPI", "USD", FIXED_NOW + timedelta(hours=2))], stale=False)
    mgr = NewsRiskManager(fail_closed=True)
    decision = mgr.evaluate("EUR_USD", FIXED_NOW, [], news_snapshot=fresh_snapshot)
    assert decision.blocked is False


def test_not_fail_closed_ignores_unknown_calendar() -> None:
    unknown = NewsSnapshot(events=[], stale=True)
    mgr = NewsRiskManager(fail_closed=False)
    decision = mgr.evaluate("EUR_USD", FIXED_NOW, [], news_snapshot=unknown)
    assert decision.blocked is False


# === Step 18: NewsAuditRecorder ==============================================


@dataclass
class _FakeJournal:
    events: list[dict] | None = None

    def __post_init__(self) -> None:
        self.events = []

    def log_event(self, event_type: str, message: str, *, level: str = "info", payload: dict | None = None) -> None:
        self.events.append({"event_type": event_type, "message": message, "level": level, "payload": payload})


def test_audit_recorder_logs_news_event() -> None:
    journal = _FakeJournal()
    recorder = NewsAuditRecorder(journal)
    event = _event("US CPI", "USD", FIXED_NOW)
    recorder.log_event_released(event)
    assert journal.events[-1]["event_type"] == "news_audit.event"
    assert "US CPI" in journal.events[-1]["message"]
    assert journal.events[-1]["payload"]["currency"] == "USD"


def test_audit_recorder_logs_forecast_actual() -> None:
    journal = _FakeJournal()
    recorder = NewsAuditRecorder(journal)
    event = NewsEvent(
        name="US CPI",
        currency="USD",
        impact="high",
        starts_at=FIXED_NOW,
        ends_at=FIXED_NOW + timedelta(minutes=15),
        previous=3.3,
        forecast=3.1,
        actual=3.5,
    )
    recorder.log_forecast(event)
    recorder.log_previous(event)
    recorder.log_actual(event)
    types = [e["event_type"] for e in journal.events]
    assert "news_audit.forecast" in types
    assert "news_audit.actual" in types
    assert journal.events[-1]["payload"]["actual"] == 3.5


def test_audit_recorder_full_timeline() -> None:
    journal = _FakeJournal()
    recorder = NewsAuditRecorder(journal)
    event = _event("US CPI", "USD", FIXED_NOW)
    recorder.log_event_released(event)
    recorder.log_blocked("EUR_USD", "HIGH_IMPACT_NEWS", event=event)
    recorder.log_unlocked("EUR_USD", spread_pips=1.2, atr_ratio=2.1)
    recorder.log_sentiment(
        SentimentResult(event_id="e", sentiment="BEARISH", confidence=0.8, currency="USD")
    )
    recorder.log_final_signal("EUR_USD", technical_signal="BUY", news_contribution=-1, final_score=3, side=None)
    recorder.log_order_executed("EUR_USD", side="SELL", units=1000, entry_price=1.10)
    types = [e["event_type"] for e in journal.events]
    assert types == [
        "news_audit.event",
        "news_audit.blocked",
        "news_audit.unlocked",
        "news_audit.sentiment",
        "news_audit.signal",
        "news_audit.order",
    ]
    assert "ORDER EXECUTED" in journal.events[-1]["message"]


def test_audit_recorder_works_with_structured_journal(tmp_path) -> None:
    journal = StructuredJournal(f"sqlite:///{tmp_path / 'audit.db'}")
    try:
        driver = NewsAuditRecorder(journal)
        driver.log_event_released(_event("US CPI", "USD", FIXED_NOW))
        driver.log_blocked("EUR_USD", "news_lock")
        events = journal.recent_events(limit=10)
        types = [e.event_type for e in events]
        assert "news_audit.event" in types
        assert "news_audit.blocked" in types
    finally:
        journal.close()


# === Step 14: news backtesting engine ========================================


def test_symbol_currencies() -> None:
    assert symbol_currencies("EUR_USD") == {"EUR", "USD"}
    assert symbol_currencies("USDJPY") == {"USD", "JPY"}


def test_tag_trades_with_events_attaches_closest() -> None:
    usd_cpi = _event("US CPI", "USD", FIXED_NOW)
    jpy_boj = _event("BoJ", "JPY", FIXED_NOW + timedelta(hours=1))
    before = _trade("t1", "EUR_USD", FIXED_NOW - timedelta(minutes=5))
    after = _trade("t2", "EUR_USD", FIXED_NOW + timedelta(minutes=10))
    untouchable = _trade("t3", "AUD_JPY", FIXED_NOW)
    tagged = tag_trades_with_events([before, after, untouchable], [usd_cpi, jpy_boj])
    assert tagged[0].event_id == usd_cpi.event_id
    assert tagged[0].before_event is True
    assert tagged[0].minutes_from_event == pytest.approx(5.0)
    assert tagged[1].before_event is False
    assert tagged[2].event_id is None  # no relevant event


def test_compute_group_stats() -> None:
    records = [
        _trade("1", "EUR_USD", FIXED_NOW, r_multiple=1.0),
        _trade("2", "EUR_USD", FIXED_NOW, r_multiple=-0.5),
        _trade("3", "EUR_USD", FIXED_NOW, r_multiple=1.0),
    ]
    stats = compute_group_stats("g", records)
    assert stats.trades == 3
    assert stats.win_rate == pytest.approx(2 / 3)
    assert stats.avg_r == pytest.approx(0.5)
    assert stats.profit_factor == pytest.approx(2.0 / 0.5)
    assert stats.max_drawdown_r == pytest.approx(0.5)


def test_group_news_stats_before_after() -> None:
    cpi = _event("US CPI", "USD", FIXED_NOW)
    before = _trade("1", "EUR_USD", FIXED_NOW - timedelta(minutes=1), r_multiple=1.0)
    after = _trade("2", "EUR_USD", FIXED_NOW + timedelta(minutes=1), r_multiple=-0.5)
    tagged = tag_trades_with_events([before, after], [cpi])
    groups = group_news_stats(tagged)
    assert len(groups) == 2
    by_key = {g.group: g for g in groups}
    assert "US CPI:before" in by_key
    assert "US CPI:after" in by_key
    assert by_key["US CPI:before"].avg_r == pytest.approx(1.0)
    assert by_key["US CPI:after"].avg_r == pytest.approx(-0.5)


# === Step 15: blackout window comparison =====================================


def test_trade_within_window() -> None:
    cpi = _event("US CPI", "USD", FIXED_NOW)
    near = _trade("1", "EUR_USD", FIXED_NOW + timedelta(minutes=10))
    far = _trade("2", "EUR_USD", FIXED_NOW + timedelta(minutes=45))
    [near_tagged, far_tagged] = tag_trades_with_events([near, far], [cpi])
    assert trade_within_window(near_tagged, 15) is True
    assert trade_within_window(near_tagged, 5) is False
    assert trade_within_window(far_tagged, 15) is False
    assert trade_within_window(near_tagged, None) is False  # no blackout


def test_apply_blackout_window_filters() -> None:
    cpi = _event("US CPI", "USD", FIXED_NOW)
    near = _trade("1", "EUR_USD", FIXED_NOW + timedelta(minutes=5))
    far = _trade("2", "EUR_USD", FIXED_NOW + timedelta(minutes=45))
    tagged = tag_trades_with_events([near, far], [cpi])
    allowed = apply_blackout_window(tagged, 30)
    assert len(allowed) == 1
    assert allowed[0].trade_id == "2"


def test_compare_blackout_windows_reports_all_windows() -> None:
    cpi = _event("US CPI", "USD", FIXED_NOW)
    trades = [
        _trade(f"{i}", "EUR_USD", FIXED_NOW + timedelta(minutes=delta), r_multiple=(1.0 if i % 2 == 0 else -0.5))
        for i, delta in enumerate([3, 8, 14, 22, 35, 50, 65])
    ]
    tagged = tag_trades_with_events(trades, [cpi])
    results = compare_blackout_windows(tagged)
    assert len(results) == len(DEFAULT_BLACKOUT_WINDOWS)
    assert all(isinstance(r, BlackoutWindowResult) for r in results)
    # More trades survive at smaller windows.
    counts = {r.window_minutes: r.trades for r in results}
    assert counts[None] > counts[5] > counts[15] > counts[60]


def test_compare_blackout_windows_includes_baseline_none() -> None:
    cpi = _event("US CPI", "USD", FIXED_NOW)
    trades = [_trade("1", "EUR_USD", FIXED_NOW + timedelta(minutes=2))]
    tagged = tag_trades_with_events(trades, [cpi])
    results = compare_blackout_windows(tagged)
    baseline = next(r for r in results if r.window_minutes is None)
    assert baseline.trades == 1


# === Step 16: feature ablation A/B/C/D =======================================


def _sentiment_map() -> dict[str, SentimentResult]:
    return {
        "USD": SentimentResult(event_id="e1", sentiment="BULLISH", confidence=0.9, currency="USD"),
        "EUR": SentimentResult(event_id="e2", sentiment="BULLISH", confidence=0.9, currency="EUR"),
    }


def test_sentiment_confirm_matches_side() -> None:
    smap = _sentiment_map()
    cpi = _event("US CPI", "USD", FIXED_NOW)
    buy = tag_trades_with_events([_trade("1", "EUR_USD", FIXED_NOW + timedelta(minutes=20), side="BUY")], [cpi])[0]
    sell = tag_trades_with_events([_trade("2", "EUR_USD", FIXED_NOW + timedelta(minutes=20), side="SELL")], [cpi])[0]
    assert sentiment_confirm(buy, smap) is True
    assert sentiment_confirm(sell, smap) is False


def test_run_experiment_technical_only_keeps_everything() -> None:
    cpi = _event("US CPI", "USD", FIXED_NOW)
    near = _trade("1", "EUR_USD", FIXED_NOW + timedelta(minutes=5), side="BUY")
    far = _trade("2", "EUR_USD", FIXED_NOW + timedelta(minutes=45), side="BUY")
    tagged = tag_trades_with_events([near, far], [cpi])
    exp = StrategyExperiment(label="A", name="Technical only")
    metrics = run_experiment(tagged, exp, sentiment_map=_sentiment_map())
    assert metrics.trade_count == 2


def test_run_experiment_blackout_removes_near_trades() -> None:
    cpi = _event("US CPI", "USD", FIXED_NOW)
    near = _trade("1", "EUR_USD", FIXED_NOW + timedelta(minutes=5), side="BUY")
    far = _trade("2", "EUR_USD", FIXED_NOW + timedelta(minutes=45), side="BUY")
    tagged = tag_trades_with_events([near, far], [cpi])
    exp = StrategyExperiment(label="B", name="+ blackout", blackout_window_minutes=30)
    metrics = run_experiment(tagged, exp)
    assert metrics.trade_count == 1


def test_run_experiment_sentiment_confirm_filters_direction() -> None:
    cpi = _event("US CPI", "USD", FIXED_NOW)
    buy = _trade("1", "EUR_USD", FIXED_NOW + timedelta(minutes=45), side="BUY")
    sell = _trade("2", "EUR_USD", FIXED_NOW + timedelta(minutes=45), side="SELL")
    tagged = tag_trades_with_events([buy, sell], [cpi])
    exp = StrategyExperiment(label="C", name="+ sentiment", require_sentiment_confirm=True)
    metrics = run_experiment(tagged, exp, sentiment_map=_sentiment_map())
    assert metrics.trade_count == 1
    assert metrics.trade_count >= 1


def test_default_extreme_at_entry_blocks_extreme_ratio() -> None:
    import fxbot.news_backtest as nb

    cpi = _event("US CPI", "USD", FIXED_NOW + timedelta(hours=1))
    calm = _trade("1", "EUR_USD", FIXED_NOW + timedelta(hours=2), side="BUY")
    wild = _trade("2", "EUR_USD", FIXED_NOW - timedelta(minutes=2), side="BUY")
    tagged = tag_trades_with_events([calm, wild], [cpi])
    readings = {
        "EUR_USD": [
            (FIXED_NOW - timedelta(minutes=4), 3.5),  # extreme, right before wild entry
            (FIXED_NOW + timedelta(hours=2), 1.0),  # calm
        ]
    }
    guard = default_extreme_at_entry(readings, lookback_minutes=5.0)
    assert guard(tagged[0]) is False
    assert guard(tagged[1]) is True


def test_run_experiment_extreme_volatility_filters() -> None:
    cpi = _event("US CPI", "USD", FIXED_NOW + timedelta(hours=2))
    calm = _trade("1", "EUR_USD", FIXED_NOW + timedelta(hours=1), side="BUY")
    wild = _trade("2", "EUR_USD", FIXED_NOW - timedelta(minutes=2), side="BUY")
    tagged = tag_trades_with_events([calm, wild], [cpi])
    readings = {"EUR_USD": [(FIXED_NOW - timedelta(minutes=4), 3.5)]}
    guard = default_extreme_at_entry(readings, lookback_minutes=5.0)
    exp = StrategyExperiment(label="D", name="+ vol", block_extreme_volatility=True)
    metrics = run_experiment(tagged, exp, extreme_at_entry=guard)
    assert metrics.trade_count == 1


def test_compare_strategy_experiments_returns_table() -> None:
    cpi = _event("US CPI", "USD", FIXED_NOW)
    trades = [
        _trade("1", "EUR_USD", FIXED_NOW + timedelta(minutes=60), side="BUY", r_multiple=1.0, mae=0.3, mfe=1.5),
        _trade("2", "EUR_USD", FIXED_NOW + timedelta(minutes=5), side="BUY", r_multiple=-0.4, mae=0.8, mfe=0.2),
        _trade("3", "EUR_USD", FIXED_NOW + timedelta(minutes=45), side="SELL", r_multiple=1.2, mae=0.5, mfe=1.8),
    ]
    tagged = tag_trades_with_events(trades, [cpi])
    experiments = default_experiment_set(blackout_window_minutes=15)
    results = compare_strategy_experiments(tagged, experiments, sentiment_map=_sentiment_map())
    assert [r.label for r in results] == ["A", "B", "C", "D"]
    assert all(isinstance(r, ExperimentMetrics) for r in results)
    baseline = results[0]
    # Adding a blackout never increases trade count.
    assert baseline.trade_count >= results[1].trade_count


def test_average_spread_reported_as_slippage() -> None:
    cpi = _event("US CPI", "USD", FIXED_NOW + timedelta(hours=2))
    trade = _trade("1", "EUR_USD", FIXED_NOW + timedelta(hours=1), side="BUY", spread_pips=2.5)
    tagged = tag_trades_with_events([trade], [cpi])
    exp = StrategyExperiment(label="A", name="Technical only")
    metrics = run_experiment(tagged, exp)
    assert metrics.avg_spread_pips == pytest.approx(2.5)


def test_default_experiment_set_shape() -> None:
    experiments = default_experiment_set()
    assert len(experiments) == 4
    assert experiments[0].label == "A"
    assert experiments[0].blackout_window_minutes is None
    assert experiments[3].block_extreme_volatility is True