"""Concrete failures discovered while auditing PR #12; fake broker only."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import hashlib
import json
from threading import Event
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import Mock

import joblib
import pytest

from fxbot.ai.exit_intelligence import (
    EXIT_ACTIONS, EXIT_FEATURE_COLUMNS, EXIT_FEATURE_VERSION,
    ExitPredictionService, build_exit_snapshot,
)
from fxbot.ai.exit_observation_dataset import build_observed_exit_rows, export_observed_exit_rows
from fxbot.ai.exit_observer import ExitObservationQueue
from fxbot.ai.exit_policy import ExitPolicySettings, evaluate_exit_policy
from fxbot.ai.model_registry import ModelRegistry, RegistryError
from fxbot.config import BrokerSettings, ExitAiSettings, FxBotSettings, RuntimeSettings
from fxbot.forward import ForwardTestWorker
from test_fx_exit_intelligence import _inputs, FakeExitEstimator
from test_fx_exit_observation_dataset import _observation
from test_fx_exit_policy import _sample, _evaluate
from test_fx_model_registry import _candidate


def test_missing_excursions_are_unknown_and_future_telemetry_is_excluded():
    now, trade, quote, instrument = _inputs()
    snap = build_exit_snapshot(trade, quote, instrument, now)
    assert snap["mfe_pips"] is None and snap["mae_pips"] is None
    assert snap["drawdown_from_peak_pips"] is None
    assert snap["excursion_quality"] == "historical_excursions_unavailable"
    snap = build_exit_snapshot(trade, quote, instrument, now, recorded_payload={
        "sniper_excursions": {"mfe": .9, "mae": .9, "last_sample": (now+timedelta(seconds=1)).isoformat()}})
    assert snap["mfe_pips"] is None


def test_configured_quote_freshness_jpy_and_initial_stop_attribution():
    now, trade, quote, instrument = _inputs(-1000)
    quote.time = now - timedelta(seconds=20)
    with pytest.raises(ValueError, match="fresh"):
        build_exit_snapshot(trade, quote, instrument, now, maximum_quote_age_seconds=15)
    instrument.name = trade["instrument"] = "USD_JPY"
    instrument.pip_size = .01
    trade["price"], quote.bid, quote.ask = 150., 149.98, 150.01
    trade["mt5"] = {"volume": .01}
    snap = build_exit_snapshot(trade, quote, instrument, now, recorded_payload={
        "strategy_context": {"candidate_id": "c", "initial_stop": 151., "strategy_hash": "old-strategy"}},
        strategy_version="new-strategy")
    assert snap["pnl_pips"] == pytest.approx(-1)
    assert snap["spread_pips"] == pytest.approx(3)
    assert snap["broker_volume_lots"] == .01
    assert snap["initial_stop_loss"] == 151. and snap["strategy_version"] == "old-strategy"


@pytest.mark.parametrize("horizon", [60, 180, 300])
@pytest.mark.parametrize("delay,known", [(0, True), (20, True), (21, False), (-1, False)])
def test_exact_lag_boundaries(horizon, delay, known):
    rows = build_observed_exit_rows([_observation(), _observation(horizon+delay, offset_pips=3)])
    assert (rows[0][f"observed_mark_return_{horizon}s_pips"] is not None) is known


def test_reused_ticket_with_changed_entry_and_stale_base_quote_do_not_mix():
    first, later = _observation(), _observation(60, offset_pips=4)
    later["snapshot"]["entry_price"] += .01
    rows = build_observed_exit_rows([first, later])
    assert rows[0]["observed_mark_return_60s_pips"] is None
    stale = _observation(quote_lag=21)
    assert build_observed_exit_rows([stale]) == []


def test_conflicting_duplicates_export_stably_and_hash_actual_bytes(tmp_path):
    first = _observation()
    changed = deepcopy(first)
    changed["snapshot"]["liquidation_price"] += .001
    later = _observation(60, offset_pips=2)
    a = build_observed_exit_rows([changed, first, later])
    b = build_observed_exit_rows([later, first, changed])
    assert a == b
    path = tmp_path / "rows.csv"
    meta = export_observed_exit_rows(a, path)
    assert meta["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("field", ["tick_size", "pip_size", "stop_distance", "freeze_distance"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_trailing_bad_broker_metadata_is_rejected(field, value):
    snap, pred = _sample()
    args = dict(advisory_released=True, quote_age_seconds=1, pip_size=.0001,
                tick_size=.00001, stop_distance=.00015, freeze_distance=.0001)
    args[field] = value
    assert not evaluate_exit_policy(snap, pred, ExitPolicySettings(), **args).eligible


@pytest.mark.parametrize("field", ["trailing_atr_multiplier", "minimum_stop_change_pips", "min_volume",
                                    "volume_step", "maximum_quote_age_seconds"])
def test_policy_rejects_nan_configuration(field):
    with pytest.raises(ValueError):
        ExitPolicySettings(**{field: float("nan")})


def test_gross_profit_and_unverified_partial_state_are_never_authority():
    snap, pred = _sample("TAKE_PROFIT_NOW")
    snap["estimated_net_pl"] = 1000
    assert not _evaluate(snap, pred).eligible
    snap, pred = _sample("REDUCE_POSITION")
    snap["pending_partial_close"] = True
    assert not _evaluate(snap, pred).eligible


@pytest.mark.parametrize("gross,eligible", [(1.0, True), (.999999, False), (float("nan"), False), (float("inf"), False)])
def test_verified_net_profit_exact_threshold(gross, eligible):
    snap, pred = _sample("TAKE_PROFIT_NOW")
    snap["net_profit_evidence"] = {"gross_liquidation_pl": gross, "swap": 0.,
        "paid_commission": 0., "closing_commission": 0., "slippage_cost": 0.,
        "costs_in_account_currency": True, "account_currency": "USD",
        "quote_timestamp": snap["quote_timestamp"], "liquidation_price": snap["bid"]}
    assert _evaluate(snap, pred).eligible is eligible


def test_prediction_contract_rejects_failed_decisions_and_mismatched_confidence():
    _, pred = _sample("HOLD")
    with pytest.raises(ValueError):
        replace(pred, status="error")
    with pytest.raises(ValueError):
        replace(pred, confidence=.1)


class FractionalClasses(FakeExitEstimator):
    classes_ = [.1, 1.1, 2.1, 3.1, 4.1]


class InvalidProbabilityEstimator(FakeExitEstimator):
    def predict_proba(self, frame):
        return [[float("nan"), .75, .1, .05, .05]]


@pytest.mark.parametrize("estimator", [FractionalClasses(), InvalidProbabilityEstimator()])
def test_invalid_model_classes_and_probabilities_fail_closed_with_attribution(tmp_path, estimator):
    now, trade, quote, instrument = _inputs()
    snap = build_exit_snapshot(trade, quote, instrument, now)
    model, meta = tmp_path / "model", tmp_path / "meta"
    joblib.dump(estimator, model)
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    meta.write_text(json.dumps({"target": "EXIT_ACTION", "model_version": "v", "model_sha256": digest,
        "feature_builder_version": EXIT_FEATURE_VERSION, "class_labels": list(EXIT_ACTIONS),
        "feature_columns": list(EXIT_FEATURE_COLUMNS)}))
    result = ExitPredictionService(model_path=str(model), metadata_path=str(meta)).predict(snap)
    assert result.status == "error" and result.decision is None
    assert result.model_version == "v" and result.model_sha256 == digest


def test_registry_rejects_cross_family_targets_and_corrupt_nested_records(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    model, meta = _candidate(tmp_path / "model", "v")
    with pytest.raises(RegistryError):
        registry.register(model_path=model, metadata_path=meta, model_type="exit_management")
    row = registry.register(model_path=model, metadata_path=meta, model_type="entry_quality")
    assert registry.register(model_path=model, metadata_path=meta, model_type="entry_quality") == row
    contents = json.loads(registry.manifest.read_text())
    contents["models"][row["model_id"]] = None
    registry.manifest.write_text(json.dumps(contents))
    with pytest.raises(RegistryError):
        registry.status()


def test_slow_optional_observer_is_bounded_and_submit_does_not_wait():
    started, release = Event(), Event()
    def slow(*args):
        started.set()
        release.wait(timeout=3)
    observer = ExitObservationQueue(slow, capacity=1)
    try:
        start = perf_counter()
        assert observer.submit("first", ())
        assert started.wait(timeout=1)
        assert observer.submit("second", ())
        assert not observer.submit("third", ())
        assert not observer.submit("first", ())
        assert perf_counter() - start < 1
    finally:
        observer.close()
        release.set()
        observer.thread.join(timeout=1)
    assert not observer.thread.is_alive()


def test_failure_throttle_and_model_failure_cannot_call_broker():
    now, trade, quote, instrument = _inputs()
    journal = Mock()
    journal.find_trade.side_effect = RuntimeError("DB offline")
    worker = SimpleNamespace(settings=FxBotSettings(), strategy_hash="s", code_version="c",
        journal=journal, client=Mock(), exit_predictor=Mock(), _exit_last_observed={})
    ForwardTestWorker._observe_exit_shadow(worker, now, trade, instrument, quote)
    ForwardTestWorker._observe_exit_shadow(worker, now+timedelta(seconds=1), trade, instrument, quote)
    assert journal.find_trade.call_count == 1
    assert worker.client.mock_calls == []
    worker.exit_predictor.predict.assert_not_called()


def test_non_demo_constructor_and_invalid_intervals_are_consistent():
    assert FxBotSettings(broker=BrokerSettings(demo_only=False)).exit_ai.mode == "off"
    for interval in [float("nan"), float("inf"), 0, 86401]:
        with pytest.raises(ValueError):
            ExitAiSettings(evaluation_interval_seconds=interval)


def test_exit_api_auth_bounds_corrupt_registry_and_nested_secret_redaction(tmp_path):
    from fastapi.testclient import TestClient
    from fxbot.api import create_app
    settings = FxBotSettings(runtime=RuntimeSettings(database_url=f"sqlite:///{tmp_path/'api.db'}",
        log_jsonl_path=None, api_key="audit-test-key", start_worker_with_api=False),
        exit_ai=ExitAiSettings(registry_path=str(tmp_path / "registry")))
    with TestClient(create_app(settings)) as client:
        headers = {"X-API-Key": settings.runtime.api_key}
        assert client.get("/api/exit-ai").status_code == 401
        assert client.get("/api/exit-ai?limit=101", headers=headers).status_code == 422
        journal = client.app.state.journal
        journal.log_event("exit_ai_observation", "secret-path", payload={
            "snapshot": {"position_id": "one", "internal": {"secret": "never-expose"}},
            "prediction": {"status": "error", "private_model_path": "never-expose"}})
        for _ in range(1001):
            journal.log_event("other", "heartbeat")
        result = client.get("/api/exit-ai?limit=1", headers=headers)
        assert result.status_code == 200
        assert len(result.json()["recent_observations"]) == 1
        assert "never-expose" not in result.text and "secret-path" not in result.text
        registry = ModelRegistry(settings.exit_ai.registry_path)
        registry.root.mkdir()
        registry.manifest.write_text('{"broken":true}')
        assert client.get("/api/exit-ai", headers=headers).json()["registry"]["status"] == "unavailable"


@pytest.mark.parametrize("mode", ["off", "shadow"])
@pytest.mark.parametrize("closing", [True, False])
def test_original_position_operations_precede_optional_capture(mode, closing, monkeypatch):
    now, trade, quote, instrument = _inputs()
    journal, broker = Mock(), Mock()
    quote.mid = (quote.bid+quote.ask)/2
    quote.quote_to_home_factor = None
    broker.open_trades.return_value = [trade, {**trade, "id": "second"}]
    worker = ForwardTestWorker(FxBotSettings(exit_ai=ExitAiSettings(mode=mode)), client=broker, journal=journal)
    trace = []
    worker._manage_sniper_trade = Mock(side_effect=lambda *args: trace.append("sniper") or closing)
    worker._maybe_move_stop_to_breakeven = Mock(side_effect=lambda *args: trace.append("breakeven"))
    worker._maybe_update_trailing_stop = Mock(side_effect=lambda *args: trace.append("trailing"))
    worker.exit_predictor = Mock(side_effect=AssertionError("synchronous inference"))
    monkeypatch.setattr("fxbot.forward.estimated_daily_financing_home", lambda *args, **kwargs: 0.)
    try:
        observations = worker._sync_open_trades(now, {instrument.name: instrument}, {instrument.name: quote}, {})
        assert trace == (["sniper", "sniper"] if closing else ["sniper", "breakeven", "trailing"]*2)
        assert len(observations) == (0 if closing or mode == "off" else 2)
        assert journal.upsert_trade.call_count == 2
        assert journal.record_external_order.call_count == 2
        worker.exit_predictor.predict.assert_not_called()
        broker.set_trade_dependent_orders.assert_not_called()
        broker.close_position.assert_not_called()
    finally:
        worker.close()


def test_scan_news_protection_completes_before_observation_dispatch(monkeypatch):
    from datetime import datetime, timezone
    from fxbot.instruments import FxInstrument, PriceSnapshot
    from fxbot.models import BotRunState
    now = datetime.now(timezone.utc)
    instrument = FxInstrument("EUR_USD")
    quote = PriceSnapshot("EUR_USD", bid=1.1, ask=1.1002, time=now)
    journal, broker = Mock(), Mock()
    journal.get_state.return_value.state = BotRunState.RUNNING.value
    broker.pricing.return_value = SimpleNamespace(prices={instrument.name: quote}, conversion_rates={})
    worker = ForwardTestWorker(FxBotSettings(), client=broker, journal=journal)
    trace = []
    worker._load_instruments = Mock(return_value={instrument.name: instrument})
    worker._check_price_freshness = Mock(return_value=False)
    worker._reconcile_unknown_orders = Mock()
    worker._sync_trade_history = Mock()
    worker._sync_open_trades = Mock(side_effect=lambda *args: trace.append("position_protection") or [({"id": "one"}, instrument, quote)])
    worker._protect_positions_for_news = Mock(side_effect=lambda *args: trace.append("news_protection"))
    worker.news.ensure_current = Mock(return_value=SimpleNamespace(events=[]))
    worker._exit_observer = Mock(closed=False)
    worker._exit_observer.submit.side_effect = lambda *args: trace.append("dispatch")
    try:
        worker.scan_once()
        assert trace == ["position_protection", "news_protection", "dispatch"]
        broker.create_market_order.assert_not_called()
    finally:
        worker.close()


def test_registry_failed_activation_preserves_champion_and_hides_nested_private_fields(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    a, am = _candidate(tmp_path / "a", "a")
    first = registry.register(model_path=a, metadata_path=am, model_type="entry_quality")
    registry.approve(first["model_id"], approver="reviewer", evidence="private-review-evidence")
    registry.activate(first["model_id"])
    b, bm = _candidate(tmp_path / "b", "b")
    second = registry.register(model_path=b, metadata_path=bm, model_type="entry_quality")
    registry.approve(second["model_id"], approver="reviewer", evidence="private-review-evidence")
    b.write_bytes(b"corrupt")
    with pytest.raises(RegistryError):
        registry.activate(second["model_id"])
    assert registry.status()["active"]["entry_quality"] == first["model_id"]
    contents = json.loads(registry.manifest.read_text())
    contents["models"][first["model_id"]]["internal"] = {"token": "nested-secret"}
    registry.manifest.write_text(json.dumps(contents))
    status = json.dumps(registry.status())
    assert "nested-secret" not in status and "private-review-evidence" not in status


def test_freeze_zone_and_future_atr_are_rejected():
    snap, pred = _sample()
    snap["stop_loss"] = snap["bid"] - .00001
    assert _evaluate(snap, pred).reason == "existing_stop_in_freeze_zone"
    snap, pred = _sample()
    snap["atr_timestamp"] = "2099-01-01T00:00:00Z"
    assert _evaluate(snap, pred).reason == "noncausal_trailing_indicator"


def test_cached_model_bytes_keep_original_version_when_disk_changes(tmp_path):
    now, trade, quote, instrument = _inputs()
    snap = build_exit_snapshot(trade, quote, instrument, now)
    model, meta = tmp_path / "model", tmp_path / "meta"
    joblib.dump(FakeExitEstimator(), model)
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    meta.write_text(json.dumps({"target": "EXIT_ACTION", "model_version": "original", "model_sha256": digest,
        "feature_builder_version": EXIT_FEATURE_VERSION, "class_labels": list(EXIT_ACTIONS),
        "feature_columns": list(EXIT_FEATURE_COLUMNS)}))
    service = ExitPredictionService(model_path=str(model), metadata_path=str(meta))
    assert service.predict(snap).status == "ok"
    model.write_bytes(b"corrupted or replaced")
    result = service.predict(snap)
    assert result.model_version == "original" and result.model_sha256 == digest
    restarted = ExitPredictionService(model_path=str(model), metadata_path=str(meta))
    assert restarted.predict(snap).status == "error"


def test_recovered_position_cannot_inherit_another_lifetimes_candidate():
    now, trade, quote, instrument = _inputs()
    journal = Mock()
    journal.find_trade.return_value = SimpleNamespace(entry_price=1.09,
        entry_time=trade["openTime"], strategy_hash="old", payload={
            "strategy_context": {"candidate_id": "old-candidate"}})
    worker = SimpleNamespace(settings=FxBotSettings(), strategy_hash="s", code_version="c",
        journal=journal, client=Mock(), exit_predictor=ExitPredictionService(), _exit_last_observed={})
    ForwardTestWorker._observe_exit_shadow(worker, now, trade, instrument, quote)
    snap = journal.log_event.call_args.kwargs["payload"]["snapshot"]
    assert snap["candidate_id"] is None and snap["strategy_version"] is None


def test_extreme_broker_metadata_and_missing_configured_model_fail_closed(tmp_path):
    snap, pred = _sample("REDUCE_POSITION")
    snap["broker_volume_step"] = 1e-308
    snap["broker_volume_lots"] = 1e308
    assert not _evaluate(snap, pred).eligible
    now, trade, quote, instrument = _inputs()
    snapshot = build_exit_snapshot(trade, quote, instrument, now)
    result = ExitPredictionService(model_path=str(tmp_path/"missing"), metadata_path=str(tmp_path/"metadata")).predict(snapshot)
    assert result.status == "unavailable" and result.decision is None
