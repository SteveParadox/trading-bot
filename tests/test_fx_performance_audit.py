"""Regression evidence for the M15 performance audit (no profitability claims)."""
from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

import indicators
from fxbot.config import StrategySettings, settings_from_env
from fxbot.strategy import prepare_indicators, evaluate_signal_frame, CandleFrameError
from fxbot.historical_reconstruction import replay_indicator_window
from fxbot.instruments import FxInstrument
from fxbot.performance import OperationTimings, measured, operation_timings
from fxbot.ai.predictor import PredictionService
from fxbot.config import MlPredictionSettings
from test_fx_prediction_service import _FakeLoader, _EntryLoader, _TargetLoader, _request
import test_fx_multi_timeframe as mtf
from test_fx_multi_timeframe import trend


class M15ConfirmationTests(mtf.MultiTimeframeTests):
    """Run all agreement, expiry, conflict and order-leg tests on intended M15."""
    def setUp(self):
        super().setUp()
        self.settings = replace(self.settings, entry_timeframe="15m", tp_timeframe="5m")
        self.entry = trend("15min", "2026-01-08T11:45:00Z", .00008)



def test_clean_environment_defaults_preserve_m15_and_demo_safety():
    with patch.dict("os.environ", {}, clear=True):
        settings = settings_from_env()
    assert (settings.strategy.entry_timeframe, settings.strategy.structure_timeframe, settings.strategy.htf_timeframe) == ("15m", "1h", "4h")
    assert settings.broker.demo_only and not settings.runtime.live_trading_enabled
    assert settings.strategy.tp_timeframe is None  # opt-in exit intelligence


@pytest.mark.parametrize("size", [0, 1, 14, 15, 60, 180, 1000])
@pytest.mark.parametrize("period", [0, 1, 14, 28])
def test_wilder_array_recurrence_matches_original(size, period):
    values = pd.Series(np.random.default_rng(42).uniform(0, 5, size), name="test")
    expected = pd.Series(np.nan, index=values.index, dtype=float)
    if period > 0 and size > period:
        expected.iloc[period] = values.iloc[1:period + 1].sum()
        for i in range(period + 1, size):
            previous = expected.iloc[i - 1]
            expected.iloc[i] = previous - previous / period + values.iloc[i]
    pd.testing.assert_series_equal(indicators._wilders_smooth(values, period), expected, check_exact=True)


@pytest.mark.parametrize("position", [1, 14, 40, 100])
def test_wilder_missing_values_keep_original_propagation(position):
    values = pd.Series(np.arange(180, dtype=float))
    values.iloc[position] = np.nan
    result = indicators._wilders_smooth(values, 14)
    assert result.iloc[max(position, 14):].isna().all()


def test_replay_uses_live_completed_window_and_same_recursive_seed():
    frame = trend("15min", "2026-01-08T12:15:00Z", .00008)
    # Vary volatility to make recursive seed differences observable.
    frame["high"] += np.random.default_rng(7).uniform(0, .005, len(frame))
    now = datetime(2026, 1, 8, 12, 0, tzinfo=timezone.utc)
    expected_raw = frame.loc[frame.index + pd.Timedelta(minutes=15) <= pd.Timestamp(now)].tail(90)
    expected = prepare_indicators(expected_raw)
    actual = replay_indicator_window(frame, "15m", now, 90)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual.index[-1] == pd.Timestamp("2026-01-08T11:45:00Z")
    assert len(actual) == 90


@pytest.mark.parametrize("column", ["open", "high", "low", "close", "volume"])
def test_infinite_raw_market_values_are_rejected(column):
    frame = trend("15min", "2026-01-08T11:45:00Z", .00008)
    frame.loc[frame.index[-1], column] = np.inf
    with pytest.raises(CandleFrameError, match="nonfinite_ohlcv"):
        prepare_indicators(frame)


@pytest.mark.parametrize("column", ["adx", "di_plus", "di_minus", "ma7", "ma14", "ma28", "close"])
def test_nonfinite_precomputed_indicators_cannot_confirm_direction(column):
    entry = prepare_indicators(trend("15min", "2026-01-08T11:45:00Z", .00008))
    entry.loc[entry.index[-1], column] = np.inf
    decision = evaluate_signal_frame(entry, entry, instrument=FxInstrument("EUR_USD"), settings=StrategySettings())
    assert decision.signal is None and decision.reason == "indicator_unavailable"


CONTEXT = {"entry_timeframe": "15m", "structure_timeframe": "1h", "htf_timeframe": "4h", "tp_timeframe": "5m"}


@pytest.mark.parametrize("manifest", [None, {**CONTEXT, "htf_timeframe": "1h"}, {**CONTEXT, "entry_timeframe": "5m"}, {**CONTEXT, "tp_timeframe": None}])
def test_models_must_explicitly_match_active_strategy(manifest):
    loader = _FakeLoader()
    loader.artifact.metadata["validated_strategy_context"] = manifest
    service = PredictionService(MlPredictionSettings(mode="shadow"), loader=loader)
    result = service.predict(replace(_request(), strategy_context=CONTEXT))
    assert result.status == "unavailable" and result.error == "model_strategy_incompatible"
    assert result.tp_before_sl_probability is None


@pytest.mark.parametrize("other", ["primary", "entry", "auxiliary"])
def test_every_contributing_model_requires_validation(other):
    loader, entry, aux = _FakeLoader(), _EntryLoader(), _TargetLoader("FAKE_BREAKOUT", .2, "v1")
    for item in (loader, entry, aux):
        item.artifact.metadata["validated_strategy_context"] = CONTEXT
    service = PredictionService(MlPredictionSettings(mode="shadow"), loader=loader,
                                entry_loader=entry, auxiliary_loaders={"FAKE_BREAKOUT": aux})
    request = replace(_request(), strategy_context=CONTEXT)
    assert service.predict(request).successful
    {"primary": loader, "entry": entry, "auxiliary": aux}[other].artifact.metadata.pop("validated_strategy_context")
    assert service.predict(request).error == "model_strategy_incompatible"


def test_metrics_are_bounded_thread_safe_and_preserve_exceptions():
    timings = OperationTimings(capacity=5, max_operations=2)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: timings.observe("test", 3), range(100)))
    timings.observe("other", 1, True)
    timings.observe("ignored", 1)
    snapshot = timings.snapshot()
    assert snapshot["test"]["count"] == 100 and snapshot["test"]["window_size"] == 5
    assert snapshot["other"]["failures"] == 1 and "ignored" not in snapshot
    @measured("test_failure")
    def fail():
        raise RuntimeError("preserved")
    with pytest.raises(RuntimeError, match="preserved"):
        fail()
    assert operation_timings.snapshot()["test_failure"]["failures"] > 0


@pytest.mark.parametrize("kind", ["missing", "stale", "invalid_atr"])
def test_replay_blocks_unavailable_configured_tp_data(tmp_path, kind):
    from fxbot.config import FxBotSettings
    from fxbot.journal import StructuredJournal
    from fxbot.historical_reconstruction import HistoricalReconstructionConfig, reconstruct_historical_candidates
    from test_fx_ml_baseline import _HistoricalSource
    source = _HistoricalSource()
    original = source.historical_candles
    def candles(instrument, timeframe, start, end):
        if timeframe != "5m":
            return original(instrument, timeframe, start, end)
        if kind == "missing":
            return pd.DataFrame(columns=source.entry.columns)
        last = "2023-12-31T10:00:00Z" if kind == "stale" else "2024-01-01T00:25:00Z"
        result = trend("5min", last, .00008)
        if kind == "invalid_atr":
            result[["open", "high", "low", "close"]] = 1.1
        return result
    source.historical_candles = candles
    strategy = StrategySettings(tp_timeframe="5m", require_volume_confirmation=False,
        min_atr_pips=.1, max_atr_pips=30, adx_min=10, htf_adx_min=10,
        trade_sessions_utc=(), avoid_rollover_minutes=0)
    with_journal = StructuredJournal(f"sqlite:///{tmp_path}/tp.db")
    try:
        report = reconstruct_historical_candidates(source=source, settings=FxBotSettings(strategy=strategy),
            journal=with_journal, config=HistoricalReconstructionConfig(
                start=datetime(2024,1,1,tzinfo=timezone.utc), end=datetime(2024,1,1,0,30,tzinfo=timezone.utc)),
            symbols=["EUR_USD"])
        stats = report["symbols"]["EUR_USD"]
        assert stats["strategy_candidates"] > 0
        assert stats["tp_data_blocks"] == stats["candidates_with_entry_quotes"]
        candidates = with_journal.recent_candidates()
        assert len(candidates) == stats["tp_data_blocks"]
        assert all(row.rejection_reason == "historical_tp_data_unavailable" and not row.executed for row in candidates)
        assert not with_journal.recent_candidate_outcomes()
    finally:
        with_journal.close()


def test_monitoring_api_exposes_timings_and_actual_configuration(tmp_path):
    from fastapi.testclient import TestClient
    from fxbot.api import create_app
    from fxbot.config import FxBotSettings, RuntimeSettings
    settings = FxBotSettings(strategy=StrategySettings(entry_timeframe="15m", htf_timeframe="4h", structure_timeframe="1h", tp_timeframe="5m"),
        runtime=RuntimeSettings(database_url=f"sqlite:///{tmp_path}/api.db", log_jsonl_path=None,
            start_worker_with_api=False, api_key="local-test-key"))
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/monitoring").status_code == 401
        headers = {"X-API-Key": "local-test-key"}
        result = client.get("/api/monitoring", headers=headers)
        assert result.status_code == 200
        payload = result.json()["performance"]
        assert payload["process_cpu_seconds"] >= 0
        assert payload["sample_capacity"] == 256
        strategy = client.get("/api/config", headers=headers).json()["strategy"]
        assert [strategy[key] for key in ("entry_timeframe", "structure_timeframe", "htf_timeframe", "tp_timeframe")] == ["15m", "1h", "4h", "5m"]


def test_equal_entry_and_bias_timeframes_are_rejected():
    with pytest.raises(ValueError, match="shorter than HTF"):
        StrategySettings(entry_timeframe="1h", htf_timeframe="1h")


@pytest.mark.parametrize("getter", ["positions_get", "orders_get"])
def test_failed_broker_lookup_is_never_interpreted_as_no_existing_order(getter):
    from types import SimpleNamespace
    from fxbot.mt5 import Mt5Client, Mt5Error
    from fxbot.config import BrokerSettings
    module = SimpleNamespace(positions_get=lambda: (), orders_get=lambda: (), last_error=lambda: (-1, "disconnected"))
    setattr(module, getter, lambda: None)
    client = Mt5Client(BrokerSettings(), module=module)
    client._ensure_connected = lambda: None
    with pytest.raises(Mt5Error, match="duplicate-order lookup"):
        client.order_by_client_id("same-candidate")
    assert not client._connected


def test_provenance_includes_shared_indicator_sources(tmp_path, monkeypatch):
    from fxbot.security import code_version
    monkeypatch.delenv("FX_CODE_VERSION", raising=False)
    (tmp_path / "fxbot").mkdir()
    (tmp_path / "fxbot" / "strategy.py").write_text("strategy-v1")
    shared = tmp_path / "indicators.py"
    shared.write_text("indicators-v1")
    before = code_version(tmp_path)
    shared.write_text("indicators-v2")
    assert code_version(tmp_path) != before


def test_slow_websocket_snapshot_does_not_block_control_api(tmp_path, monkeypatch):
    from threading import Event
    from fastapi.testclient import TestClient
    from fxbot.api import create_app
    from fxbot.config import FxBotSettings, RuntimeSettings
    entered, release = Event(), Event()
    def slow_snapshot(*args):
        entered.set()
        release.wait(3)
        return {"snapshot": "ready"}
    monkeypatch.setattr("fxbot.api.live_snapshot", slow_snapshot)
    settings = FxBotSettings(runtime=RuntimeSettings(database_url=f"sqlite:///{tmp_path}/slow.db",
        log_jsonl_path=None, start_worker_with_api=False, api_key="local-test-key"))
    with TestClient(create_app(settings)) as client:
        with client.websocket_connect("/ws/live?api_key=local-test-key") as websocket:
            try:
                assert entered.wait(2)
                with ThreadPoolExecutor(max_workers=1) as pool:
                    response = pool.submit(client.get, "/api/config", headers={"X-API-Key": "local-test-key"}).result(timeout=2)
                    assert response.status_code == 200 and not release.is_set()
            finally:
                release.set()
            assert websocket.receive_json() == {"snapshot": "ready"}


@pytest.mark.parametrize("period", [1, 14, 28])
@pytest.mark.parametrize("kind", ["random", "flat", "short"])
def test_adx_array_loop_matches_original_pandas_recurrence(period, kind):
    size = 12 if kind == "short" else 180
    rng = np.random.default_rng(28)
    index = pd.date_range("2026-01-01", periods=size, freq="15min", tz="UTC")
    tr = pd.Series(np.zeros(size) if kind == "flat" else rng.uniform(.1, 2, size), index=index)
    plus = pd.Series(rng.uniform(0, .1, size), index=index)
    minus = pd.Series(rng.uniform(0, .1, size), index=index)
    safe = indicators._wilders_smooth(tr, period).replace(0, np.nan)
    expected_plus = 100 * indicators._wilders_smooth(plus, period) / safe
    expected_minus = 100 * indicators._wilders_smooth(minus, period) / safe
    dx = 100 * (expected_plus - expected_minus).abs() / (expected_plus + expected_minus).replace(0, np.nan)
    expected_adx = pd.Series(np.nan, index=index, dtype=float)
    first = 2 * period - 1
    if size > first and not dx.iloc[period:first+1].isna().any():
        expected_adx.iloc[first] = dx.iloc[period:first+1].mean()
        for i in range(first+1, size):
            current, previous = dx.iloc[i], expected_adx.iloc[i-1]
            if not pd.isna(current) and not pd.isna(previous):
                expected_adx.iloc[i] = (previous * (period-1) + current) / period
    actual = indicators._calculate_wilder_adx(tr, plus, minus, period, index)
    for result, expected in zip(actual, (expected_plus, expected_minus, expected_adx)):
        pd.testing.assert_series_equal(result, expected, check_exact=True)
