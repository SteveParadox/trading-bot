"""Regression evidence for the 2026-10-05 AI/ML merge audit."""
from contextlib import closing
from dataclasses import replace
from datetime import timedelta
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from fxbot.ai.feature_builder import FEATURE_BUILDER_VERSION, build_prediction_features
from fxbot.ai.model_loader import VersionedModelLoader, ModelLoadError
from fxbot.ai.predictor import PredictionService
from fxbot.ai_deliberation import AiDeliberationResult, apply_ai_execution_policy
from fxbot.ai_evaluation import _realized_shadow_metrics
from fxbot.api import _config_payload
from fxbot.baseline_model import XGBoostBaselineConfig, train_xgboost_baseline, _target_ready
from fxbot.chronological_split import chronological_split
from fxbot.config import AiDeliberationSettings, FxBotSettings, MlPredictionSettings, RiskSettings, RuntimeSettings, StrategySettings
from fxbot.entry_timing_model import EntryTimingModelConfig, train_entry_timing_model
from fxbot.forward import ForwardTestWorker
from fxbot.instruments import FxInstrument, PriceSnapshot
from fxbot.journal import StructuredJournal
from fxbot.models import BotRunState, FxPortfolioState
from fxbot.news import NewsSnapshot
from fxbot.outcome_tracker import CandidateOutcomeTracker
from fxbot.risk import FxRiskManager
from fxbot.training_dataset import FEATURE_COLUMNS, ENTRY_ACTIONS, _feature_row, build_training_dataset
from test_fx_forward import CapturingDeliberator, FakeMt5Client, FIXED_NOW, FixedDatetime, trending_frame
from test_fx_prediction_service import _request, _FakeLoader, _EntryLoader, _TargetLoader, _ProbModel
from test_fx_ml_baseline import _row


@pytest.fixture
def worker(tmp_path):
    settings = FxBotSettings(instruments=["EUR_USD"],
        strategy=StrategySettings(partial_tp_enabled=False, trade_sessions_utc=(), avoid_rollover_minutes=0,
            require_volume_confirmation=False, min_atr_pips=.1, max_atr_pips=30, adx_min=10, htf_adx_min=10),
        risk=RiskSettings(max_pair_exposure_pct=10, max_gross_exposure_pct=10, max_currency_exposure_pct=10),
        runtime=RuntimeSettings(database_url=f"sqlite:///{tmp_path / 'audit.db'}", log_jsonl_path=None),
        ai=AiDeliberationSettings(mode="shadow", provider="none"))
    journal = StructuredJournal(settings.runtime.database_url)
    client = FakeMt5Client(entry_frame=trending_frame(1.08, .00025), htf_frame=trending_frame(1.06, .0005))
    result = ForwardTestWorker(settings, client=client, journal=journal, deliberator=CapturingDeliberator())
    result.news.ensure_current = Mock(return_value=NewsSnapshot(events=[], stale=False))
    journal.set_state(BotRunState.RUNNING)
    yield result
    result.close()


@pytest.mark.parametrize("decision", ["TAKE", "WAIT", "SKIP"])
def test_shadow_execution_and_full_candidate_pipeline(worker, decision):
    worker.deliberator = CapturingDeliberator(decision=decision)
    with patch("fxbot.forward.datetime", FixedDatetime):
        worker.scan_once()
    assert len(worker.client.created_orders) == 1
    candidate = worker.journal.recent_candidates()[0]
    assert candidate.executed
    assert worker.journal.find_ai_deliberation(candidate.candidate_id).decision == decision
    assert worker.journal.find_candidate_outcome(candidate.candidate_id)
    start = candidate.timestamp.replace(tzinfo=FIXED_NOW.tzinfo)
    instrument = FxInstrument("EUR_USD")
    # Full forward sampling, including both executable sides.
    for seconds in range(10, 1801, 10):
        t = start + timedelta(seconds=seconds)
        bid = candidate.entry + seconds * 0.000001
        worker.outcomes.observe(candidate=candidate, price=PriceSnapshot("EUR_USD", bid, bid+.0001, t), instrument=instrument, observed_at=t)
    frame = build_training_dataset(worker.journal)
    assert frame.candidate_id.tolist() == [candidate.candidate_id]
    from fxbot.ai_evaluation import ai_value_report
    assert ai_value_report(worker.journal)["sample_size"] == 1
    # Serving/training feature values agree, not just the column names.
    req = replace(_request(), candidate_id=candidate.candidate_id,
                  market_snapshot=candidate.payload["market_snapshot"],
                  strategy_signal=candidate.strategy_signal, strategy_score=candidate.payload["signal_score"],
                  execution_cost_pips_round_trip=candidate.payload["execution_cost_pips_round_trip"])
    served = build_prediction_features(req)
    for key in FEATURE_COLUMNS:
        actual = frame.iloc[0][key]
        assert (pd.isna(actual) and served[key] is None) or actual == served[key], key


@pytest.mark.parametrize("fault", ["quote", "spread", "news", "risk", "disconnect", "candle", "halt"])
def test_post_ai_revalidation_is_mandatory_without_sniper(worker, fault):
    original = worker.deliberator.deliberate
    def deliberate(evidence):
        response = original(evidence)
        if fault == "quote":
            worker.client.price = replace(worker.client.price, time=FIXED_NOW-timedelta(minutes=10))
        elif fault == "spread":
            worker.client.price = replace(worker.client.price, bid=worker.client.price.bid-.01)
        elif fault == "news":
            worker.settings = replace(worker.settings, strategy=replace(worker.settings.strategy, require_news_data=True))
            worker.news.ensure_current.return_value = NewsSnapshot(events=[], stale=True)
        elif fault == "risk":
            worker.client.account_summary = Mock(return_value={"NAV": 8000, "balance": 8000, "currency": "USD"})
        elif fault == "disconnect":
            worker.client.pricing = Mock(side_effect=RuntimeError("disconnected"))
        elif fault == "candle":
            worker.client.entry_frame.index -= pd.Timedelta(days=1)
        else:
            worker.journal.set_state(BotRunState.HALTED)
        return response
    worker.deliberator.deliberate = deliberate
    with patch("fxbot.forward.datetime", FixedDatetime):
        worker.scan_once()
    assert worker.client.created_orders == []
    assert worker.journal.recent_candidates()[0].rejection_reason == "execution_revalidation_failed"


@pytest.mark.parametrize("fault", ["provider", "lookup", "write", "prediction_write", "annotation"])
def test_shadow_optional_failures_do_not_block(worker, fault):
    if fault == "provider":
        worker.deliberator.deliberate = Mock(side_effect=TimeoutError("secret endpoint"))
    elif fault == "lookup":
        worker.journal.find_ai_deliberation = Mock(side_effect=RuntimeError("database busy"))
    elif fault == "write":
        worker.journal.record_ai_deliberation = Mock(side_effect=RuntimeError("database busy"))
    elif fault == "annotation":
        worker.journal.update_signal = Mock(side_effect=RuntimeError("database busy"))
    else:
        original = worker.journal.update_candidate
        def update(*args, **kwargs):
            if "numerical_prediction" in (kwargs.get("payload_update") or {}):
                raise RuntimeError("database busy")
            return original(*args, **kwargs)
        worker.journal.update_candidate = update
    with patch("fxbot.forward.datetime", FixedDatetime):
        worker.scan_once()
    assert len(worker.client.created_orders) == 1


def test_required_confirmation_cannot_fail_open():
    settings = AiDeliberationSettings(mode="advisory", advisory_require_confirmation=True, fail_policy="fail_open")
    assert not apply_ai_execution_policy(hard_safety_allowed=True, settings=settings,
        result=AiDeliberationResult(response=None, latency_ms=0, failure_reason="timeout")).allowed


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1., 1.5])
def test_invalid_auxiliary_probabilities_leave_primary_intact(bad):
    result = PredictionService(MlPredictionSettings(mode="shadow"), loader=_FakeLoader(),
        auxiliary_loaders={"FAKE_BREAKOUT": _TargetLoader("FAKE_BREAKOUT", bad, "v1")}).predict(_request())
    assert result.status == "ok" and result.tp_before_sl_probability == .83
    assert result.fake_breakout_probability is None
    assert "FAKE_BREAKOUT" in result.auxiliary_errors


def test_invalid_timing_distribution_leaves_primary_intact():
    loader = _EntryLoader()
    loader.artifact.model.predict_proba = lambda f: [[.9]*5]
    result = PredictionService(MlPredictionSettings(mode="shadow"), loader=_FakeLoader(), entry_loader=loader).predict(_request())
    assert result.status == "ok" and result.entry_action is None and result.entry_action_error


def test_artifact_missing_version_and_wrong_classes_fail_closed(tmp_path):
    model_path, metadata_path = tmp_path/"model.joblib", tmp_path/"meta.json"
    joblib.dump(_ProbModel(), model_path)
    metadata = {"target":"TP_BEFORE_SL", "feature_columns":FEATURE_COLUMNS,
                "model_sha256":hashlib.sha256(model_path.read_bytes()).hexdigest()}
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ModelLoadError, match="feature-builder"):
        VersionedModelLoader(model_path=model_path, metadata_path=metadata_path).load()
    loader = _FakeLoader()
    loader.artifact.model.classes_ = [1,0]
    assert PredictionService(MlPredictionSettings(mode="shadow"), loader=loader).predict(_request()).status == "error"


def test_candidate_snapshot_is_immutable(worker):
    with patch("fxbot.forward.datetime", FixedDatetime):
        worker.scan_once()
    candidate = worker.journal.recent_candidates()[0]
    before = _feature_row(candidate, worker.journal.find_candidate_outcome(candidate.candidate_id))
    worker.journal.update_candidate(candidate.candidate_id, stop_loss=.1, take_profit=99,
        payload_update={"market_snapshot":{"bid":999}, "execution_cost_pips_round_trip":999})
    after = _feature_row(worker.journal.find_candidate(candidate.candidate_id), worker.journal.find_candidate_outcome(candidate.candidate_id))
    assert all(before[k] == after[k] for k in FEATURE_COLUMNS)


def test_purge_duplicate_ids_and_longer_label_horizons():
    frame = pd.DataFrame([_row("2024-12-31T23:00:00Z", "late-label", 1)])
    frame["label_end_timestamp"] = "2025-01-01T00:01:00Z"
    assert chronological_split(frame).train.empty
    with pytest.raises(ValueError, match="duplicate"):
        chronological_split(pd.concat([frame,frame]))


def test_timing_minimum_support_and_nullable_binary_target(tmp_path):
    rows=[]
    for period, year in [("train",2024),("validation",2025),("test",2026)]:
        for i, action in enumerate(ENTRY_ACTIONS):
            row=_row(f"{year}-01-{i+1:02d}T12:00:00Z", f"{period}-{i}", i%2)
            row["ENTRY_ACTION_LABEL"]=action
            rows.append(row)
    with pytest.raises(ValueError, match="per-class"):
        train_entry_timing_model(pd.DataFrame(rows),tmp_path)
    result=_target_ready(pd.DataFrame({"FAKE_BREAKOUT":[None,0,1,np.nan]}),"FAKE_BREAKOUT")
    assert result.FAKE_BREAKOUT.tolist()==[0,1]


def test_real_model_handles_missing_categories_at_fit_and_serve(tmp_path):
    rows=[]
    for year in [2024,2025,2026]:
        for i in range(4):
            row=_row(f"{year}-02-{i+1:02d}T12:00:00Z",f"{year}-{i}",i%2)
            row["upcoming_news_currency"]=None
            row["free_margin"]=None
            rows.append(row)
    artifacts=train_xgboost_baseline(pd.DataFrame(rows),tmp_path,model_config=XGBoostBaselineConfig(n_estimators=3))
    loader=VersionedModelLoader(model_path=artifacts.model_path,metadata_path=artifacts.metadata_path)
    result=PredictionService(MlPredictionSettings(mode="shadow"),loader=loader).predict(_request())
    assert result.successful
    assert loader.load() is loader.load()


def test_config_redacts_database_credentials():
    payload=_config_payload(FxBotSettings(runtime=RuntimeSettings(database_url="postgresql://user:SECRET@host/db", log_jsonl_path="/private/log")))
    assert "SECRET" not in json.dumps(payload) and "/private/log" not in json.dumps(payload)


def test_realized_pnl_aggregates_closed_legs_idempotently(worker):
    for trade_id, pnl in [("leg1",2.),("leg2",3.)]:
        worker.journal.upsert_trade(broker_trade_id=trade_id,instrument="EUR_USD",side="LONG",units=1000,state="open",
            payload={"strategy_context":{"candidate_id":"split", "account_currency":"USD"}})
    worker.journal.upsert_trade(broker_trade_id="leg1",instrument="EUR_USD",side="LONG",units=1000,state="closed",realized_pl=2.)
    assert worker.journal.candidate_realized_pnl("split")["final_net_pnl"] is None
    for _ in range(2):
        worker.journal.upsert_trade(broker_trade_id="leg2",instrument="EUR_USD",side="LONG",units=1000,state="closed",realized_pl=3.,financing=-.1)
    assert worker.journal.candidate_realized_pnl("split")["final_net_pnl"] == pytest.approx(4.9)


def test_unknown_pnl_currency_is_not_summed():
    frame=pd.DataFrame([{"executed":True,"final_net_pnl":5.,"final_net_pnl_currency":None}])
    assert not _realized_shadow_metrics(frame)["comparable"]


def test_hedged_pair_exposure_and_external_stop_risk_are_counted(worker):
    inst=FxInstrument("EUR_USD")
    price=worker.client.price
    worker.client.open_trades=Mock(return_value=[{"instrument":"EUR_USD", "currentUnits":10000,
        "stopLossOrder":{"price":price.bid-.002}}])
    portfolio=worker._portfolio_from_broker(FIXED_NOW,{"EUR_USD":inst},{"EUR_USD":price},{},
        account=worker.client.account_summary(),positions=[{"instrument":"EUR_USD", "long":{"units":10000},"short":{"units":-10000}}])
    assert portfolio.gross_exposure == pytest.approx(20000*price.mid)
    assert portfolio.pair_exposures["EUR_USD"] > 0
    assert portfolio.currency_exposures["USD"] == 0
    assert portfolio.portfolio_risk >= 20 - 1e-8
    worker.client.open_trades.return_value[0]["stopLossOrder"]=None
    from fxbot.mt5 import Mt5Error
    with pytest.raises(Mt5Error,match="stop risk"):
        worker._portfolio_from_broker(FIXED_NOW,{"EUR_USD":inst},{"EUR_USD":price},{}, account=worker.client.account_summary(),positions=[])


def test_demo_account_switch_and_unknown_account_mode_are_rejected():
    from fxbot.mt5 import Mt5Client, Mt5CredentialsMissing
    from fxbot.config import BrokerSettings
    client=Mt5Client(BrokerSettings(), module=SimpleNamespace())
    for mode in [2,-1]:
        with pytest.raises(Mt5CredentialsMissing):
            client._assert_demo_account(SimpleNamespace(trade_mode=mode))


def test_strict_parser_rejects_duplicate_properties_and_live_legacy_codes():
    from fxbot.ai_deliberation import _extract_provider_json, validate_ai_audit_response, AiResponseValidationError
    with pytest.raises(AiResponseValidationError, match="duplicate"):
        _extract_provider_json({"output_text":'{"decision":"SKIP","decision":"TAKE","confidence":0.8,"reason_codes":["trend_alignment"]}'})
    with pytest.raises(AiResponseValidationError):
        validate_ai_audit_response({"decision":"TAKE","confidence":.8,"reason_codes":["legacy_confirm"]})


def test_outcome_restart_backlog_progress_and_late_tick_no_path_labels(worker):
    start=FIXED_NOW-timedelta(hours=2)
    for i in range(3):
        worker.journal.record_candidate(candidate_id=f"old-{i}",timestamp=start,symbol="EUR_USD",direction="LONG",
            entry=1.1,spread=.0001,strategy_signal="signal_confirmed")
    tracker=CandidateOutcomeTracker(worker.journal)
    tracker.candidate_scan_limit=1
    for _ in range(3):
        tracker.observe_active(prices={},instruments={},observed_at=FIXED_NOW)
    assert worker.journal.pending_outcome_candidates()==[]
    row,_=worker.journal.record_candidate(candidate_id="late",timestamp=start,symbol="EUR_USD",direction="LONG",
        entry=1.1,spread=.0001,stop_loss=1.0,take_profit=1.2,strategy_signal="signal_confirmed")
    tracker.seed(candidate=row,price=PriceSnapshot("EUR_USD",1.3,1.3001,FIXED_NOW),instrument=FxInstrument("EUR_USD"),observed_at=FIXED_NOW)
    outcome=worker.journal.find_candidate_outcome("late")
    assert outcome.status=="incomplete" and not outcome.tp_hit and outcome.mfe_pips==0


def test_additive_sqlite_migration_preserves_historical_rows(tmp_path):
    import sqlite3
    path=tmp_path/"migration.db"
    with closing(StructuredJournal(f"sqlite:///{path}")) as journal:
        journal.record_candidate(candidate_id="preserved",timestamp=FIXED_NOW,symbol="EUR_USD",direction="LONG",entry=1.1,spread=.0001,strategy_signal="s")
    with sqlite3.connect(path) as connection:
        connection.execute("ALTER TABLE candidate_outcomes DROP COLUMN final_net_pnl_at")
    with closing(StructuredJournal(f"sqlite:///{path}")) as journal:
        assert journal.find_candidate("preserved") is not None
        journal.ensure_candidate_outcome(candidate_id="preserved",started_at=FIXED_NOW)
        assert journal.find_candidate_outcome("preserved").final_net_pnl_at is None


def test_sqlite_retry_only_retries_lock_errors():
    from fxbot.database import sqlite_retry_operation
    from sqlalchemy.exc import OperationalError
    operation=Mock(side_effect=[OperationalError("update",{},Exception("database is locked")),"ok"])
    with patch("fxbot.database.time.sleep"):
        assert sqlite_retry_operation(operation)=="ok"
    assert operation.call_count==2


def test_research_models_cannot_be_configured_on_live_accounts():
    from fxbot.config import BrokerSettings
    with pytest.raises(ValueError,match="demo/research-only"):
        FxBotSettings(broker=BrokerSettings(demo_only=False), ml_prediction=MlPredictionSettings(mode="shadow"))


def test_research_api_auth_validation_and_no_state_mutation(tmp_path):
    from fastapi.testclient import TestClient
    from fxbot.api import create_app
    settings=FxBotSettings(runtime=RuntimeSettings(database_url=f"sqlite:///{tmp_path/'api.db'}",log_jsonl_path=None,
        api_key="audit-test-key-not-a-real-secret",start_worker_with_api=False))
    with TestClient(create_app(settings)) as client:
        headers={"X-API-Key":settings.runtime.api_key}
        for path in ["/api/ai-evaluation","/api/ai-deliberations","/api/config","/api/performance"]:
            assert client.get(path).status_code==401
            assert client.get(path,headers=headers).status_code==200
        assert client.get("/api/ai-evaluation?limit=-1",headers=headers).status_code==422
        assert client.get("/api/ai-evaluation?min_wait_improvement_pips=nan",headers=headers).status_code==422
        assert client.app.state.journal.get_state().state==BotRunState.STOPPED.value


def test_news_context_considers_all_nearby_events_and_exact_release():
    from fxbot.config import NewsEvent
    from fxbot.market_snapshot import build_news_context
    events=[NewsEvent("Low", "USD", "low", FIXED_NOW+timedelta(minutes=1),FIXED_NOW+timedelta(minutes=2)),
            NewsEvent("High", "USD", "high", FIXED_NOW+timedelta(minutes=2),FIXED_NOW+timedelta(minutes=3)),
            NewsEvent("Now", "EUR", "high", FIXED_NOW,FIXED_NOW+timedelta(minutes=1))]
    context=build_news_context(symbol="EUR_USD",events=events,observed_at=FIXED_NOW,stale=False,age_seconds=0,
                              last_updated=FIXED_NOW,source="fixture",before_minutes=30,after_minutes=30)
    assert context.risk_level=="HIGH" and context.event_just_occurred
    assert context.recent_event.minutes_since_event==0
