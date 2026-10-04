from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from fxbot.ai_contract import AiCandidateEvaluator, AiTradeDecision, validate_ai_trade_recommendation
from fxbot.instruments import FxInstrument, PriceSnapshot
from fxbot.market_snapshot import MARKET_SNAPSHOT_VERSION, build_market_snapshot
from fxbot.models import FxPortfolioState, FxSignalIntent, Side
from fxbot.risk import FxExitPlan


NOW = datetime(2026, 10, 4, 12, 7, tzinfo=timezone.utc)


def test_ai_job_is_strictly_take_wait_skip() -> None:
    assert [decision.value for decision in AiTradeDecision] == ["TAKE", "WAIT", "SKIP"]
    recommendation = validate_ai_trade_recommendation(
        {
            "decision": "WAIT",
            "confidence": 0.72,
            "reason_codes": ["pullback_risk"],
            "warnings": ["spread_elevated"],
        }
    )
    assert recommendation.decision is AiTradeDecision.WAIT
    assert recommendation.to_dict()["decision"] == "WAIT"


def test_candidate_evaluator_only_returns_structured_recommendation() -> None:
    seen = {}
    evaluator = AiCandidateEvaluator(
        lambda snapshot: (
            seen.update(snapshot)
            or {
                "decision": "TAKE",
                "confidence": 0.81,
                "reason_codes": ["trend_alignment"],
                "warnings": [],
            }
        )
    )
    result = evaluator.evaluate({"candidate_id": "fxsig-test", "symbol": "EUR_USD"})
    assert seen["candidate_id"] == "fxsig-test"
    assert result.decision is AiTradeDecision.TAKE


def test_ai_contract_rejects_trade_generation_or_risk_fields() -> None:
    with pytest.raises(ValueError, match="TAKE, WAIT, or SKIP"):
        validate_ai_trade_recommendation(
            {"decision": "BUY", "confidence": 0.9, "reason_codes": [], "warnings": []}
        )
    with pytest.raises(ValueError, match="unsupported fields"):
        validate_ai_trade_recommendation(
            {
                "decision": "TAKE",
                "confidence": 0.9,
                "reason_codes": [],
                "warnings": [],
                "position_size": 2000,
            }
        )


def test_market_snapshot_contains_causal_candidate_state() -> None:
    frame = _frame()
    instrument = FxInstrument("EUR_USD")
    intent = FxSignalIntent(
        instrument="EUR_USD",
        side=Side.LONG,
        timestamp=NOW - timedelta(minutes=7),
        entry_price=1.1001,
        signal_row={"atr": 0.001, "adx": 31.0},
    )
    portfolio = FxPortfolioState(
        equity=10_000,
        balance=10_000,
        margin_used=250,
        open_positions=2,
        portfolio_risk=120,
        gross_exposure=3_500,
        pair_exposures={"EUR_USD": 1_100},
        currency_exposures={"EUR": 1_100, "USD": -1_100},
    )
    snapshot = build_market_snapshot(
        candidate_id="fxsig-EURUSD-test",
        intent=intent,
        instrument=instrument,
        price=PriceSnapshot("EUR_USD", bid=1.1000, ask=1.1002, time=NOW),
        entry_frame=frame,
        timeframe="15m",
        portfolio=portfolio,
        exit_plan=FxExitPlan(
            stop_loss=1.0981,
            take_profit=1.1041,
            risk_distance=0.002,
            reward_distance=0.004,
            risk_reward=2.0,
            stop_pips=20,
            take_profit_pips=40,
        ),
        observed_at=NOW,
        sessions={"london", "new_york"},
    )

    payload = snapshot.to_dict()
    assert payload["version"] == MARKET_SNAPSHOT_VERSION
    assert payload["candidate_id"] == "fxsig-EURUSD-test"
    assert payload["symbol"] == "EUR_USD"
    assert payload["direction"] == "LONG"
    assert payload["bid"] == 1.1000
    assert payload["ask"] == 1.1002
    assert payload["spread"] == pytest.approx(0.0002)
    assert payload["spread_pips"] == pytest.approx(2.0)
    assert len(payload["recent_candles"]) == 20
    assert payload["atr"] == pytest.approx(0.001)
    assert 0 <= payload["rsi"] <= 100
    assert payload["momentum"] is not None
    assert payload["trend_strength"] == 31.0
    assert payload["support_distance_pips"] is not None
    assert payload["resistance_distance_pips"] is not None
    assert payload["session"] == ["london", "new_york"]
    assert payload["volatility"] == pytest.approx(1.0)
    assert payload["proposed_entry"] == 1.1001
    assert payload["stop_loss"] == 1.0981
    assert payload["take_profit"] == 1.1041
    assert payload["risk_reward"] == 2.0
    assert payload["current_exposure"]["open_positions"] == 2
    assert payload["current_exposure"]["portfolio_risk"] == 120
    assert payload["current_exposure"]["gross_exposure"] == 3_500
    assert payload["current_exposure"]["pair_exposure"] == 1_100


def test_market_snapshot_excludes_forming_candle() -> None:
    frame = _frame()
    forming_index = frame.index[-1]
    frame.loc[forming_index, ["open", "high", "low", "close"]] = [9.0, 10.0, 0.1, 9.5]
    snapshot = build_market_snapshot(
        candidate_id="fxsig-EURUSD-forming",
        intent=FxSignalIntent(
            instrument="EUR_USD",
            side=Side.LONG,
            timestamp=NOW - timedelta(minutes=7),
            entry_price=1.1001,
            signal_row={"atr": 0.001, "adx": 31.0},
        ),
        instrument=FxInstrument("EUR_USD"),
        price=PriceSnapshot("EUR_USD", bid=1.1000, ask=1.1002, time=NOW),
        entry_frame=frame,
        timeframe="15m",
        portfolio=FxPortfolioState(10_000, 10_000, 0, 0),
        exit_plan=None,
        observed_at=NOW,
        sessions=set(),
    )
    assert all(candle.timestamp < forming_index.to_pydatetime() for candle in snapshot.recent_candles)
    assert max(candle.high for candle in snapshot.recent_candles) < 2.0


def _frame() -> pd.DataFrame:
    # The final row opens at 12:00 and is still forming at NOW=12:07.
    index = pd.date_range(end="2026-10-04T12:00:00Z", periods=32, freq="15min")
    closes = [
        1.1000 + ((i % 5) - 2) * 0.0002 + i * 0.00001
        for i in range(len(index))
    ]
    return pd.DataFrame(
        {
            "open": [close - 0.00005 for close in closes],
            "high": [close + 0.0004 for close in closes],
            "low": [close - 0.0004 for close in closes],
            "close": closes,
            "volume": [100 + i for i in range(len(index))],
            "atr": [0.001 for _ in index],
            "adx": [31.0 for _ in index],
        },
        index=index,
    )
