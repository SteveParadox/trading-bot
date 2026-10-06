"""Build serving-time model features from an immutable candidate snapshot."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from fxbot.ai.schemas import PredictionRequest
from fxbot.training_dataset import FEATURE_COLUMNS


FEATURE_BUILDER_VERSION = "v2"


def build_prediction_features(request: PredictionRequest) -> dict[str, Any]:
    """Return exactly the feature manifest used by baseline model training."""

    snapshot = request.market_snapshot
    if snapshot.get("version") != "v1":
        raise ValueError("unsupported market snapshot version")
    snapshot_candidate_id = str(snapshot.get("candidate_id") or "")
    if snapshot_candidate_id and snapshot_candidate_id != request.candidate_id:
        raise ValueError("market snapshot candidate_id does not match prediction request")

    symbol = str(snapshot.get("symbol") or request.candidate_trade.get("symbol") or "").upper()
    direction = str(snapshot.get("direction") or request.candidate_trade.get("direction") or "").upper()
    if not symbol or not direction:
        raise ValueError("market snapshot requires symbol and direction")

    timestamp = pd.Timestamp(snapshot.get("timestamp"))
    if pd.isna(timestamp):
        raise ValueError("market snapshot timestamp is invalid")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    exposure = snapshot.get("current_exposure") or {}
    news = snapshot.get("news_context") or {}
    upcoming = news.get("upcoming_event") or {}
    recent = news.get("recent_event") or {}
    freshness = news.get("freshness") or {}
    atr = _finite(snapshot.get("atr"))
    spread = _finite(snapshot.get("spread"))
    spread_pips = _finite(snapshot.get("spread_pips"))
    rr = _finite(snapshot.get("risk_reward"))

    features = {
        "symbol": symbol,
        "direction": direction,
        "strategy_signal": str(request.strategy_signal),
        "strategy_score": float(request.strategy_score),
        "hour_utc": int(timestamp.hour),
        "day_of_week": int(timestamp.weekday()),
        "session": "+".join(str(item) for item in (snapshot.get("session") or [])) or "none",
        "bid": _finite(snapshot.get("bid")),
        "ask": _finite(snapshot.get("ask")),
        "spread_pips": spread_pips,
        "spread_relative_to_atr": (
            spread / atr
            if spread is not None and atr is not None and atr > 0
            else None
        ),
        "execution_cost_pips_round_trip": float(request.execution_cost_pips_round_trip),
        "atr_price": atr,
        "atr_pips": atr / request.pip_size if atr is not None and atr > 0 else None,
        "rsi": _finite(snapshot.get("rsi")),
        "momentum": _finite(snapshot.get("momentum")),
        "trend_strength": _finite(snapshot.get("trend_strength")),
        "volatility": _finite(snapshot.get("volatility")),
        "support_distance_pips": _finite(snapshot.get("support_distance_pips")),
        "resistance_distance_pips": _finite(snapshot.get("resistance_distance_pips")),
        "risk_reward": rr,
        "proposed_entry": _finite(snapshot.get("proposed_entry")),
        "stop_loss": _finite(snapshot.get("stop_loss")),
        "take_profit": _finite(snapshot.get("take_profit")),
        "open_positions": _finite(exposure.get("open_positions")),
        "portfolio_risk": _finite(exposure.get("portfolio_risk")),
        "gross_exposure": _finite(exposure.get("gross_exposure")),
        "pair_exposure": _finite(exposure.get("pair_exposure")),
        "free_margin": _finite(exposure.get("free_margin")),
        "account_currency": exposure.get("account_currency"),
        "news_risk": str(news.get("risk_level") or "UNKNOWN").upper(),
        "news_risk_score": _news_risk_score(str(news.get("risk_level") or "UNKNOWN")),
        "upcoming_news_currency": upcoming.get("currency"),
        "upcoming_news_impact": upcoming.get("impact_level"),
        "minutes_until_news": _finite(upcoming.get("minutes_until_event")),
        "recent_news_currency": recent.get("currency"),
        "minutes_since_news": _finite(recent.get("minutes_since_event")),
        "news_event_just_occurred": bool(news.get("event_just_occurred", False)),
        "news_freshness_state": freshness.get("state"),
        "news_stale": bool(freshness.get("stale", False)),
        "news_age_seconds": _finite(freshness.get("age_seconds")),
    }

    missing = [column for column in FEATURE_COLUMNS if column not in features]
    extra = [column for column in features if column not in FEATURE_COLUMNS]
    if missing or extra:
        raise ValueError(
            f"prediction feature schema mismatch; missing={missing}, extra={extra}"
        )
    return {column: features[column] for column in FEATURE_COLUMNS}


def _news_risk_score(level: str) -> int:
    return {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "UNKNOWN": -1}.get(level.upper(), -1)


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
