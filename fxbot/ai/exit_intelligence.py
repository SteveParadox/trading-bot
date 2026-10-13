"""Observation-only, versioned exit intelligence.

This service is intentionally broker-free. It cannot send, amend, or close
an MT5 order. Every observation is causal and its ML decision is a proposal.
An execution-capable advisory policy is NOT implemented or approved here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any
from uuid import uuid4


EXIT_FEATURE_VERSION = "exit-v2"
EXIT_ACTIONS = ("HOLD", "TAKE_PROFIT_NOW", "TRAIL_STOP", "REDUCE_POSITION", "EXIT")
EXIT_FEATURE_COLUMNS = (
    "direction", "spread_pips", "holding_seconds", "pnl_pips",
    "mfe_pips", "mae_pips", "drawdown_from_peak_pips",
    "distance_to_sl_pips", "distance_to_tp_pips", "position_units",
    "broker_unrealized_pl", "atr_pips", "rsi", "momentum",
)


class ExitAction(str, Enum):
    HOLD = "HOLD"
    TAKE_PROFIT_NOW = "TAKE_PROFIT_NOW"
    TRAIL_STOP = "TRAIL_STOP"
    REDUCE_POSITION = "REDUCE_POSITION"
    EXIT = "EXIT"


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _time(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError("missing or invalid timestamp")
    if parsed.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _order_price(order: Any) -> float | None:
    if isinstance(order, dict):
        value = _finite(order.get("price"))
        return value if value is not None and value > 0 else None
    return None


@dataclass(frozen=True)
class ExitPrediction:
    prediction_id: str
    position_id: str
    timestamp: str
    status: str
    decision: str | None
    confidence: float | None
    reason_codes: tuple[str, ...]
    model_version: str | None
    feature_version: str
    prompt_version: str | None
    strategy_version: str | None
    model_sha256: str | None = None
    probabilities: dict[str, float] | None = None
    error: str | None = None
    feature_builder_version: str = EXIT_FEATURE_VERSION
    model_name: str | None = None
    model_evaluation: str = "configured_trusted_shadow_artifact"

    def __post_init__(self) -> None:
        if self.status not in {"ok", "unavailable", "error"}:
            raise ValueError("invalid prediction status")
        if self.decision is not None and self.decision not in EXIT_ACTIONS:
            raise ValueError("invalid exit decision")
        if self.status != "ok" and (self.decision is not None or self.confidence is not None or self.probabilities is not None):
            raise ValueError("failed prediction cannot contain a recommendation")
        if self.status == "ok":
            if self.decision is None or self.confidence is None or not 0 <= self.confidence <= 1:
                raise ValueError("successful prediction requires valid action/confidence")
            if not self.probabilities or set(self.probabilities) != set(EXIT_ACTIONS):
                raise ValueError("successful prediction requires all action probabilities")
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in self.probabilities.values()):
                raise ValueError("invalid exit probabilities")
            if not math.isclose(sum(self.probabilities.values()), 1, abs_tol=1e-6):
                raise ValueError("exit probabilities must sum to one")
            if (not math.isclose(self.confidence, self.probabilities[self.decision], abs_tol=1e-6)
                    or self.probabilities[self.decision] != max(self.probabilities.values())):
                raise ValueError("decision/confidence does not match probabilities")
            if not self.model_version or not self.model_sha256 or len(self.model_sha256) != 64:
                raise ValueError("successful prediction requires model attribution")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_exit_snapshot(
    trade: dict[str, Any],
    price: Any,
    instrument: Any,
    timestamp: datetime,
    *,
    recorded_payload: dict[str, Any] | None = None,
    strategy_version: str | None = None,
    maximum_quote_age_seconds: float = 120,
) -> dict[str, Any]:
    """Capture only broker state and observations known as of timestamp."""
    now = _time(timestamp)
    quote_at = _time(price.time)
    if (not math.isfinite(maximum_quote_age_seconds) or maximum_quote_age_seconds <= 0
            or quote_at > now or (now - quote_at).total_seconds() > maximum_quote_age_seconds):
        raise ValueError("exit snapshot needs a fresh causal quote")
    ticket = str(trade.get("id") or "").strip()
    units = _finite(trade.get("currentUnits"))
    entry = _finite(trade.get("price"))
    bid, ask = _finite(price.bid), _finite(price.ask)
    pip_size = _finite(instrument.pip_size)
    if not ticket or units is None or units == 0 or entry is None or entry <= 0:
        raise ValueError("invalid broker position")
    if bid is None or ask is None or bid <= 0 or ask < bid or pip_size is None or pip_size <= 0:
        raise ValueError("invalid executable quote")
    opened = _time(trade.get("openTime"))
    if opened > now:
        raise ValueError("position timestamp lies in future")
    if quote_at < opened:
        raise ValueError("quote predates position entry")
    if str(trade.get("instrument") or instrument.name) != instrument.name:
        raise ValueError("position instrument mismatch")
    side = 1 if units > 0 else -1
    liquidation = bid if side == 1 else ask
    move = side * (liquidation - entry) / pip_size
    stop = _order_price(trade.get("stopLossOrder"))
    take_profit = _order_price(trade.get("takeProfitOrder"))
    recorded = recorded_payload or {}
    excursions = recorded.get("sniper_excursions") or {}
    mfe_raw, mae_raw = _finite(excursions.get("mfe")), _finite(excursions.get("mae"))
    excursion_at = _time(excursions["last_sample"]) if excursions.get("last_sample") else None
    observed = (mfe_raw is not None and mae_raw is not None and mfe_raw >= 0 and mae_raw >= 0
                and excursion_at is not None and opened <= excursion_at <= now)
    mfe = max(mfe_raw, move * pip_size, 0.) / pip_size if observed else None
    mae = max(mae_raw, -move * pip_size, 0.) / pip_size if observed else None
    # Sampled excursions are lower bounds, not tick-complete extrema.
    context = recorded.get("strategy_context") or {}
    lifetime = hashlib.sha256(json.dumps(
        [ticket, instrument.name, opened.isoformat(), entry, side], separators=(",", ":")
    ).encode()).hexdigest()
    return {
        "schema_version": EXIT_FEATURE_VERSION,
        "position_id": ticket,
        "position_lifetime_id": lifetime,
        "candidate_id": context.get("candidate_id"),
        "timestamp": now.isoformat(),
        "quote_timestamp": quote_at.isoformat(),
        "symbol": str(trade.get("instrument") or instrument.name),
        "direction": "BUY" if side == 1 else "SELL",
        "entry_timestamp": opened.isoformat(),
        "entry_price": entry,
        "units": units,
        "broker_volume_lots": _finite((trade.get("mt5") or {}).get("volume")),
        "pip_size": pip_size,
        "liquidation_price": liquidation,
        "bid": bid,
        "ask": ask,
        "spread_pips": (ask - bid) / pip_size,
        "pnl_pips": move,
        "broker_unrealized_pl": _finite(trade.get("unrealizedPL")),
        # Net P&L cannot be inferred without reliable broker commission,
        # conversion, swap and anticipated exit execution costs.
        "estimated_net_pl": None,
        "stop_loss": stop,
        "take_profit": take_profit,
        "initial_stop_loss": _finite(context.get("initial_stop")),
        "initial_take_profit": _finite(context.get("initial_take_profit")),
        "distance_to_sl_pips": abs(liquidation - stop) / pip_size if stop is not None else None,
        "distance_to_tp_pips": abs(take_profit - liquidation) / pip_size if take_profit is not None else None,
        "holding_seconds": (now - opened).total_seconds(),
        "mfe_pips": mfe,
        "mae_pips": mae,
        "drawdown_from_peak_pips": max(0., mfe - move) if mfe is not None else None,
        "excursion_quality": "sampled_lower_bound" if observed else "historical_excursions_unavailable",
        "sampled_excursions_only": True,
        "strategy_version": context.get("strategy_hash") or strategy_version,
        "quote_max_age_seconds": maximum_quote_age_seconds,
        "atr_pips": None,
        "rsi": None,
        "momentum": None,
        "previous_partial_closes": None,
        "previous_stop_modifications": None,
        "news_context": None,
    }


def exit_features(snapshot: dict[str, Any]) -> dict[str, Any]:
    if snapshot.get("schema_version") != EXIT_FEATURE_VERSION:
        raise ValueError("exit snapshot feature version mismatch")
    direction = snapshot.get("direction")
    if direction not in {"BUY", "SELL"}:
        raise ValueError("invalid position direction")
    units = _finite(snapshot.get("units"))
    if units is None or units == 0 or (units > 0) != (direction == "BUY"):
        raise ValueError("invalid signed position units")
    values = {
        "direction": direction,
        "spread_pips": snapshot.get("spread_pips"),
        "holding_seconds": snapshot.get("holding_seconds"),
        "pnl_pips": snapshot.get("pnl_pips"),
        "mfe_pips": snapshot.get("mfe_pips"),
        "mae_pips": snapshot.get("mae_pips"),
        "drawdown_from_peak_pips": snapshot.get("drawdown_from_peak_pips"),
        "distance_to_sl_pips": snapshot.get("distance_to_sl_pips"),
        "distance_to_tp_pips": snapshot.get("distance_to_tp_pips"),
        "position_units": abs(units),
        "broker_unrealized_pl": snapshot.get("broker_unrealized_pl"),
        "atr_pips": snapshot.get("atr_pips"),
        "rsi": snapshot.get("rsi"),
        "momentum": snapshot.get("momentum"),
    }
    for key, value in values.items():
        if key != "direction" and value is not None and _finite(value) is None:
            raise ValueError(f"invalid exit feature: {key}")
    return values


class ExitPredictionService:
    """SHA-checked, lazy local predictor. An absent model never yields HOLD as fake ML."""

    def __init__(self, *, model_path: str = "", metadata_path: str = "", verify_hash: bool = True) -> None:
        if not verify_hash:
            raise ValueError("exit model integrity verification cannot be disabled")
        self.model_path = Path(model_path) if model_path else None
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.verify_hash = verify_hash
        self._artifact: tuple[Any, dict[str, Any], str] | None = None

    def _load(self) -> tuple[Any, dict[str, Any], str]:
        if self._artifact is not None:
            return self._artifact
        if self.model_path is None or self.metadata_path is None:
            raise FileNotFoundError("no approved exit model configured")
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or metadata.get("target") != "EXIT_ACTION":
            raise ValueError("exit model target mismatch")
        if metadata.get("feature_columns") != list(EXIT_FEATURE_COLUMNS):
            raise ValueError("exit model feature manifest mismatch")
        if metadata.get("feature_builder_version") != EXIT_FEATURE_VERSION:
            raise ValueError("exit model feature-builder mismatch")
        if metadata.get("class_labels") != list(EXIT_ACTIONS):
            raise ValueError("exit model class order mismatch")
        if not isinstance(metadata.get("model_version"), str) or not metadata["model_version"].strip():
            raise ValueError("exit model version missing")
        raw = self.model_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if self.verify_hash and (not metadata.get("model_sha256") or digest != metadata["model_sha256"]):
            raise ValueError("exit model SHA-256 mismatch")
        import joblib
        model = joblib.load(io.BytesIO(raw))
        if not callable(getattr(model, "predict_proba", None)):
            raise ValueError("exit artifact lacks predict_proba")
        self._artifact = (model, metadata, digest)
        return self._artifact

    def predict(self, snapshot: dict[str, Any]) -> ExitPrediction:
        timestamp = str(snapshot["timestamp"])
        position_id = str(snapshot["position_id"])
        base = {
            "prediction_id": uuid4().hex,
            "position_id": position_id,
            "timestamp": timestamp,
            "feature_version": EXIT_FEATURE_VERSION,
            "prompt_version": None,
            "strategy_version": snapshot.get("strategy_version"),
        }
        if self.model_path is None or self.metadata_path is None:
            return ExitPrediction(**base, status="unavailable", decision=None,
                                  confidence=None, reason_codes=("MODEL_NOT_CONFIGURED",), model_version=None)
        metadata, digest = {}, None
        try:
            features = exit_features(snapshot)
            model, metadata, digest = self._load()
            import pandas as pd
            classes = list(model.classes_)
            if classes != list(range(len(EXIT_ACTIONS))):
                raise ValueError("exit estimator classes mismatch")
            probabilities = [float(v) for v in model.predict_proba(pd.DataFrame([features], columns=EXIT_FEATURE_COLUMNS))[0]]
            if len(probabilities) != len(EXIT_ACTIONS):
                raise ValueError("exit probability count mismatch")
            distribution = dict(zip(EXIT_ACTIONS, probabilities))
            best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
            return ExitPrediction(**base, status="ok", decision=EXIT_ACTIONS[best_index],
                                  confidence=probabilities[best_index], reason_codes=("MODEL_RECOMMENDATION",),
                                  model_version=str(metadata.get("model_version") or ""),
                                  model_name=metadata.get("model_name"),
                                  model_sha256=digest, probabilities=distribution)
        except FileNotFoundError:
            return ExitPrediction(**base, status="unavailable", decision=None, confidence=None,
                                  reason_codes=("MODEL_FILE_UNAVAILABLE",), model_version=None)
        except Exception as exc:
            # Do not reveal filesystem paths or model payloads in the journal.
            return ExitPrediction(**base, status="error", decision=None, confidence=None,
                                  reason_codes=("PREDICTION_FAILED",), model_version=metadata.get("model_version"),
                                  model_sha256=digest, model_name=metadata.get("model_name"),
                                  error=type(exc).__name__)
