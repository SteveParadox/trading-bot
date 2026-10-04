"""Causal market snapshots for strategy-generated FX trade candidates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any

import pandas as pd

from fxbot.config import NewsEvent
from fxbot.instruments import FxInstrument, PriceSnapshot, split_instrument_name
from fxbot.models import FxPortfolioState, FxSignalIntent
from fxbot.risk import FxExitPlan
from fxbot.strategy import last_closed_window


MARKET_SNAPSHOT_VERSION = "v1"
DEFAULT_CANDLE_LOOKBACK = 20
DEFAULT_MOMENTUM_LOOKBACK = 3
DEFAULT_RSI_PERIOD = 14


@dataclass(frozen=True)
class CandleSnapshot:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": _utc(self.timestamp).isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


@dataclass(frozen=True)
class ExposureSnapshot:
    account_currency: str
    open_positions: int
    portfolio_risk: float
    gross_exposure: float
    pair_exposure: float
    currency_exposures: dict[str, float]
    free_margin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_currency": self.account_currency,
            "open_positions": self.open_positions,
            "portfolio_risk": self.portfolio_risk,
            "gross_exposure": self.gross_exposure,
            "pair_exposure": self.pair_exposure,
            "currency_exposures": dict(self.currency_exposures),
            "free_margin": self.free_margin,
        }


@dataclass(frozen=True)
class NewsEventSnapshot:
    name: str
    currency: str
    impact_level: str
    impact_score: int
    scheduled_at: datetime
    minutes_until_event: float | None
    minutes_since_event: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "currency": self.currency,
            "impact_level": self.impact_level,
            "impact_score": self.impact_score,
            "scheduled_at": _utc(self.scheduled_at).isoformat(),
            "minutes_until_event": self.minutes_until_event,
            "minutes_since_event": self.minutes_since_event,
        }


@dataclass(frozen=True)
class NewsContextSnapshot:
    affected_currencies: tuple[str, ...]
    upcoming_event: NewsEventSnapshot | None
    recent_event: NewsEventSnapshot | None
    event_just_occurred: bool
    freshness_state: str
    stale: bool
    age_seconds: float | None
    last_updated: datetime | None
    source: str
    risk_level: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "affected_currencies": list(self.affected_currencies),
            "upcoming_event": self.upcoming_event.to_dict() if self.upcoming_event else None,
            "recent_event": self.recent_event.to_dict() if self.recent_event else None,
            "event_just_occurred": self.event_just_occurred,
            "freshness": {
                "state": self.freshness_state,
                "stale": self.stale,
                "age_seconds": self.age_seconds,
                "last_updated": _utc(self.last_updated).isoformat() if self.last_updated else None,
                "source": self.source,
            },
            "risk_level": self.risk_level,
        }


@dataclass(frozen=True)
class MarketSnapshot:
    """Immutable features known when a strategy candidate is evaluated.

    Feature definitions are deliberately simple and versioned:
    - RSI: Wilder-style RSI over closed entry-timeframe candles.
    - momentum: close change over three closed candles, normalized by ATR.
    - trend_strength: current ADX from the strategy's closed signal candle.
    - volatility: current ATR divided by the median prior ATR over the snapshot
      lookback.
    - support/resistance distance: distance from executable entry to the nearest
      prior closed-candle low/high on the appropriate side of price.

    The latest forming candle is never used.
    """

    version: str
    candidate_id: str
    symbol: str
    timestamp: datetime
    signal_timestamp: datetime
    direction: str
    bid: float
    ask: float
    spread: float
    spread_pips: float
    recent_candles: tuple[CandleSnapshot, ...]
    atr: float | None
    rsi: float | None
    momentum: float | None
    trend_strength: float | None
    support_distance_pips: float | None
    resistance_distance_pips: float | None
    session: tuple[str, ...]
    volatility: float | None
    proposed_entry: float
    stop_loss: float | None
    take_profit: float | None
    risk_reward: float | None
    current_exposure: ExposureSnapshot
    news_context: NewsContextSnapshot

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "candidate_id": self.candidate_id,
            "symbol": self.symbol,
            "timestamp": _utc(self.timestamp).isoformat(),
            "signal_timestamp": _utc(self.signal_timestamp).isoformat(),
            "direction": self.direction,
            "bid": self.bid,
            "ask": self.ask,
            "spread": self.spread,
            "spread_pips": self.spread_pips,
            "recent_candles": [candle.to_dict() for candle in self.recent_candles],
            "atr": self.atr,
            "rsi": self.rsi,
            "momentum": self.momentum,
            "trend_strength": self.trend_strength,
            "support_distance_pips": self.support_distance_pips,
            "resistance_distance_pips": self.resistance_distance_pips,
            "session": list(self.session),
            "volatility": self.volatility,
            "proposed_entry": self.proposed_entry,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risk_reward": self.risk_reward,
            "current_exposure": self.current_exposure.to_dict(),
            "news_context": self.news_context.to_dict(),
        }


def build_market_snapshot(
    *,
    candidate_id: str,
    intent: FxSignalIntent,
    instrument: FxInstrument,
    price: PriceSnapshot,
    entry_frame: pd.DataFrame,
    timeframe: str,
    portfolio: FxPortfolioState,
    exit_plan: FxExitPlan | None,
    observed_at: datetime,
    sessions: set[str] | tuple[str, ...] | list[str],
    news_events: list[NewsEvent],
    news_stale: bool,
    news_age_seconds: float | None,
    news_last_updated: datetime | None,
    news_source: str,
    news_before_minutes: int,
    news_after_minutes: int,
    candle_lookback: int = DEFAULT_CANDLE_LOOKBACK,
    momentum_lookback: int = DEFAULT_MOMENTUM_LOOKBACK,
    rsi_period: int = DEFAULT_RSI_PERIOD,
) -> MarketSnapshot:
    """Build a point-in-time snapshot using only closed candles.

    This is an observation layer. It does not create trades, alter the strategy
    signal, size positions, or weaken any deterministic risk gate.
    """

    if candle_lookback < max(rsi_period + 1, momentum_lookback + 1):
        raise ValueError("candle_lookback is too short for configured features")
    if instrument.pip_size <= 0:
        raise ValueError("instrument pip size must be positive")
    observed = _utc(observed_at)
    window = last_closed_window(
        entry_frame,
        timeframe,
        timestamp=observed,
        lookback=candle_lookback,
        min_candles=min(candle_lookback, rsi_period + 1),
    )
    if window.empty:
        raise ValueError("no closed candles available for market snapshot")

    candles = tuple(_candle_snapshot(index, row) for index, row in window.iterrows())
    closes = pd.to_numeric(window["close"], errors="coerce")
    atrs = pd.to_numeric(window["atr"], errors="coerce") if "atr" in window else pd.Series(dtype=float)

    current_atr = _finite_or_none(atrs.iloc[-1]) if not atrs.empty else _finite_or_none(intent.signal_row.get("atr"))
    rsi = _rsi(closes, rsi_period)
    momentum = _normalized_momentum(closes, current_atr, momentum_lookback)
    trend_strength = _finite_or_none(intent.signal_row.get("adx"))
    volatility = _atr_volatility_ratio(atrs)

    prior = window.iloc[:-1]
    support_distance, resistance_distance = _support_resistance_distances(
        prior,
        entry=float(intent.entry_price),
        pip_size=instrument.pip_size,
    )

    spread = float(price.ask - price.bid)
    if not math.isfinite(spread) or spread <= 0:
        raise ValueError("market snapshot requires a positive executable spread")

    exposure = ExposureSnapshot(
        account_currency=str(portfolio.account_currency).upper(),
        open_positions=int(portfolio.open_positions),
        portfolio_risk=float(portfolio.portfolio_risk),
        gross_exposure=float(portfolio.gross_exposure),
        pair_exposure=float(portfolio.pair_exposures.get(instrument.name, 0.0)),
        currency_exposures={str(key): float(value) for key, value in portfolio.currency_exposures.items()},
        free_margin=float(portfolio.free_margin),
    )
    news_context = build_news_context(
        symbol=instrument.name,
        events=news_events,
        observed_at=observed,
        stale=news_stale,
        age_seconds=news_age_seconds,
        last_updated=news_last_updated,
        source=news_source,
        before_minutes=news_before_minutes,
        after_minutes=news_after_minutes,
    )

    if not candidate_id.strip():
        raise ValueError("candidate_id is required")
    return MarketSnapshot(
        version=MARKET_SNAPSHOT_VERSION,
        candidate_id=candidate_id,
        symbol=instrument.name,
        timestamp=observed,
        signal_timestamp=_utc(intent.timestamp),
        direction=intent.side.value,
        bid=float(price.bid),
        ask=float(price.ask),
        spread=spread,
        spread_pips=spread / instrument.pip_size,
        recent_candles=candles,
        atr=current_atr,
        rsi=rsi,
        momentum=momentum,
        trend_strength=trend_strength,
        support_distance_pips=support_distance,
        resistance_distance_pips=resistance_distance,
        session=tuple(sorted({str(value).lower() for value in sessions})),
        volatility=volatility,
        proposed_entry=float(intent.entry_price),
        stop_loss=float(exit_plan.stop_loss) if exit_plan is not None else None,
        take_profit=float(exit_plan.take_profit) if exit_plan is not None else None,
        risk_reward=float(exit_plan.risk_reward) if exit_plan is not None else None,
        current_exposure=exposure,
        news_context=news_context,
    )


def build_news_context(
    *,
    symbol: str,
    events: list[NewsEvent],
    observed_at: datetime,
    stale: bool,
    age_seconds: float | None,
    last_updated: datetime | None,
    source: str,
    before_minutes: int,
    after_minutes: int,
) -> NewsContextSnapshot:
    observed = _utc(observed_at)
    base, quote = split_instrument_name(symbol)
    affected = {base, quote}
    related = [event for event in events if event.currency.upper() in affected]
    future = sorted(
        (event for event in related if _utc(event.starts_at) > observed),
        key=lambda event: _utc(event.starts_at),
    )
    past = sorted(
        (event for event in related if _utc(event.starts_at) <= observed),
        key=lambda event: _utc(event.starts_at),
        reverse=True,
    )
    upcoming = _news_event_snapshot(future[0], observed) if future else None
    recent = _news_event_snapshot(past[0], observed) if past else None
    just_occurred = bool(
        recent is not None
        and recent.minutes_since_event is not None
        and recent.minutes_since_event <= max(0, after_minutes)
    )

    nearby: list[NewsEventSnapshot] = []
    if upcoming is not None and upcoming.minutes_until_event is not None and upcoming.minutes_until_event <= max(0, before_minutes):
        nearby.append(upcoming)
    if just_occurred and recent is not None:
        nearby.append(recent)

    if stale:
        risk_level = "UNKNOWN"
        freshness_state = "UNKNOWN"
    else:
        risk_level = max(
            (event.impact_level for event in nearby),
            key=_impact_rank,
            default="NONE",
        )
        freshness_state = "FRESH" if events else "EMPTY"

    return NewsContextSnapshot(
        affected_currencies=tuple(sorted(affected)),
        upcoming_event=upcoming,
        recent_event=recent,
        event_just_occurred=just_occurred,
        freshness_state=freshness_state,
        stale=bool(stale),
        age_seconds=_finite_or_none(age_seconds),
        last_updated=_utc(last_updated) if last_updated is not None else None,
        source=str(source or "unknown"),
        risk_level=risk_level,
    )


def _news_event_snapshot(event: NewsEvent, observed_at: datetime) -> NewsEventSnapshot:
    scheduled = _utc(event.starts_at)
    delta_minutes = (scheduled - observed_at).total_seconds() / 60.0
    return NewsEventSnapshot(
        name=event.name,
        currency=event.currency.upper(),
        impact_level=_event_impact_level(event),
        impact_score=int(event.impact_score or 0),
        scheduled_at=scheduled,
        minutes_until_event=delta_minutes if delta_minutes >= 0 else None,
        minutes_since_event=abs(delta_minutes) if delta_minutes < 0 else None,
    )


def _event_impact_level(event: NewsEvent) -> str:
    score = int(event.impact_score or 0)
    if score >= 71:
        return "HIGH"
    if score >= 31:
        return "MEDIUM"
    label = str(event.impact or "").upper()
    if label in {"HIGH", "MEDIUM", "LOW"}:
        return label
    return "LOW"


def _impact_rank(level: str) -> int:
    return {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "UNKNOWN": 4}.get(level, 0)


def _candle_snapshot(index: Any, row: pd.Series) -> CandleSnapshot:
    stamp = pd.Timestamp(index)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    volume = _finite_or_none(row.get("volume"))
    return CandleSnapshot(
        timestamp=stamp.to_pydatetime(),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=volume,
    )


def _rsi(closes: pd.Series, period: int) -> float | None:
    values = pd.to_numeric(closes, errors="coerce").dropna()
    if len(values) < period + 1:
        return None
    delta = values.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().iloc[-1]
    avg_loss = losses.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().iloc[-1]
    if not math.isfinite(float(avg_gain)) or not math.isfinite(float(avg_loss)):
        return None
    if avg_loss == 0 and avg_gain == 0:
        return 50.0
    if avg_loss == 0:
        return 100.0
    rs = float(avg_gain / avg_loss)
    return 100.0 - (100.0 / (1.0 + rs))


def _normalized_momentum(closes: pd.Series, atr: float | None, lookback: int) -> float | None:
    values = pd.to_numeric(closes, errors="coerce").dropna()
    if atr is None or atr <= 0 or len(values) < lookback + 1:
        return None
    value = (float(values.iloc[-1]) - float(values.iloc[-1 - lookback])) / atr
    return value if math.isfinite(value) else None


def _atr_volatility_ratio(atrs: pd.Series) -> float | None:
    values = pd.to_numeric(atrs, errors="coerce").dropna()
    if len(values) < 2:
        return None
    current = float(values.iloc[-1])
    baseline = float(values.iloc[:-1].median())
    if current <= 0 or baseline <= 0 or not math.isfinite(current) or not math.isfinite(baseline):
        return None
    ratio = current / baseline
    return ratio if math.isfinite(ratio) else None


def _support_resistance_distances(
    prior: pd.DataFrame,
    *,
    entry: float,
    pip_size: float,
) -> tuple[float | None, float | None]:
    if prior.empty:
        return None, None
    lows = [float(value) for value in pd.to_numeric(prior["low"], errors="coerce").dropna() if float(value) < entry]
    highs = [float(value) for value in pd.to_numeric(prior["high"], errors="coerce").dropna() if float(value) > entry]
    support = max(lows) if lows else None
    resistance = min(highs) if highs else None
    support_distance = (entry - support) / pip_size if support is not None else None
    resistance_distance = (resistance - entry) / pip_size if resistance is not None else None
    return support_distance, resistance_distance


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
