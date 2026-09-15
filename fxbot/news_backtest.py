"""News backtesting engine (news3.txt: steps 14-16).

Answers the questions that should precede trusting any news guard:
  - "Does my bot actually perform better when it avoids high-impact news?"
  - "Which blackout window is best (None / +-5 / +-10 / +-15 / +-30 / +-60)?" 
  - "Does adding sentiment / post-news volatility guards actually contribute?"

Design notes:
  - Pure, deterministic, dependency-light.  Trades are represented by
    ``BacktestTradeRecord`` with the market context the spec asks for (price,
    spread, ATR, bot decision, trade result) so any historical dataset -- from
    the SQLAlchemy journal, the legacy backtester, or a CSV -- can be mapped in.
  - Events are plain ``NewsEvent`` records from ``fxbot.news`` (shared schema).
  - Every statistic is computed over R-multiples so results are broker-agnostic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from statistics import mean
from typing import Callable, Mapping

from fxbot.config import NewsEvent
from fxbot.instruments import split_instrument_name
from fxbot.news import (
    SentimentResult,
    VolatilityDetector,
    VolatilityReading,
    _as_utc,
    sentiment_contribution,
)

DEFAULT_BLACKOUT_WINDOWS: tuple[int | None, ...] = (None, 5, 10, 15, 30, 60)


def symbol_currencies(symbol: str) -> set[str]:
    """Return the two currencies that make up an FX pair."""
    base, quote = split_instrument_name(symbol)
    return {base, quote}


@dataclass(frozen=True)
class MarketContext:
    """Market state at/around trade entry (Step 14 dataset row)."""

    price: float
    spread_pips: float
    atr_pips: float
    baseline_atr_pips: float | None = None

    @property
    def atr_ratio(self) -> float | None:
        if self.baseline_atr_pips is None or self.baseline_atr_pips <= 0:
            return None
        return self.atr_pips / self.baseline_atr_pips


@dataclass(frozen=True)
class BacktestTradeRecord:
    """One historical trade with news context attached.

    ``r_multiple`` is the outcome in R units (positive = win) so statistics are
    comparable across account sizes.  ``mae``/``mfe`` are the max adverse and
    max favourable excursions in R during the trade's lifetime.
    """

    trade_id: str
    symbol: str
    entry_time: datetime
    side: str  # "BUY" / "SELL"
    r_multiple: float
    pnl: float = 0.0
    mae: float = 0.0
    mfe: float = 0.0
    context: MarketContext | None = None
    exit_time: datetime | None = None
    # News tags populated by tag_trades_with_events.
    event_id: str | None = None
    event_name: str | None = None
    event_currency: str | None = None
    minutes_from_event: float | None = None
    before_event: bool | None = None


# ── Step 14: attach news context to trades ───────────────────────────────────


def _closest_event(trade: BacktestTradeRecord, events: list[NewsEvent]) -> NewsEvent | None:
    """Return the temporally closest event that affects the trade's symbol."""
    currencies = symbol_currencies(trade.symbol)
    entry = _as_utc(trade.entry_time)
    best: NewsEvent | None = None
    best_distance: float | None = None
    for event in events:
        if event.currency.upper() not in currencies:
            continue
        start = _as_utc(event.starts_at)
        distance = abs((start - entry).total_seconds()) / 60.0
        if best is None or distance < best_distance:
            best = event
            best_distance = distance
    return best


def tag_trades_with_events(
    trades: list[BacktestTradeRecord],
    events: list[NewsEvent],
) -> list[BacktestTradeRecord]:
    """Return trade copies tagged with their closest relevant news event."""
    tagged: list[BacktestTradeRecord] = []
    for trade in trades:
        event = _closest_event(trade, events)
        if event is None:
            tagged.append(trade)
            continue
        entry = _as_utc(trade.entry_time)
        start = _as_utc(event.starts_at)
        tagged.append(
            replace(
                trade,
                event_id=event.event_id,
                event_name=event.name,
                event_currency=event.currency,
                minutes_from_event=abs((start - entry).total_seconds()) / 60.0,
                before_event=entry < start,
            )
        )
    return tagged


# ── Step 14: statistics ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class TradeGroupStats:
    """Performance statistics for a group of trades."""

    group: str
    trades: int
    win_rate: float
    avg_r: float
    profit_factor: float
    avg_mae: float
    avg_mfe: float
    net_r: float
    max_drawdown_r: float


def _profit_factor(r_multiples: list[float]) -> float:
    gross_profit = sum(value for value in r_multiples if value > 0)
    gross_loss = abs(sum(value for value in r_multiples if value < 0))
    if gross_loss > 0:
        return gross_profit / gross_loss
    return math.inf if gross_profit > 0 else 0.0


def _max_drawdown_r(r_multiples: list[float]) -> float:
    """Max peak-to-trough drawdown on the cumulative R curve."""
    peak = 0.0
    running = 0.0
    max_dd = 0.0
    for value in r_multiples:
        running += value
        peak = max(peak, running)
        max_dd = max(max_dd, peak - running)
    return max_dd


def compute_group_stats(group: str, records: list[BacktestTradeRecord]) -> TradeGroupStats:
    """Compute win rate / profit factor / drawdown for a list of trades."""
    r_multiples = [record.r_multiple for record in records]
    wins = [value for value in r_multiples if value > 0]
    if not r_multiples:
        return TradeGroupStats(
            group=group,
            trades=0,
            win_rate=0.0,
            avg_r=0.0,
            profit_factor=0.0,
            avg_mae=0.0,
            avg_mfe=0.0,
            net_r=0.0,
            max_drawdown_r=0.0,
        )
    return TradeGroupStats(
        group=group,
        trades=len(records),
        win_rate=(len(wins) / len(records)) if records else 0.0,
        avg_r=mean(r_multiples),
        profit_factor=_profit_factor(r_multiples),
        avg_mae=mean(record.mae for record in records) if records else 0.0,
        avg_mfe=mean(record.mfe for record in records) if records else 0.0,
        net_r=sum(r_multiples),
        max_drawdown_r=_max_drawdown_r(r_multiples),
    )


def group_news_stats(
    tagged: list[BacktestTradeRecord],
    *,
    split_before_after: bool = True,
) -> list[TradeGroupStats]:
    """Group tagged trades by event (optionally before/after) and report stats.

    Mirrors the Step 14 example:
        NFP         Trades before : 124, after : 137
        Win rate    before : 48%,  after : 56%
        Avg return  +0.32R ...
    """
    buckets: dict[str, list[BacktestTradeRecord]] = {}
    for trade in tagged:
        if trade.event_name is None:
            continue
        if split_before_after and trade.before_event is not None:
            key = f"{trade.event_name}:{'before' if trade.before_event else 'after'}"
        else:
            key = trade.event_name
        buckets.setdefault(key, []).append(trade)
    return [compute_group_stats(key, records) for key, records in sorted(buckets.items())]


# ── Step 15: compare blackout windows ────────────────────────────────────────


@dataclass(frozen=True)
class BlackoutWindowResult:
    """Outcome when a given blackout window is applied to a trade set."""

    window_minutes: int | None
    trades: int
    win_rate: float
    profit_factor: float
    max_drawdown_r: float
    avg_r: float
    net_r: float


def trade_within_window(tagged: BacktestTradeRecord, window_minutes: int | None) -> bool:
    """Return True when the trade falls inside a blackout window around news.

    ``None`` means "no blackout" and never filters (matches the spec's
    baseline row).
    """
    if window_minutes is None:
        return False
    if tagged.minutes_from_event is None:
        return False
    return tagged.minutes_from_event <= window_minutes


def apply_blackout_window(
    tagged: list[BacktestTradeRecord],
    window_minutes: int | None,
) -> list[BacktestTradeRecord]:
    """Return trades that survive a blackout window (i.e. would be allowed)."""
    return [trade for trade in tagged if not trade_within_window(trade, window_minutes)]


def compare_blackout_windows(
    tagged: list[BacktestTradeRecord],
    *,
    windows: tuple[int | None, ...] = DEFAULT_BLACKOUT_WINDOWS,
) -> list[BlackoutWindowResult]:
    """Compare win rate / profit factor / drawdown across blackout windows.

    Lets the operator discover (e.g.) that +-15 minutes beats +-30 instead of
    guessing.  When ``windows`` is (None, 5, 10, 15, 30, 60) the first result
    is the untouched baseline.
    """
    results: list[BlackoutWindowResult] = []
    for window in windows:
        allowed = apply_blackout_window(tagged, window)
        stats = compute_group_stats(f"window={window}", allowed)
        results.append(
            BlackoutWindowResult(
                window_minutes=window,
                trades=stats.trades,
                win_rate=stats.win_rate,
                profit_factor=stats.profit_factor,
                max_drawdown_r=stats.max_drawdown_r,
                avg_r=stats.avg_r,
                net_r=stats.net_r,
            )
        )
    return results


# ── Step 16: feature ablation A/B/C/D ────────────────────────────────────────


@dataclass(frozen=True)
class StrategyExperiment:
    """Configuration for one strategy variant in the A/B/C/D comparison."""

    label: str  # "A", "B", "C", "D"
    name: str  # "Technical only", ...
    blackout_window_minutes: int | None = None
    require_sentiment_confirm: bool = False
    block_extreme_volatility: bool = False


@dataclass(frozen=True)
class ExperimentMetrics:
    """Comparison row for a strategy variant."""

    label: str
    name: str
    trade_count: int
    win_rate: float
    profit_factor: float
    max_drawdown_r: float
    net_r: float  # Return
    sharpe: float
    avg_r: float
    avg_spread_pips: float  # Slippage proxy


def sentiment_confirm(
    tagged: BacktestTradeRecord,
    sentiment_map: Mapping[str, SentimentResult],
) -> bool:
    """True when LLM sentiment agrees with the trade direction.

    Only used when an experiment sets ``require_sentiment_confirm``; trades
    with no matching sentiment are dropped (the strategy would not take them).
    """
    if tagged.event_currency is None:
        return False
    result = sentiment_map.get(tagged.event_currency.upper())
    if result is None:
        return False
    contribution = sentiment_contribution(result.sentiment)
    if contribution == 0:
        return False
    side = tagged.side.upper()
    return (contribution > 0 and side == "BUY") or (contribution < 0 and side == "SELL")


def default_extreme_at_entry(
    readings: Mapping[str, list[tuple[datetime, float]]],
    detector: VolatilityDetector | None = None,
    lookback_minutes: float = 5.0,
) -> Callable[[BacktestTradeRecord], bool]:
    """Return a guard that blocks trades entered during extreme volatility.

    ``readings`` maps symbol -> list of (timestamp, atr_ratio) samples.  A
    trade is blocked when any sample within ``lookback_minutes`` *before* its
    entry shows an extreme volatility ratio.
    """
    detector = detector or VolatilityDetector()

    def guard(tagged: BacktestTradeRecord) -> bool:
        samples = readings.get(tagged.symbol, [])
        entry = _as_utc(tagged.entry_time)
        for timestamp, ratio in samples:
            if ratio <= 0:
                continue
            reading = detector.measure(1.0, ratio)
            sampled_at = _as_utc(timestamp)
            within_window = 0 <= (entry - sampled_at).total_seconds() / 60.0 <= lookback_minutes
            if reading.is_extreme and within_window:
                return True
        return False

    return guard


def _sharpe_of_r(r_multiples: list[float]) -> float:
    if len(r_multiples) < 2:
        return 0.0
    volatility = _population_std(r_multiples)
    if volatility <= 0:
        return 0.0
    return mean(r_multiples) / volatility * math.sqrt(len(r_multiples))


def _population_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def run_experiment(
    tagged: list[BacktestTradeRecord],
    experiment: StrategyExperiment,
    *,
    sentiment_map: Mapping[str, SentimentResult] | None = None,
    extreme_at_entry: Callable[[BacktestTradeRecord], bool] | None = None,
) -> ExperimentMetrics:
    """Evaluate one strategy variant over the tagged trade set."""
    selected = tagged
    if experiment.blackout_window_minutes is not None:
        selected = apply_blackout_window(selected, experiment.blackout_window_minutes)
    if experiment.require_sentiment_confirm:
        selected = [t for t in selected if sentiment_confirm(t, sentiment_map or {})]
    if experiment.block_extreme_volatility:
        if extreme_at_entry is not None:
            selected = [t for t in selected if not extreme_at_entry(t)]

    r_multiples = [trade.r_multiple for trade in selected]
    stats = compute_group_stats(experiment.label or experiment.name, selected)
    spreads = [
        trade.context.spread_pips for trade in selected if trade.context is not None and trade.context.spread_pips is not None
    ]
    return ExperimentMetrics(
        label=experiment.label,
        name=experiment.name,
        trade_count=stats.trades,
        win_rate=stats.win_rate,
        profit_factor=stats.profit_factor,
        max_drawdown_r=stats.max_drawdown_r,
        net_r=stats.net_r,
        sharpe=_sharpe_of_r(r_multiples),
        avg_r=stats.avg_r,
        avg_spread_pips=mean(spreads) if spreads else 0.0,
    )


def compare_strategy_experiments(
    tagged: list[BacktestTradeRecord],
    experiments: list[StrategyExperiment],
    *,
    sentiment_map: Mapping[str, SentimentResult] | None = None,
    extreme_at_entry: Callable[[BacktestTradeRecord], bool] | None = None,
) -> list[ExperimentMetrics]:
    """Run the full A/B/C/D ablation (or any variant list) and return a table.

    Default experiments follow the Step 16 sequence:
      A. Technical only
      B. Technical + news blackout
      C. Technical + blackout + sentiment
      D. Technical + blackout + sentiment + post-news volatility
    """
    if experiments is None or len(experiments) == 0:  # pragma: no cover - defensive
        raise ValueError("at least one experiment is required")
    return [
        run_experiment(
            tagged,
            experiment,
            sentiment_map=sentiment_map,
            extreme_at_entry=extreme_at_entry,
        )
        for experiment in experiments
    ]


def default_experiment_set(blackout_window_minutes: int = 15) -> list[StrategyExperiment]:
    """Return the Step 16 A/B/C/D configurations."""
    return [
        StrategyExperiment(label="A", name="Technical only", blackout_window_minutes=None),
        StrategyExperiment(
            label="B",
            name="Technical + news blackout",
            blackout_window_minutes=blackout_window_minutes,
        ),
        StrategyExperiment(
            label="C",
            name="Technical + blackout + sentiment",
            blackout_window_minutes=blackout_window_minutes,
            require_sentiment_confirm=True,
        ),
        StrategyExperiment(
            label="D",
            name="Technical + blackout + sentiment + post-news volatility",
            blackout_window_minutes=blackout_window_minutes,
            require_sentiment_confirm=True,
            block_extreme_volatility=True,
        ),
    ]