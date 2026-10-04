"""Reconstruct historical FX strategy candidates and outcomes from MT5 history.

This is conditional candidate research, not a portfolio backtest. It reuses the
live FX strategy, market/news permission gate, market snapshot builder, risk
exit-plan logic, candidate journal, outcome tracker, and training-dataset
builder. No order is ever submitted.

Historical reconstruction requires executable bid/ask ticks for candidate
entry/outcome labels. Missing quote coverage produces missing/incomplete labels,
never inferred wins or losses.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

import pandas as pd

from fxbot.config import FxBotSettings, NewsEvent, load_news_events, settings_from_env
from fxbot.instruments import FxInstrument, PriceSnapshot, normalize_instrument_name, split_instrument_name
from fxbot.journal import StructuredJournal
from fxbot.market_hours import active_sessions, can_trade
from fxbot.market_snapshot import build_market_snapshot
from fxbot.models import FxPortfolioState, Side
from fxbot.mt5 import Mt5Client
from fxbot.outcome_tracker import CandidateOutcomeTracker, MAX_OUTCOME_HORIZON_SECONDS
from fxbot.risk import FxRiskManager, reward_covers_spread
from fxbot.strategy import (
    FX_MIN_HTF_CANDLES,
    FX_MIN_SIGNAL_CANDLES,
    TIMEFRAME_DELTAS,
    build_signal_intent,
    evaluate_signal_frame,
    last_closed_row,
    prepare_indicators,
)
from fxbot.training_dataset import export_training_dataset


HISTORICAL_RECONSTRUCTION_VERSION = "v1"


class HistoricalDataSource(Protocol):
    def instruments(self, names: list[str]) -> dict[str, FxInstrument]: ...
    def historical_candles(
        self,
        instrument: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame: ...
    def historical_ticks(
        self,
        instrument: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame: ...


@dataclass(frozen=True)
class HistoricalReconstructionConfig:
    start: datetime
    end: datetime
    entry_quote_tolerance_seconds: int = 30
    outcome_lag_tolerance_seconds: int = 30
    warmup_bars: int = 120

    def validate(self) -> None:
        start = _utc(self.start)
        end = _utc(self.end)
        if end <= start:
            raise ValueError("historical reconstruction end must be after start")
        if self.entry_quote_tolerance_seconds <= 0:
            raise ValueError("entry quote tolerance must be positive")
        if self.outcome_lag_tolerance_seconds <= 0:
            raise ValueError("outcome lag tolerance must be positive")
        if self.warmup_bars < max(FX_MIN_SIGNAL_CANDLES, FX_MIN_HTF_CANDLES):
            raise ValueError("warmup_bars is too small for the strategy")


def reconstruct_historical_candidates(
    *,
    source: HistoricalDataSource,
    settings: FxBotSettings,
    journal: StructuredJournal,
    config: HistoricalReconstructionConfig,
    symbols: list[str] | None = None,
    news_events: list[NewsEvent] | None = None,
    news_authoritative: bool = False,
) -> dict[str, Any]:
    """Replay historical market states into the same candidate/outcome schema."""

    config.validate()
    start = _utc(config.start)
    end = _utc(config.end)
    events = list(news_events or [])
    if settings.strategy.require_news_data and not news_authoritative:
        raise ValueError(
            "Historical reconstruction cannot match production with "
            "FX_REQUIRE_NEWS_DATA=true unless authoritative historical news is supplied"
        )

    requested = [normalize_instrument_name(symbol) for symbol in (symbols or settings.instruments)]
    instruments = source.instruments(requested)
    risk = FxRiskManager(settings.risk, settings.strategy)
    tracker = CandidateOutcomeTracker(
        journal,
        observation_lag_tolerance_seconds=float(config.outcome_lag_tolerance_seconds),
    )
    strategy_hash = _settings_hash(settings)
    report: dict[str, Any] = {
        "kind": "historical_conditional_candidate_reconstruction",
        "not_portfolio_backtest": True,
        "version": HISTORICAL_RECONSTRUCTION_VERSION,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "strategy_hash": strategy_hash,
        "news_authoritative": news_authoritative,
        "account_state_replayed": False,
        "ai_replayed": False,
        "sniper_execution_gate_replayed": False,
        "strategy_replay_semantics": "current configured strategy applied counterfactually to historical market data",
        "symbols": {},
    }

    for symbol in requested:
        instrument = instruments[symbol]
        symbol_events = _events_for_symbol(events, symbol)
        symbol_report = _reconstruct_symbol(
            source=source,
            settings=settings,
            journal=journal,
            tracker=tracker,
            risk=risk,
            instrument=instrument,
            config=config,
            news_events=symbol_events,
            news_authoritative=news_authoritative,
            strategy_hash=strategy_hash,
        )
        report["symbols"][symbol] = symbol_report

    report["totals"] = {
        key: sum(int(value.get(key, 0)) for value in report["symbols"].values())
        for key in (
            "decision_points",
            "permission_blocks",
            "strategy_candidates",
            "candidates_with_entry_quotes",
            "complete_outcomes",
            "incomplete_outcomes",
        )
    }
    return report


def _reconstruct_symbol(
    *,
    source: HistoricalDataSource,
    settings: FxBotSettings,
    journal: StructuredJournal,
    tracker: CandidateOutcomeTracker,
    risk: FxRiskManager,
    instrument: FxInstrument,
    config: HistoricalReconstructionConfig,
    news_events: list[NewsEvent],
    news_authoritative: bool,
    strategy_hash: str,
) -> dict[str, Any]:
    start = _utc(config.start)
    end = _utc(config.end)
    entry_delta = TIMEFRAME_DELTAS[settings.strategy.entry_timeframe]
    htf_delta = TIMEFRAME_DELTAS[settings.strategy.htf_timeframe]
    warmup_start = start - max(entry_delta, htf_delta) * config.warmup_bars

    entry_raw = source.historical_candles(
        instrument.name,
        settings.strategy.entry_timeframe,
        warmup_start,
        end,
    )
    htf_raw = source.historical_candles(
        instrument.name,
        settings.strategy.htf_timeframe,
        warmup_start,
        end,
    )
    if entry_raw.empty or htf_raw.empty:
        return {
            "decision_points": 0,
            "permission_blocks": 0,
            "strategy_candidates": 0,
            "candidates_with_entry_quotes": 0,
            "complete_outcomes": 0,
            "incomplete_outcomes": 0,
            "reason": "missing_historical_candles",
        }

    entry = prepare_indicators(_ohlcv(entry_raw))
    htf = prepare_indicators(_ohlcv(htf_raw))
    tp_frame: pd.DataFrame | None = None
    if settings.strategy.tp_timeframe is not None:
        if settings.strategy.tp_timeframe == settings.strategy.entry_timeframe:
            tp_frame = entry
        elif settings.strategy.tp_timeframe == settings.strategy.htf_timeframe:
            tp_frame = htf
        else:
            tp_raw = source.historical_candles(
                instrument.name,
                settings.strategy.tp_timeframe,
                warmup_start,
                end,
            )
            tp_frame = prepare_indicators(_ohlcv(tp_raw)) if not tp_raw.empty else None

    stats = {
        "decision_points": 0,
        "permission_blocks": 0,
        "strategy_candidates": 0,
        "candidates_with_entry_quotes": 0,
        "complete_outcomes": 0,
        "incomplete_outcomes": 0,
    }

    for candle_open in entry.index:
        decision_time = _utc((candle_open + pd.Timedelta(entry_delta)).to_pydatetime())
        if decision_time < start or decision_time >= end:
            continue
        stats["decision_points"] += 1

        permission = can_trade(
            instrument.name,
            decision_time,
            settings=settings.strategy,
            news_state=SimpleNamespace(
                events=news_events,
                stale=not news_authoritative,
            ),
        )
        if not permission.allowed:
            stats["permission_blocks"] += 1
            continue

        entry_slice = entry.loc[entry.index <= pd.Timestamp(decision_time)]
        htf_slice = htf.loc[htf.index <= pd.Timestamp(decision_time)]
        decision = evaluate_signal_frame(
            entry_slice,
            htf_slice,
            instrument=instrument,
            settings=settings.strategy,
            timestamp=decision_time,
        )
        if decision.signal is None:
            continue

        stats["strategy_candidates"] += 1
        quote_end = decision_time + timedelta(
            seconds=MAX_OUTCOME_HORIZON_SECONDS
            + config.outcome_lag_tolerance_seconds
        )
        quotes = source.historical_ticks(instrument.name, decision_time, quote_end)
        first_quote = _first_entry_quote(
            quotes,
            decision_time,
            config.entry_quote_tolerance_seconds,
        )
        if first_quote is None:
            _record_missing_quote_candidate(
                journal=journal,
                instrument=instrument,
                decision=decision,
                decision_time=decision_time,
                strategy_hash=strategy_hash,
            )
            continue

        stats["candidates_with_entry_quotes"] += 1
        price = PriceSnapshot(
            instrument.name,
            bid=float(first_quote.bid),
            ask=float(first_quote.ask),
            time=_utc(pd.Timestamp(first_quote.timestamp).to_pydatetime()),
        )
        entry_price = price.ask if decision.signal is Side.LONG else price.bid

        intent = build_signal_intent(
            entry_slice,
            htf_slice,
            instrument=instrument,
            settings=settings.strategy,
            entry_price=entry_price,
            timestamp=decision_time,
            entry_price_source="historical_mt5_executable_tick",
        )
        if intent is None:
            continue

        metadata = {
            **intent.metadata,
            "execution_cost_price": settings.strategy.execution_cost_pips_round_trip * instrument.pip_size,
            "historical_reconstruction": True,
            "historical_reconstruction_version": HISTORICAL_RECONSTRUCTION_VERSION,
        }
        if settings.strategy.tp_timeframe is not None and tp_frame is not None:
            tp_slice = tp_frame.loc[tp_frame.index <= pd.Timestamp(decision_time)]
            tp_row = last_closed_row(
                tp_slice,
                settings.strategy.tp_timeframe,
                timestamp=decision_time,
            )
            if tp_row is not None:
                tp_atr = _finite_or_none(tp_row.get("atr"))
                if tp_atr is not None and tp_atr > 0:
                    metadata["tp_atr"] = tp_atr
                    metadata["tp_timeframe"] = settings.strategy.tp_timeframe
        intent = replace(intent, metadata=metadata)
        exit_plan = risk.build_exit_plan(intent, instrument)
        candidate_id = _historical_candidate_id(intent, strategy_hash)

        neutral_portfolio = FxPortfolioState(
            equity=0.0,
            balance=0.0,
            margin_used=0.0,
            open_positions=0,
            account_currency=settings.risk.account_currency,
        )
        snapshot_payload: dict[str, Any] | None = None
        try:
            snapshot_payload = build_market_snapshot(
                candidate_id=candidate_id,
                intent=intent,
                instrument=instrument,
                price=price,
                entry_frame=entry_slice,
                timeframe=settings.strategy.entry_timeframe,
                portfolio=neutral_portfolio,
                exit_plan=exit_plan,
                observed_at=price.time,
                sessions=active_sessions(decision_time),
                news_events=news_events,
                news_stale=not news_authoritative,
                news_age_seconds=None,
                news_last_updated=None,
                news_source="historical_archive" if news_authoritative else "historical_news_unavailable",
                news_before_minutes=settings.strategy.news_blackout_before_minutes,
                news_after_minutes=settings.strategy.news_blackout_after_minutes,
            ).to_dict()
            # Historical conditional reconstruction does not replay a portfolio.
            exposure = snapshot_payload.get("current_exposure") or {}
            for key in (
                "open_positions",
                "portfolio_risk",
                "gross_exposure",
                "pair_exposure",
                "free_margin",
            ):
                exposure[key] = None
            exposure["currency_exposures"] = {}
        except Exception:
            snapshot_payload = None

        rejection_reason = _market_quality_rejection(
            intent=intent,
            exit_plan=exit_plan,
            instrument=instrument,
            price=price,
            settings=settings,
        )
        candidate, _ = journal.record_candidate(
            candidate_id=candidate_id,
            timestamp=decision_time,
            symbol=instrument.name,
            direction=intent.side.value,
            entry=entry_price,
            stop_loss=exit_plan.stop_loss if exit_plan else None,
            take_profit=exit_plan.take_profit if exit_plan else None,
            spread=price.ask - price.bid,
            atr=_finite_or_none(intent.signal_row.get("atr")),
            momentum=(snapshot_payload or {}).get("momentum"),
            trend_strength=_finite_or_none(intent.signal_row.get("adx")),
            news_risk=(snapshot_payload or {}).get("news_context") or {},
            strategy_signal=str(intent.metadata.get("decision") or decision.reason),
            executed=False,
            rejection_reason=rejection_reason,
            status="historical_reconstructed",
            payload={
                "market_snapshot": snapshot_payload,
                "signal_score": intent.score,
                "signal_time": decision_time.isoformat(),
                "execution_cost_pips_round_trip": settings.strategy.execution_cost_pips_round_trip,
                "historical_reconstruction": True,
                "historical_permission": permission.as_dict(),
                "portfolio_state_replayed": False,
            },
            strategy_hash=strategy_hash,
            code_version=f"historical-reconstruction-{HISTORICAL_RECONSTRUCTION_VERSION}",
            data_hash=_candidate_data_hash(intent, price),
        )

        tracker.seed(
            candidate=candidate,
            price=price,
            instrument=instrument,
            observed_at=price.time,
        )
        if snapshot_payload is None:
            journal.update_candidate_outcome(
                candidate_id,
                values={"data_quality": "degraded"},
                payload_update={"feature_snapshot_missing": True},
            )
        for quote in quotes.itertuples(index=False):
            quote_time = _utc(pd.Timestamp(quote.timestamp).to_pydatetime())
            if quote_time <= price.time:
                continue
            tracker.observe(
                candidate=candidate,
                price=PriceSnapshot(
                    instrument.name,
                    bid=float(quote.bid),
                    ask=float(quote.ask),
                    time=quote_time,
                ),
                instrument=instrument,
                observed_at=quote_time,
            )

        outcome = journal.find_candidate_outcome(candidate_id)
        if outcome is not None and outcome.status == "tracking":
            journal.update_candidate_outcome(
                candidate_id,
                values={
                    "status": "incomplete",
                    "completed_at": quote_end,
                    "data_quality": "degraded",
                },
                payload_update={"incomplete_reason": "historical_quote_coverage_incomplete"},
            )
            outcome = journal.find_candidate_outcome(candidate_id)
        if outcome is not None and outcome.status == "complete":
            stats["complete_outcomes"] += 1
        else:
            stats["incomplete_outcomes"] += 1

    return stats


def _record_missing_quote_candidate(
    *,
    journal: StructuredJournal,
    instrument: FxInstrument,
    decision: Any,
    decision_time: datetime,
    strategy_hash: str,
) -> None:
    key = (
        f"{instrument.name}:{decision.signal.value if decision.signal else 'NONE'}:"
        f"{decision_time.isoformat()}:{strategy_hash}:missing_quote"
    )
    candidate_id = "fxhist-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    journal.record_candidate(
        candidate_id=candidate_id,
        timestamp=decision_time,
        symbol=instrument.name,
        direction=decision.signal.value if decision.signal else "UNKNOWN",
        entry=0.0,
        spread=0.0,
        strategy_signal=decision.reason,
        executed=False,
        rejection_reason="historical_entry_quote_unavailable",
        status="historical_incomplete",
        payload={
            "historical_reconstruction": True,
            "market_snapshot": None,
        },
        strategy_hash=strategy_hash,
        code_version=f"historical-reconstruction-{HISTORICAL_RECONSTRUCTION_VERSION}",
    )


def _market_quality_rejection(
    *,
    intent: Any,
    exit_plan: Any,
    instrument: FxInstrument,
    price: PriceSnapshot,
    settings: FxBotSettings,
) -> str | None:
    spread = price.ask - price.bid
    atr = _finite_or_none(intent.signal_row.get("atr"))
    signal_close = _finite_or_none(intent.signal_row.get("close"))
    if spread <= 0 or atr is None or atr <= 0:
        return "volatility_or_spread_unavailable"
    if spread / atr > settings.strategy.max_spread_atr_ratio:
        return "spread_to_atr_filter"
    if signal_close is None:
        return "signal_close_unavailable"
    deviation_pips = abs(intent.entry_price - signal_close) / instrument.pip_size
    if deviation_pips > settings.strategy.max_entry_deviation_pips:
        return "entry_deviation_filter"
    if exit_plan is None:
        return "exit_plan_unavailable"
    if not reward_covers_spread(
        exit_plan,
        spread,
        settings.strategy.min_reward_to_spread_ratio,
    ):
        return "target_cost_filter"
    return None


def _first_entry_quote(
    quotes: pd.DataFrame,
    decision_time: datetime,
    tolerance_seconds: int,
) -> Any | None:
    if quotes.empty:
        return None
    decision = pd.Timestamp(_utc(decision_time))
    eligible = quotes.loc[
        (quotes["timestamp"] >= decision)
        & (quotes["timestamp"] <= decision + pd.Timedelta(seconds=tolerance_seconds))
    ]
    if eligible.empty:
        return None
    return next(eligible.sort_values("timestamp").itertuples(index=False))


def _events_for_symbol(events: list[NewsEvent], symbol: str) -> list[NewsEvent]:
    base, quote = split_instrument_name(symbol)
    currencies = {base, quote}
    return [event for event in events if event.currency.upper() in currencies]


def _ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"historical candle frame missing columns: {missing}")
    return frame.loc[:, required].copy()


def _historical_candidate_id(intent: Any, strategy_hash: str) -> str:
    stamp = _utc(intent.timestamp).isoformat()
    raw = f"{intent.instrument}:{intent.side.value}:{stamp}:{intent.entry_price:.8f}:{strategy_hash}"
    return "fxhist-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _candidate_data_hash(intent: Any, price: PriceSnapshot) -> str:
    payload = {
        "instrument": intent.instrument,
        "side": intent.side.value,
        "timestamp": _utc(intent.timestamp).isoformat(),
        "entry": intent.entry_price,
        "bid": price.bid,
        "ask": price.ask,
        "signal_row": intent.signal_row,
    }
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _settings_hash(settings: FxBotSettings) -> str:
    payload = {
        "strategy": asdict(settings.strategy),
        "risk_exit_assumptions": {
            "account_currency": settings.risk.account_currency,
        },
    }
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _finite_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_utc(value: str) -> datetime:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.to_pydatetime()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="UTC start, e.g. 2023-01-01")
    parser.add_argument("--end", required=True, help="UTC exclusive end, e.g. 2026-07-01")
    parser.add_argument("--symbols", default="", help="Comma-separated FX symbols; defaults to configured instruments")
    parser.add_argument(
        "--database-url",
        default="sqlite:///./data/fx_historical_candidates.db",
    )
    parser.add_argument(
        "--dataset-output",
        type=Path,
        default=Path("./data/training/candidate_training_v1.csv"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("./reports/fx_historical_reconstruction.json"),
    )
    parser.add_argument("--entry-quote-tolerance-seconds", type=int, default=30)
    parser.add_argument("--outcome-lag-tolerance-seconds", type=int, default=30)
    parser.add_argument(
        "--news-authoritative",
        action="store_true",
        help="Treat FX_NEWS_EVENTS_JSON/FILE as a complete historical archive for this period",
    )
    args = parser.parse_args()

    settings = settings_from_env()
    symbols = [
        normalize_instrument_name(item)
        for item in args.symbols.split(",")
        if item.strip()
    ] or settings.instruments
    events = load_news_events()
    source = Mt5Client(settings.broker)
    journal = StructuredJournal(args.database_url)
    try:
        report = reconstruct_historical_candidates(
            source=source,
            settings=settings,
            journal=journal,
            config=HistoricalReconstructionConfig(
                start=_parse_utc(args.start),
                end=_parse_utc(args.end),
                entry_quote_tolerance_seconds=args.entry_quote_tolerance_seconds,
                outcome_lag_tolerance_seconds=args.outcome_lag_tolerance_seconds,
            ),
            symbols=symbols,
            news_events=events,
            news_authoritative=args.news_authoritative,
        )
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        dataset, metadata = export_training_dataset(journal, args.dataset_output)
        print(json.dumps({
            "report": str(args.report_output),
            "dataset": str(dataset),
            "dataset_metadata": str(metadata),
            "totals": report["totals"],
        }, sort_keys=True))
    finally:
        journal.close()


if __name__ == "__main__":
    main()
