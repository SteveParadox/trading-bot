from __future__ import annotations

import numpy as np

from pathlib import Path
from typing import Any

from forex_agent.config import load_config
from forex_agent.data.ingestion import compute_r_multiple, compute_r_values, extract_session
from forex_agent.data.schemas import (
    ConfidenceLevel,
    CounterfactualResult,
    DiagnosticDimension,
    DiagnosticFactor,
    DiagnosticLevel,
    EvidencePackage,
    FailureCategory,
    HealthStatus,
    HypothesisStatus,
    MarketRegime,
    PerformanceMetrics,
    ResearchHypothesis,
    SimilarTradeResult,
    StrategyHealthScore,
    TradeDiagnostic,
    TradeFailureAnalysis,
    TradeRecord,
)
from forex_agent.log import get_logger

logger = get_logger(__name__)


def _get_r_values(trades: list[TradeRecord]) -> np.ndarray:
    """R-multiples for a batch of trades as a numpy array.

    Delegates to the canonical ``compute_r_values`` helper so there is a
    single source of truth for R-multiple math.
    """
    return np.array(compute_r_values(trades))


def _max_consecutive(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for v in mask:
        if v:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


class TradeAnalyst:
    def __init__(self, database_url: str, jsonl_path: str) -> None:
        self.database_url = database_url
        self.jsonl_path = jsonl_path
        self._trades: list[TradeRecord] = []
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load_trades()
            self._loaded = True

    def _load_trades(self) -> None:
        from forex_agent.data.ingestion import (
            load_trades_from_database,
            load_trades_from_jsonl,
        )

        jsonl_path = Path(self.jsonl_path)
        if jsonl_path.exists():
            try:
                self._trades = load_trades_from_jsonl(str(jsonl_path))
            except Exception as exc:  # noqa: BLE001 - defensive after internal handling
                logger.warning("Failed to load trades from JSONL %s: %s", jsonl_path, exc)
                self._trades = []

        if not self._trades and self.database_url:
            try:
                self._trades = load_trades_from_database(self.database_url)
            except Exception as exc:  # noqa: BLE001 - defensive after internal handling
                logger.warning("Failed to load trades from DB %s: %s", self.database_url, exc)
                self._trades = []

    @property
    def trades(self) -> list[TradeRecord]:
        self._ensure_loaded()
        return list(self._trades)

    def run_full_analysis(self) -> AnalysisReport:
        trades = self.trades
        metrics = self._compute_metrics(trades)
        health = self._compute_health(trades, metrics)
        failures = self._analyze_failures(trades)
        winners = self._analyze_winners(trades)
        anomalies = self._detect_anomalies(trades)
        regime = self._analyze_regime(trades)
        risk = self._analyze_risk(trades)
        execution = self._analyze_execution(trades)
        patterns = self._detect_patterns(trades, failures)
        alerts = self._generate_alerts(trades, metrics, anomalies)

        # Enhanced: bootstrap CI for expectancy
        from forex_agent.analysis.statistics_enhanced import bootstrap_expectancy_ci
        bootstrap_ci = bootstrap_expectancy_ci(trades)

        return AnalysisReport(
            total_trades=len(trades),
            metrics=metrics,
            health=health,
            failures=failures,
            winner_analyses=winners,
            anomalies=anomalies,
            regime_analysis=regime,
            risk_analysis=risk,
            execution_analysis=execution,
            recurring_patterns=patterns,
            alerts=alerts,
            bootstrap_expectancy_ci=bootstrap_ci,
        )

    def get_alerts(self) -> list[Any]:
        trades = self.trades
        metrics = self._compute_metrics(trades)
        anomalies = self._detect_anomalies(trades)
        return self._generate_alerts(trades, metrics, anomalies)

    def _generate_alerts(
        self,
        trades: list[TradeRecord],
        metrics: PerformanceMetrics,
        anomalies: list[dict[str, Any]],
    ) -> list[Any]:
        from forex_agent.analysis.alerts import AlertEngine

        config = load_config()
        engine = AlertEngine(config.alert_thresholds)
        return engine.check(trades, metrics, anomalies)

    def analyze_single_trade(self, trade_id: str) -> TradeFailureAnalysis | None:
        trades = self.trades
        trade = next((t for t in trades if t.trade_id == trade_id), None)
        if trade is None:
            return None
        return self._classify_trade(trade, trades)

    def get_strategy_health(self) -> StrategyHealthScore:
        trades = self.trades
        metrics = self._compute_metrics(trades)
        return self._compute_health(trades, metrics)

    def get_anomalies(self) -> list[dict[str, Any]]:
        return self._detect_anomalies(self.trades)

    def get_regime_analysis(self) -> dict[str, Any]:
        return self._analyze_regime(self.trades)

    # -----------------------------------------------------------------
    # Enhanced diagnostic methods (spec sections 4-13)
    # -----------------------------------------------------------------

    def diagnose_trade(self, trade_id: str) -> TradeDiagnostic | None:
        """Multi-factor diagnostic for a single trade (spec Section 4)."""
        trades = self.trades
        trade = next((t for t in trades if t.trade_id == trade_id), None)
        if trade is None:
            return None
        return self._build_diagnostic(trade, trades)

    def build_evidence_package(self, trade_id: str) -> EvidencePackage | None:
        """Structured evidence package for a trade (spec Section 6)."""
        from forex_agent.analysis.evidence import build_evidence_package
        trades = self.trades
        trade = next((t for t in trades if t.trade_id == trade_id), None)
        if trade is None:
            return None
        return build_evidence_package(trade, trades)

    def find_similar(self, trade_id: str) -> SimilarTradeResult | None:
        """Find historically comparable trades (spec Section 7)."""
        from forex_agent.analysis.similarity import find_similar_trades
        trades = self.trades
        trade = next((t for t in trades if t.trade_id == trade_id), None)
        if trade is None:
            return None
        return find_similar_trades(trade, trades)

    def critique_finding(
        self, finding_text: str, trade_ids: list[str] | None = None,
        initial_confidence: float = 0.7,
    ) -> Any:
        """Run the critic/anti-bias agent on a finding (spec Section 13)."""
        from forex_agent.analysis.critic import assess_finding
        trades = self.trades
        matched = None
        if trade_ids:
            matched = [t for t in trades if t.trade_id in trade_ids]
        return assess_finding(finding_text, trades, matched, initial_confidence)

    def analyze_all_trades(self) -> list[TradeDiagnostic]:
        """Diagnose ALL trades (winners and losers) per spec Section 5."""
        trades = self.trades
        return [self._build_diagnostic(t, trades) for t in trades if t.exit_price is not None]

    def get_research_memory_summary(self) -> dict[str, Any]:
        """Get summary of research memory."""
        from forex_agent.agent.research_memory import ResearchMemory
        config = load_config()
        mem = ResearchMemory(config.research_memory_path)
        return mem.summary()

    def explain_trade(self, trade_id: str) -> dict[str, Any] | None:
        """Full LLM-powered explanation for a trade.

        Builds diagnostic + evidence + critic, then sends to the LLM reasoner.
        Returns dict with explanation, provider, model, and fallback info.
        """
        from forex_agent.analysis.llm_reasoner import generate_llm_explanation
        from forex_agent.analysis.evidence import build_evidence_package
        from forex_agent.analysis.critic import assess_finding

        trades = self.trades
        trade = next((t for t in trades if t.trade_id == trade_id), None)
        if trade is None:
            return None

        diagnostic = self._build_diagnostic(trade, trades)
        evidence = build_evidence_package(trade, trades)

        # Build critic assessment for this trade's primary diagnosis
        critic = assess_finding(
            diagnostic.primary_diagnosis,
            trades,
            initial_confidence=diagnostic.confidence,
        )

        result = generate_llm_explanation(trade, diagnostic, evidence, critic)
        result["diagnostic"] = diagnostic.to_dict()
        result["evidence"] = evidence.to_dict()
        result["critic"] = critic.to_dict()
        return result

    def _build_diagnostic(
        self, trade: TradeRecord, all_trades: list[TradeRecord]
    ) -> TradeDiagnostic:
        """Build a multi-factor diagnostic for a single trade."""
        r_mult = compute_r_multiple(trade)
        is_win = trade.pnl > 0 if trade.exit_price is not None else False
        outcome = "win" if is_win else ("open" if trade.exit_price is None else "loss")

        factors: list[DiagnosticFactor] = []
        protective: list[DiagnosticFactor] = []
        observations: list[str] = []
        hypotheses: list[str] = []
        unknowns: list[str] = []

        closed = [t for t in all_trades if t.exit_price is not None]
        same_symbol = [t for t in closed if t.symbol == trade.symbol]

        # --- SIGNAL dimension ---
        if trade.signal_strength is not None:
            if trade.signal_strength >= 0.8:
                protective.append(DiagnosticFactor(
                    dimension=DiagnosticDimension.SIGNAL,
                    label="strong_setup",
                    level=DiagnosticLevel.OBSERVATION,
                    description=f"Signal strength {trade.signal_strength:.2f} is strong",
                ))
            elif trade.signal_strength < 0.4:
                factors.append(DiagnosticFactor(
                    dimension=DiagnosticDimension.SIGNAL,
                    label="weak_setup",
                    level=DiagnosticLevel.OBSERVATION,
                    description=f"Signal strength {trade.signal_strength:.2f} is weak",
                ))
        else:
            unknowns.append("Signal strength not recorded")

        # --- MARKET REGIME dimension ---
        if trade.regime is not None:
            regime_trades = [t for t in closed if t.regime == trade.regime]
            if len(regime_trades) >= 5:
                regime_r = compute_r_values(regime_trades)
                regime_exp = float(np.mean(regime_r)) if regime_r else 0.0
                observations.append(
                    f"Regime {trade.regime.value}: {len(regime_trades)} historical trades, "
                    f"expectancy {regime_exp:.2f}R"
                )
                if regime_exp < -0.2 and not is_win:
                    factors.append(DiagnosticFactor(
                        dimension=DiagnosticDimension.MARKET_REGIME,
                        label="regime_underperformance",
                        level=DiagnosticLevel.ASSOCIATION,
                        description=f"Strategy historically underperforms in {trade.regime.value}",
                        sample_size=len(regime_trades),
                        effect_size=regime_exp,
                    ))
                elif regime_exp > 0.2 and is_win:
                    protective.append(DiagnosticFactor(
                        dimension=DiagnosticDimension.MARKET_REGIME,
                        label="regime_alignment",
                        level=DiagnosticLevel.ASSOCIATION,
                        description=f"Trade aligned with favorable {trade.regime.value} regime",
                        sample_size=len(regime_trades),
                        effect_size=regime_exp,
                    ))
        else:
            unknowns.append("Market regime not recorded")

        # --- EXECUTION dimension ---
        if trade.spread_at_entry > 0:
            spreads = [t.spread_at_entry for t in same_symbol if t.spread_at_entry > 0]
            if spreads:
                med_spread = float(np.median(spreads))
                ratio = trade.spread_at_entry / med_spread if med_spread > 0 else 0
                observations.append(
                    f"Spread at entry: {trade.spread_at_entry:.1f} pips "
                    f"(median for {trade.symbol}: {med_spread:.1f} pips, ratio: {ratio:.1f}x)"
                )
                if ratio > 2.0:
                    factors.append(DiagnosticFactor(
                        dimension=DiagnosticDimension.EXECUTION,
                        label="excessive_spread",
                        level=DiagnosticLevel.OBSERVATION,
                        description=f"Spread was {ratio:.1f}x the historical median",
                        sample_size=len(spreads),
                    ))
        if trade.slippage_pips > 1.0:
            factors.append(DiagnosticFactor(
                dimension=DiagnosticDimension.EXECUTION,
                label="slippage",
                level=DiagnosticLevel.OBSERVATION,
                description=f"Slippage of {trade.slippage_pips:.1f} pips",
            ))

        # --- RISK dimension ---
        if trade.risk_amount > 0 and trade.account_balance > 0:
            risk_pct = trade.risk_amount / trade.account_balance
            if risk_pct > 0.02:
                factors.append(DiagnosticFactor(
                    dimension=DiagnosticDimension.RISK,
                    label="excessive_position_size",
                    level=DiagnosticLevel.OBSERVATION,
                    description=f"Risk per trade {risk_pct:.2%} exceeds 2% guideline",
                ))

        if trade.entry_price > 0 and trade.stop_loss > 0:
            sl_dist = abs(trade.entry_price - trade.stop_loss)
            if trade.take_profit is not None:
                tp_dist = abs(trade.take_profit - trade.entry_price)
                rr = tp_dist / sl_dist if sl_dist > 0 else 0
                if rr < 1.0:
                    factors.append(DiagnosticFactor(
                        dimension=DiagnosticDimension.RISK,
                        label="poor_reward_risk",
                        level=DiagnosticLevel.OBSERVATION,
                        description=f"Risk/reward ratio 1:{rr:.1f} is unfavorable",
                    ))

        if not trade.follow_stop_loss or not trade.follow_take_profit:
            factors.append(DiagnosticFactor(
                dimension=DiagnosticDimension.TRADE_MANAGEMENT,
                label="discretionary_intervention",
                level=DiagnosticLevel.OBSERVATION,
                description=f"SL followed: {trade.follow_stop_loss}, TP followed: {trade.follow_take_profit}",
            ))

        # --- TIMING dimension ---
        if trade.entry_time is not None:
            session = extract_session(trade.entry_time)
            session_trades = [t for t in same_symbol if extract_session(t.entry_time) == session]
            if len(session_trades) >= 5:
                sess_r = compute_r_values(session_trades)
                sess_exp = float(np.mean(sess_r)) if sess_r else 0.0
                observations.append(
                    f"Session {session}: {len(session_trades)} historical trades, "
                    f"expectancy {sess_exp:.2f}R"
                )

        # --- Primary diagnosis ---
        primary_dim = DiagnosticDimension.UNKNOWN
        primary_diag = "Insufficient evidence for definitive diagnosis"
        if factors:
            dim_counts: dict[DiagnosticDimension, int] = {}
            for f in factors:
                dim_counts[f.dimension] = dim_counts.get(f.dimension, 0) + 1
            primary_dim = max(dim_counts, key=dim_counts.get)  # type: ignore[arg-type]
            matching = [f for f in factors if f.dimension == primary_dim]
            primary_diag = matching[0].description if matching else str(primary_dim.value)

        if not factors and not protective:
            primary_diag = "No specific factors detected; outcome consistent with normal variance" if is_win else "No specific factors detected; loss consistent with normal variance"

        # --- Confidence ---
        n_evidence = len(factors) + len(protective) + len(observations)
        has_sample = any(f.sample_size >= 10 for f in factors + protective)
        confidence = min(1.0, (n_evidence * 0.15) + (0.2 if has_sample else 0.0))

        # --- Evidence level ---
        if any(f.level == DiagnosticLevel.CONCLUSION for f in factors + protective):
            evidence_level = DiagnosticLevel.CONCLUSION
        elif any(f.level == DiagnosticLevel.HYPOTHESIS for f in factors + protective):
            evidence_level = DiagnosticLevel.HYPOTHESIS
        elif any(f.level == DiagnosticLevel.ASSOCIATION for f in factors + protective):
            evidence_level = DiagnosticLevel.ASSOCIATION
        else:
            evidence_level = DiagnosticLevel.OBSERVATION

        return TradeDiagnostic(
            trade_id=trade.trade_id,
            outcome=outcome,
            primary_diagnosis=primary_diag,
            primary_dimension=primary_dim,
            contributing_factors=factors,
            protective_factors=protective,
            observations=observations,
            hypotheses=hypotheses,
            confidence=round(confidence, 3),
            evidence_level=evidence_level,
            sample_sizes={f.label: f.sample_size for f in factors + protective if f.sample_size > 0},
            statistical_support={},
            counterfactuals=[],
            unknowns=unknowns,
        )

    def _compute_metrics(self, trades: list[TradeRecord]) -> PerformanceMetrics:
        closed = [t for t in trades if t.exit_price is not None]
        if not closed:
            return PerformanceMetrics()

        pnls = np.array([t.pnl for t in closed])
        r_values = _get_r_values(closed)

        winners = pnls[pnls > 0]
        losers = pnls[pnls < 0]

        total = len(closed)
        win_count = int(np.sum(pnls > 0))
        lose_count = int(np.sum(pnls < 0))
        win_rate = win_count / total if total > 0 else 0.0

        avg_win = float(np.mean(winners)) if len(winners) > 0 else 0.0
        avg_loss = float(np.mean(losers)) if len(losers) > 0 else 0.0
        total_pnl = float(np.sum(pnls))

        gross_wins = float(np.sum(winners)) if len(winners) > 0 else 0.0
        gross_losses = float(np.abs(np.sum(losers))) if len(losers) > 0 else 0.0
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else (
            float("inf") if gross_wins > 0 else 0.0
        )

        expectancy = float(np.mean(r_values)) if len(r_values) > 0 else 0.0

        equity = np.cumsum(np.concatenate(([0.0], pnls)))
        running_max = np.maximum.accumulate(equity)
        drawdowns = equity - running_max
        max_dd = float(np.min(drawdowns))
        peak = float(np.max(running_max))
        max_dd_pct = abs(max_dd / peak) if peak > 0 else 0.0

        sharpe = self._compute_sharpe(r_values)

        # Sortino ratio (only penalizes downside)
        sortino = self._compute_sortino(r_values)

        avg_rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
        cons_wins = _max_consecutive(pnls > 0)
        cons_losses = _max_consecutive(pnls < 0)

        durations = [
            (t.exit_time - t.entry_time).total_seconds() / 60.0
            for t in closed
            if t.exit_time is not None
        ]
        median_duration = float(np.median(durations)) if durations else 0.0

        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
        recovery_factor = (total_pnl / abs(max_dd)) if max_dd < 0 else 0.0

        # Trade frequency (per day based on date span)
        trade_freq = self._compute_trade_frequency(closed)

        # MFE / MAE in R terms
        mfe_r_values = []
        mae_r_values = []
        for t in closed:
            if t.mfe is not None and t.stop_loss:
                risk = abs(t.entry_price - t.stop_loss)
                if risk > 0:
                    mfe_r_values.append(t.mfe / risk)
            if t.mae is not None and t.stop_loss:
                risk = abs(t.entry_price - t.stop_loss)
                if risk > 0:
                    mae_r_values.append(t.mae / risk)

        return PerformanceMetrics(
            total_trades=total,
            winning_trades=win_count,
            losing_trades=lose_count,
            win_rate=win_rate,
            loss_rate=1.0 - win_rate if total > 0 else 0.0,
            profit_factor=profit_factor,
            expectancy=expectancy,
            average_win=avg_win,
            average_loss=avg_loss,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            total_pnl=total_pnl,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            avg_r_multiple=float(np.mean(r_values)) if len(r_values) > 0 else 0.0,
            median_r_multiple=float(np.median(r_values)) if len(r_values) > 0 else 0.0,
            avg_rr_ratio=avg_rr,
            payoff_ratio=payoff_ratio,
            largest_win=float(np.max(winners)) if len(winners) > 0 else 0.0,
            largest_loss=float(np.min(losers)) if len(losers) > 0 else 0.0,
            consecutive_wins=cons_wins,
            consecutive_losses=cons_losses,
            avg_duration_minutes=float(np.mean(durations)) if durations else 0.0,
            median_duration_minutes=median_duration,
            recovery_factor=recovery_factor,
            trade_frequency_per_day=trade_freq,
            avg_mfe_r=float(np.mean(mfe_r_values)) if mfe_r_values else 0.0,
            avg_mae_r=float(np.mean(mae_r_values)) if mae_r_values else 0.0,
        )

    @staticmethod
    def _compute_sharpe(r_values: np.ndarray) -> float:
        if len(r_values) < 2:
            return 0.0
        mean = float(np.mean(r_values))
        std = float(np.std(r_values, ddof=1))
        if std == 0:
            return 0.0
        return mean / std

    @staticmethod
    def _compute_sortino(r_values: np.ndarray) -> float:
        if len(r_values) < 2:
            return 0.0
        mean = float(np.mean(r_values))
        downside = r_values[r_values < 0]
        if len(downside) == 0:
            return 0.0
        downside_std = float(np.std(downside, ddof=1))
        if downside_std == 0:
            return 0.0
        return mean / downside_std

    @staticmethod
    def _compute_trade_frequency(trades: list[TradeRecord]) -> float:
        if len(trades) < 2:
            return 0.0
        from datetime import datetime

        times = [t.entry_time for t in trades if t.entry_time is not None]
        if len(times) < 2:
            return 0.0
        span_days = max(
            (max(times) - min(times)).total_seconds() / 86400.0, 1.0
        )
        return len(trades) / span_days

    def _compute_health(
        self,
        trades: list[TradeRecord],
        metrics: PerformanceMetrics,
    ) -> StrategyHealthScore:
        config = load_config()

        if metrics.total_trades < config.min_sample_size:
            return StrategyHealthScore(
                score=0.0,
                components={"insufficient_data": 0.0},
                grade="F",
                recommendations=[
                    f"Need at least {config.min_sample_size} trades for analysis"
                ],
                health_status=HealthStatus.INSUFFICIENT_DATA,
                explanation=(
                    f"Insufficient data: only {metrics.total_trades} closed "
                    f"trades; need at least {config.min_sample_size}."
                ),
            )

        t = config.alert_thresholds
        recommendations: list[str] = []
        components: dict[str, float] = {}
        total_weight = 0.0
        weighted_sum = 0.0

        wr_score = min(100.0, (metrics.win_rate / max(t.get("min_win_rate", 0.35), 0.01)) * 80.0)
        components["win_rate"] = wr_score
        weighted_sum += wr_score * 15.0
        total_weight += 15.0

        pf_score = min(100.0, (metrics.profit_factor / max(t.get("min_profit_factor", 0.8), 0.01)) * 80.0)
        components["profit_factor"] = pf_score
        weighted_sum += pf_score * 20.0
        total_weight += 20.0

        exp_threshold = abs(t.get("min_expectancy_r", -0.3))
        exp_score = min(100.0, max(0.0, (metrics.expectancy + exp_threshold) / (2 * exp_threshold) * 100.0))
        components["expectancy"] = exp_score
        weighted_sum += exp_score * 25.0
        total_weight += 25.0

        dd_threshold = t.get("max_drawdown_pct", 0.10)
        if dd_threshold > 0 and metrics.max_drawdown_pct > 0:
            dd_score = max(0.0, (1.0 - metrics.max_drawdown_pct / dd_threshold) * 100.0)
        else:
            dd_score = 100.0
        dd_score = min(100.0, max(0.0, dd_score))
        components["drawdown"] = dd_score
        weighted_sum += dd_score * 20.0
        total_weight += 20.0

        cl_threshold = t.get("max_consecutive_losses", 5)
        if cl_threshold > 0:
            cl_score = max(0.0, (1.0 - metrics.consecutive_losses / cl_threshold) * 100.0)
        else:
            cl_score = 100.0
        cl_score = min(100.0, max(0.0, cl_score))
        components["consecutive_losses"] = cl_score
        weighted_sum += cl_score * 10.0
        total_weight += 10.0

        recency = self._compute_recency_score(trades)
        components["recent_performance"] = recency
        weighted_sum += recency * 10.0
        total_weight += 10.0

        score = weighted_sum / total_weight if total_weight > 0 else 0.0

        if metrics.win_rate < t.get("min_win_rate", 0.35):
            recommendations.append(f"Win rate ({metrics.win_rate:.1%}) below threshold")
        if metrics.profit_factor < t.get("min_profit_factor", 0.8):
            recommendations.append(f"Profit factor ({metrics.profit_factor:.2f}) below threshold")
        if metrics.expectancy < t.get("min_expectancy_r", -0.3):
            recommendations.append(f"Expectancy ({metrics.expectancy:.2f}R) is negative")
        if metrics.max_drawdown_pct > dd_threshold:
            recommendations.append(f"Drawdown ({metrics.max_drawdown_pct:.1%}) exceeds threshold")
        if metrics.consecutive_losses > cl_threshold:
            recommendations.append(
                f"Consecutive losses ({metrics.consecutive_losses}) exceed threshold"
            )

        # --- Health status classification ---
        status, status_reason = self._classify_health_status(
            metrics, recommendations, config.health_thresholds
        )

        health = StrategyHealthScore(
            score=score,
            components=components,
            grade="F",
            recommendations=recommendations,
            health_status=status,
            explanation=status_reason,
        )
        health.grade = health.compute_grade()
        return health

    @staticmethod
    def _classify_health_status(
        metrics: PerformanceMetrics,
        recommendations: list[str],
        health: dict[str, float] | None = None,
    ) -> tuple[HealthStatus, str]:
        """Classify strategy health into one of the spec statuses.

        The classification follows a documented, reproducible method based on
        the current health score and the count of crossed warning thresholds,
        rather than an arbitrary label. Thresholds come from configuration
        (``health_thresholds``) so the method can be tuned without touching
        code.
        """
        thresholds = health or load_config().health_thresholds
        min_sample = int(thresholds.get("critical_min_sample", 20))
        critical_exp = thresholds.get("critical_expectancy", -0.5)
        critical_pf = thresholds.get("critical_profit_factor", 0.7)
        critical_dd = thresholds.get("critical_max_drawdown_pct", 0.15)
        normal_exp = thresholds.get("normal_variance_expectancy", 0.15)

        if metrics.total_trades < min_sample:
            return (
                HealthStatus.INSUFFICIENT_DATA,
                f"Only {metrics.total_trades} closed trades in sample.",
            )

        crossed = len(recommendations)

        # CRITICAL: expectancy strongly negative or extreme drawdown
        if metrics.expectancy < critical_exp and metrics.profit_factor < critical_pf:
            return (
                HealthStatus.CRITICAL,
                "Expectancy strongly negative with low profit factor; "
                "edge may be structurally impaired.",
            )
        if metrics.max_drawdown_pct > critical_dd:
            return (
                HealthStatus.CRITICAL,
                f"Drawdown of {metrics.max_drawdown_pct:.1%} exceeds "
                f"{critical_dd:.0%} critical threshold.",
            )

        # DEGRADED: multiple thresholds crossed
        if crossed >= 3:
            return (
                HealthStatus.DEGRADED,
                f"{crossed} warning thresholds crossed: {'; '.join(recommendations[:3])}.",
            )

        # WATCH: one or two thresholds crossed
        if crossed >= 1:
            return (
                HealthStatus.WATCH,
                f"{crossed} warning threshold(s) crossed: {'; '.join(recommendations)}.",
            )

        # NORMAL VARIANCE: metrics within thresholds but not clearly healthy
        if metrics.expectancy < normal_exp:
            return (
                HealthStatus.NORMAL_VARIANCE,
                f"Expectancy of {metrics.expectancy:.2f}R is positive but "
                "modest; recent drawdowns within normal variance.",
            )

        # HEALTHY
        return (
            HealthStatus.HEALTHY,
            f"Expectancy {metrics.expectancy:.2f}R, profit factor "
            f"{metrics.profit_factor:.2f}, drawdown within tolerance. "
            "No thresholds crossed.",
        )

    @staticmethod
    def _compute_recency_score(trades: list[TradeRecord]) -> float:
        if len(trades) < 6:
            return 50.0
        mid = len(trades) // 2
        first = _get_r_values(trades[:mid])
        second = _get_r_values(trades[mid:])
        if len(second) == 0:
            return 50.0
        recent_exp = float(np.mean(second))
        older_exp = float(np.mean(first)) if len(first) > 0 else recent_exp
        if older_exp == 0:
            return 60.0 if recent_exp > 0 else 40.0
        ratio = recent_exp / abs(older_exp)
        return min(100.0, max(0.0, ratio * 50.0 + 50.0))

    def _classify_trade(
        self,
        trade: TradeRecord,
        all_trades: list[TradeRecord],
    ) -> TradeFailureAnalysis:
        r_mult = compute_r_multiple(trade)
        is_win = trade.pnl > 0 if trade.exit_price is not None else False

        if is_win:
            return TradeFailureAnalysis(
                trade_id=trade.trade_id,
                failure_category=FailureCategory.UNKNOWN,
                confidence=ConfidenceLevel.HIGH,
                description="Trade was a winner - no failure analysis needed",
                contributing_factors=[],
                counterfactual="",
                historical_comparison="",
                recommended_action="No action needed",
            )

        factors: list[str] = []
        category = FailureCategory.UNKNOWN
        confidence = ConfidenceLevel.LOW
        description = ""
        counterfactual = ""
        historical = ""

        if trade.slippage_pips > 3.0:
            factors.append(f"High slippage: {trade.slippage_pips:.1f} pips")
            category = FailureCategory.EXECUTION_ERROR
            confidence = ConfidenceLevel.MEDIUM
            description = f"Trade experienced excessive slippage of {trade.slippage_pips:.1f} pips"

        if trade.spread_at_entry > 3.0:
            factors.append(f"High spread at entry: {trade.spread_at_entry:.1f} pips")
            if category == FailureCategory.UNKNOWN:
                category = FailureCategory.EXECUTION_ERROR
                confidence = ConfidenceLevel.MEDIUM
                description = f"Trade entered with elevated spread of {trade.spread_at_entry:.1f} pips"

        if trade.regime is not None:
            regime_trades = [
                t for t in all_trades
                if t.regime == trade.regime and t.exit_price is not None
            ]
            if len(regime_trades) >= 5:
                regime_r = _get_r_values(regime_trades)
                if len(regime_r) > 0:
                    regime_exp = float(np.mean(regime_r))
                    if regime_exp < -0.2:
                        factors.append(
                            f"Poor performance in regime {trade.regime.value} "
                            f"(expectancy: {regime_exp:.2f}R, n={len(regime_trades)})"
                        )
                        if category == FailureCategory.UNKNOWN:
                            category = FailureCategory.REGIME_MISMATCH
                            confidence = ConfidenceLevel.MEDIUM
                            description = f"Strategy historically underperforms in {trade.regime.value} regime"
                        historical = (
                            f"In regime {trade.regime.value}: {len(regime_trades)} trades, "
                            f"expectancy {regime_exp:.2f}R"
                        )

        if trade.risk_amount > 0 and trade.account_balance > 0:
            risk_pct = trade.risk_amount / trade.account_balance
            if risk_pct > 0.02:
                factors.append(f"Risk per trade {risk_pct:.2%} exceeds 2% guideline")
                if category == FailureCategory.UNKNOWN:
                    category = FailureCategory.RISK_MISMANAGEMENT
                    confidence = ConfidenceLevel.LOW
                    description = "Position risk exceeded recommended guidelines"

        if trade.entry_price > 0 and trade.stop_loss > 0:
            sl_distance = abs(trade.entry_price - trade.stop_loss)
            entry_range = abs(trade.entry_price) * 0.005
            if sl_distance < entry_range * 0.2:
                factors.append("Stop loss may be too tight")
                if category == FailureCategory.UNKNOWN:
                    category = FailureCategory.RISK_MISMANAGEMENT
                    confidence = ConfidenceLevel.LOW
                    description = "Stop loss appears excessively tight relative to entry"

        if trade.exit_price is not None and trade.stop_loss is not None and trade.take_profit is not None:
            if trade.direction == "LONG":
                sl_dist = abs(trade.entry_price - trade.stop_loss)
                tp_dist = abs(trade.take_profit - trade.entry_price)
            else:
                sl_dist = abs(trade.stop_loss - trade.entry_price)
                tp_dist = abs(trade.entry_price - trade.take_profit)

            if sl_dist > 0 and tp_dist / sl_dist < 1.0:
                factors.append(f"Poor risk/reward ratio: 1:{tp_dist/sl_dist:.1f}")
                if category == FailureCategory.UNKNOWN:
                    category = FailureCategory.RISK_MISMANAGEMENT
                    confidence = ConfidenceLevel.LOW
                    description = "Risk/reward ratio was unfavorable at entry"

        if r_mult is not None and r_mult < -1.5:
            factors.append(f"Large loss: {r_mult:.2f}R")
            if trade.pnl < -100:
                factors.append(f"Absolute P&L loss: ${trade.pnl:.2f}")

        if not factors:
            description = "Trade loss consistent with normal statistical variance"
            category = FailureCategory.MARKET_CONDITION
            confidence = ConfidenceLevel.LOW
            factors.append("No specific failure pattern detected")

        if category == FailureCategory.UNKNOWN:
            category = FailureCategory.MARKET_CONDITION
            confidence = ConfidenceLevel.LOW
            if not description:
                description = "Insufficient evidence to determine specific cause"

        similar_losses = [
            t for t in all_trades
            if t.trade_id != trade.trade_id
            and t.symbol == trade.symbol
            and t.direction == trade.direction
            and t.exit_price is not None
            and t.pnl < 0
        ]
        if similar_losses:
            similar_r = _get_r_values(similar_losses)
            if len(similar_r) > 0:
                historical = (
                    f"Found {len(similar_losses)} similar losing trades in {trade.symbol} "
                    f"{trade.direction}: avg R = {float(np.mean(similar_r)):.2f}"
                )

        counterfactual = self._generate_counterfactual(trade, r_mult)

        recommended = self._recommend_action(category, confidence)

        return TradeFailureAnalysis(
            trade_id=trade.trade_id,
            failure_category=category,
            confidence=confidence,
            description=description,
            contributing_factors=factors,
            counterfactual=counterfactual,
            historical_comparison=historical,
            recommended_action=recommended,
        )

    @staticmethod
    def _generate_counterfactual(
        trade: TradeRecord,
        r_mult: float | None,
    ) -> str:
        parts: list[str] = []

        if trade.exit_price is not None and trade.take_profit is not None and trade.stop_loss is not None:
            if trade.direction == "LONG":
                reached_tp = trade.take_profit >= trade.entry_price
            else:
                reached_tp = trade.take_profit <= trade.entry_price
            parts.append(f"Take profit was set at {trade.take_profit}")

        if trade.entry_time is not None and trade.exit_time is not None:
            duration_min = (trade.exit_time - trade.entry_time).total_seconds() / 60.0
            parts.append(f"Trade lasted {duration_min:.0f} minutes")

        if r_mult is not None:
            if r_mult > -0.5:
                parts.append("Loss was modest (less than -0.5R) - within normal variance")
            elif r_mult < -1.0:
                parts.append("Loss exceeded -1R, suggesting stop was hit at full distance")

        if not parts:
            return "Insufficient data for counterfactual analysis"
        return "; ".join(parts)

    @staticmethod
    def _recommend_action(category: FailureCategory, confidence: ConfidenceLevel) -> str:
        actions = {
            FailureCategory.EXECUTION_ERROR: "Monitor execution quality; consider broker evaluation",
            FailureCategory.REGIME_MISMATCH: "Consider regime filter; gather more data before acting",
            FailureCategory.RISK_MISMANAGEMENT: "Review position sizing and stop placement rules",
            FailureCategory.MARKET_CONDITION: "Normal variance; continue monitoring",
            FailureCategory.EMOTIONAL_TRADE: "Review trading journal; consider process improvements",
            FailureCategory.INVALID_SETUP: "Review entry rules; ensure checklist compliance",
            FailureCategory.UNKNOWN: "Insufficient evidence; continue data collection",
        }
        action = actions.get(category, "Continue monitoring")
        if confidence == ConfidenceLevel.LOW:
            action += " (low confidence - gather more evidence)"
        return action

    def _analyze_failures(self, trades: list[TradeRecord]) -> list[TradeFailureAnalysis]:
        """Analyze all losing trades (backward-compatible with existing code)."""
        losses = [t for t in trades if t.exit_price is not None and t.pnl < 0]
        return [self._classify_trade(t, trades) for t in losses]

    def _analyze_winners(
        self, trades: list[TradeRecord]
    ) -> list[dict[str, Any]]:
        """Analyze winners per spec Section 5 - understand what worked."""
        closed = [t for t in trades if t.exit_price is not None]
        winners = [t for t in closed if t.pnl > 0]
        if not winners:
            return []

        analyses: list[dict[str, Any]] = []
        r_winners = compute_r_values(winners)
        mean_r_winners = float(np.mean(r_winners)) if r_winners else 0.0

        for t in winners:
            r = compute_r_multiple(t)
            if r is None:
                continue

            # Classify winner quality
            if r >= 2.0:
                quality = "exceptional"
            elif r >= 1.0:
                quality = "target_hit"
            elif r >= 0.5:
                quality = "moderate"
            else:
                quality = "marginal"

            factors: list[str] = []

            # Regime alignment
            if t.regime is not None:
                regime_trades = [w for w in winners if w.regime == t.regime]
                if len(regime_trades) >= 3:
                    factors.append(f"Aligned with {t.regime.value} regime")

            # Execution quality
            if t.spread_at_entry > 0:
                same_sym = [c for c in closed if c.symbol == t.symbol and c.spread_at_entry > 0]
                if same_sym:
                    med = float(np.median([s.spread_at_entry for s in same_sym]))
                    if t.spread_at_entry <= med:
                        factors.append("Favorable spread at entry")

            # MFE analysis
            if t.mfe is not None and t.mae is not None and t.mae > 0:
                efficiency = t.mfe / t.mae if t.mae > 0 else 0
                if efficiency > 2.0:
                    factors.append(f"High efficiency (MFE/MAE={efficiency:.1f})")

            analyses.append({
                "trade_id": t.trade_id,
                "symbol": t.symbol,
                "direction": t.direction,
                "r_multiple": r,
                "quality": quality,
                "factors": factors,
                "regime": t.regime.value if t.regime else "unknown",
            })

        return analyses

    def _detect_anomalies(self, trades: list[TradeRecord]) -> list[dict[str, Any]]:
        """Detect anomalies using the full anomaly-detection toolkit.

        Keeps the built-in detectors (R outliers, losing streaks, drawdown
        concentration, spread) and layers the newer detectors from
        ``anomaly_detection`` module on top.
        """
        anomalies: list[dict[str, Any]] = []
        closed = [t for t in trades if t.exit_price is not None]
        if len(closed) < 10:
            return anomalies

        r_values = _get_r_values(closed)
        if len(r_values) < 10:
            return anomalies

        mean_r = float(np.mean(r_values))
        std_r = float(np.std(r_values, ddof=1))

        if std_r > 0:
            z_scores = (r_values - mean_r) / std_r
            outlier_mask = np.abs(z_scores) > 2.5
            if np.any(outlier_mask):
                anomalies.append({
                    "type": "r_outlier",
                    "description": f"{int(np.sum(outlier_mask))} trades with extreme R-multiple values",
                    "count": int(np.sum(outlier_mask)),
                    "severity": "medium",
                })

        streak = 0
        for t in reversed(closed):
            if t.pnl < 0:
                streak += 1
            else:
                break
        if streak >= 5:
            anomalies.append({
                "type": "losing_streak",
                "description": f"Current losing streak of {streak} trades",
                "count": streak,
                "severity": "high" if streak >= 8 else "medium",
            })

        pnls = np.array([t.pnl for t in closed])
        cum_pnl = np.cumsum(np.concatenate(([0.0], pnls)))
        running_max = np.maximum.accumulate(cum_pnl)
        drawdowns = cum_pnl - running_max

        if len(drawdowns) > 5:
            recent_dd = float(np.min(drawdowns[-5:]))
            overall_dd = float(np.min(drawdowns))
            if overall_dd < 0 and recent_dd / overall_dd > 0.7:
                anomalies.append({
                    "type": "drawdown_concentration",
                    "description": "Recent trades concentrated in drawdown period",
                    "severity": "high",
                })

        spreads = np.array([t.spread_at_entry for t in closed if t.spread_at_entry > 0])
        if len(spreads) > 5:
            med_spread = float(np.median(spreads))
            std_spread = float(np.std(spreads, ddof=1))
            if std_spread > 0:
                for s in spreads[-5:]:
                    z = (s - med_spread) / std_spread
                    if z > 2.5:
                        anomalies.append({
                            "type": "spread_anomaly",
                            "description": f"Elevated spread: {s:.1f} pips (z={z:.1f})",
                            "severity": "low",
                        })
                        break

        # Layer additional detectors from the anomaly_detection module.
        from forex_agent.analysis import anomaly_detection as ad

        anomalies.extend(ad.detect_loss_clustering(trades))
        anomalies.extend(ad.detect_pair_deterioration(trades))
        anomalies.extend(ad.detect_session_deterioration(trades))

        dist_shifts = ad.detect_distribution_shift(trades)
        if dist_shifts:
            anomalies.extend(dist_shifts)

        mae_shifts = ad.detect_mae_mfe_shift(trades)
        if mae_shifts:
            anomalies.extend(mae_shifts)

        return anomalies

    def _analyze_regime(self, trades: list[TradeRecord]) -> dict[str, Any]:
        regime_trades: dict[str, list[TradeRecord]] = {}
        for t in trades:
            regime = t.regime.value if t.regime is not None else "unknown"
            regime_trades.setdefault(regime, []).append(t)

        results: dict[str, Any] = {}
        for regime_name, regime_trade_list in regime_trades.items():
            closed = [t for t in regime_trade_list if t.exit_price is not None]
            if not closed:
                results[regime_name] = {"count": len(regime_trade_list), "closed": 0}
                continue

            pnls = np.array([t.pnl for t in closed])
            r_values = _get_r_values(closed)

            results[regime_name] = {
                "count": len(regime_trade_list),
                "closed": len(closed),
                "win_rate": float(np.mean(pnls > 0)),
                "expectancy": float(np.mean(r_values)) if len(r_values) > 0 else 0.0,
                "total_pnl": float(np.sum(pnls)),
                "avg_pnl": float(np.mean(pnls)),
            }

        return results

    def _analyze_risk(self, trades: list[TradeRecord]) -> dict[str, Any]:
        closed = [t for t in trades if t.exit_price is not None]
        if not closed:
            return {"status": "no closed trades"}

        pnls = np.array([t.pnl for t in closed])
        equity = np.cumsum(np.concatenate(([0.0], pnls)))
        running_max = np.maximum.accumulate(equity)
        drawdowns = equity - running_max

        max_dd = float(np.min(drawdowns))
        peak = float(np.max(running_max))

        risk_pcts = np.array([
            t.risk_amount / t.account_balance
            for t in closed
            if t.account_balance > 0
        ])

        return {
            "max_drawdown": max_dd,
            "max_drawdown_pct": abs(max_dd / peak) if peak > 0 else 0.0,
            "avg_risk_per_trade": float(np.mean(risk_pcts)) if len(risk_pcts) > 0 else 0.0,
            "max_risk_single_trade": float(np.max(risk_pcts)) if len(risk_pcts) > 0 else 0.0,
            "trades_in_drawdown": int(np.sum(drawdowns < 0)),
        }

    def _analyze_execution(self, trades: list[TradeRecord]) -> dict[str, Any]:
        closed = [t for t in trades if t.exit_price is not None]
        if not closed:
            return {"status": "no closed trades"}

        spreads = np.array([t.spread_at_entry for t in closed if t.spread_at_entry > 0])
        slippages = np.array([t.slippage_pips for t in closed])
        commissions = np.array([t.commission for t in closed if t.commission > 0])

        result: dict[str, Any] = {}
        if len(spreads) > 0:
            result["avg_spread"] = float(np.mean(spreads))
            result["median_spread"] = float(np.median(spreads))
            result["max_spread"] = float(np.max(spreads))

        if len(slippages) > 0:
            result["avg_slippage"] = float(np.mean(slippages))
            result["max_slippage"] = float(np.max(slippages))
            result["total_slippage_cost"] = float(np.sum(np.abs(slippages)))

        if len(commissions) > 0:
            result["avg_commission"] = float(np.mean(commissions))
            result["total_commissions"] = float(np.sum(commissions))

        return result

    def _detect_patterns(
        self,
        trades: list[TradeRecord],
        failures: list[TradeFailureAnalysis],
    ) -> list[dict[str, Any]]:
        patterns: list[dict[str, Any]] = []

        category_counts: dict[str, int] = {}
        for f in failures:
            cat = f.failure_category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1

        total_failures = len(failures)
        for cat, count in category_counts.items():
            if count >= 3 and total_failures > 0:
                pct = count / total_failures
                patterns.append({
                    "pattern": f"Recurring {cat} failures",
                    "count": count,
                    "percentage": pct,
                    "description": f"{count} trades ({pct:.0%}) classified as {cat}",
                    "confidence": "high" if pct > 0.3 else "medium" if pct > 0.15 else "low",
                })

        symbol_losses: dict[str, int] = {}
        symbol_total: dict[str, int] = {}
        for t in trades:
            if t.exit_price is not None:
                pair = t.symbol
                symbol_total[pair] = symbol_total.get(pair, 0) + 1
                if t.pnl < 0:
                    symbol_losses[pair] = symbol_losses.get(pair, 0) + 1

        for pair, losses in symbol_losses.items():
            total = symbol_total.get(pair, 0)
            if total >= 5:
                loss_rate = losses / total
                if loss_rate > 0.6:
                    patterns.append({
                        "pattern": f"High loss rate for {pair}",
                        "count": losses,
                        "total": total,
                        "loss_rate": loss_rate,
                        "description": f"{pair} has {loss_rate:.0%} loss rate over {total} trades",
                        "confidence": "medium" if total >= 10 else "low",
                    })

        session_losses: dict[str, int] = {}
        session_total: dict[str, int] = {}
        for t in trades:
            if t.exit_price is not None:
                hour = t.entry_time.hour
                if 0 <= hour < 7:
                    session = "asian"
                elif 7 <= hour < 13:
                    session = "london"
                elif 13 <= hour < 21:
                    session = "new_york"
                else:
                    session = "off_hours"
                session_total[session] = session_total.get(session, 0) + 1
                if t.pnl < 0:
                    session_losses[session] = session_losses.get(session, 0) + 1

        for sess, losses in session_losses.items():
            total = session_total.get(sess, 0)
            if total >= 5:
                loss_rate = losses / total
                if loss_rate > 0.6:
                    patterns.append({
                        "pattern": f"High loss rate during {sess} session",
                        "count": losses,
                        "total": total,
                        "loss_rate": loss_rate,
                        "description": f"{sess} session shows {loss_rate:.0%} loss rate",
                        "confidence": "medium" if total >= 10 else "low",
                    })

        return patterns


class AnalysisReport:
    def __init__(
        self,
        total_trades: int,
        metrics: PerformanceMetrics,
        health: StrategyHealthScore,
        failures: list[TradeFailureAnalysis],
        anomalies: list[dict[str, Any]],
        regime_analysis: dict[str, Any],
        risk_analysis: dict[str, Any],
        execution_analysis: dict[str, Any],
        recurring_patterns: list[dict[str, Any]],
        alerts: list[Any] | None = None,
        winner_analyses: list[dict[str, Any]] | None = None,
        bootstrap_expectancy_ci: dict[str, float] | None = None,
    ) -> None:
        self.total_trades = total_trades
        self.metrics = metrics
        self.health = health
        self.failures = failures
        self.anomalies = anomalies
        self.regime_analysis = regime_analysis
        self.risk_analysis = risk_analysis
        self.execution_analysis = execution_analysis
        self.recurring_patterns = recurring_patterns
        self.alerts = alerts or []
        self.winner_analyses = winner_analyses or []
        self.bootstrap_expectancy_ci = bootstrap_expectancy_ci or {}
