from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Union

try:
    from types import UnionType
except ImportError:  # pragma: no cover - Python < 3.10
    UnionType = None


class FailureCategory(str, Enum):
    """Categorization of why a trade failed, per the system spec."""

    VALID_STRATEGY_LOSS = "valid_strategy_loss"
    SIGNAL_FAILURE = "signal_failure"
    REGIME_MISMATCH = "regime_mismatch"
    POOR_ENTRY = "poor_entry"
    EXECUTION_ERROR = "execution_error"
    RISK_MISMANAGEMENT = "risk_mismanagement"
    RULE_VIOLATION = "rule_violation"
    DATA_QUALITY_FAILURE = "data_quality_failure"
    EMOTIONAL_TRADE = "emotional_trade"
    MARKET_CONDITION = "market_condition"
    INVALID_SETUP = "invalid_setup"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class MarketRegime(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    NORMAL_VARIANCE = "normal_variance"
    WATCH = "watch"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    INSUFFICIENT_DATA = "insufficient_data"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class DataclassMixin:
    """Serialization support shared by all record dataclasses."""

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataclassMixin":
        """Reconstruct an instance from a JSON-decoded dict.

        Coerces primitive values back into enum/datetime types where the
        dataclass declares them. Nested dataclass fields are rebuilt
        recursively.
        """
        from typing import get_type_hints

        hints = get_type_hints(cls)
        kwargs: dict[str, Any] = {}
        for fname, ftype in hints.items():
            if fname not in data:
                continue
            value = data[fname]
            kwargs[fname] = _restore(value, ftype)
        return cls(**kwargs)

    @classmethod
    def from_json(cls, raw: str) -> "DataclassMixin":
        import json

        return cls.from_dict(json.loads(raw))


def _restore(value: Any, ftype: Any) -> Any:
    """Coerce a JSON-decoded value back to the dataclass field type."""
    if value is None:
        return None

    origin = getattr(ftype, "__origin__", None)
    args = getattr(ftype, "__args__", ())

    if origin in (list,):
        if args:
            return [_restore(v, args[0]) for v in value]
        return list(value)
    if origin in (dict,):
        if len(args) == 2:
            return {k: _restore(v, args[1]) for k, v in value.items()}
        return dict(value)
    if origin is not None and hasattr(origin, "__origin__"):
        return value

    # Union / Optional
    if origin is UnionType or origin is Union:
        for candidate in args:
            if candidate is type(None):
                continue
            try:
                return _restore(value, candidate)
            except Exception:
                continue
        return value

    if isinstance(ftype, type):
        if issubclass(ftype, Enum):
            try:
                return ftype(value)
            except (ValueError, TypeError):
                return value
        if ftype is datetime:
            if isinstance(value, datetime):
                return value
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if is_dataclass(ftype) and isinstance(value, dict):
            return ftype.from_dict(value)
    return value


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class TradeRecord(DataclassMixin):
    """Single forward-test trade record.

    Only fields that exist in the source data should be populated; all others
    remain None/empty (explicit missing, never invented).
    """

    trade_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    symbol: str = "EURUSD"
    direction: str = "LONG"
    entry_price: float = 1.1000
    exit_price: Optional[float] = None
    stop_loss: float = 1.0950
    take_profit: Optional[float] = 1.1100
    position_size: float = 1.0
    account_balance: float = 10000.0
    risk_amount: float = 50.0
    realized_pl: Optional[float] = None
    entry_time: datetime = field(default_factory=_now_utc)
    exit_time: Optional[datetime] = None
    spread_at_entry: float = 1.5
    slippage_pips: float = 0.0
    commission: float = 0.0
    financing: float = 0.0
    notes: str = ""
    regime: Optional[MarketRegime] = None
    follow_stop_loss: bool = True
    follow_take_profit: bool = True

    # --- Expanded fields from the system spec (all optional, default None) --
    strategy_name: Optional[str] = None
    timeframe: Optional[str] = None
    entry_reason: Optional[str] = None
    signal_strength: Optional[float] = None
    mfe: Optional[float] = None
    mae: Optional[float] = None
    exit_reason: Optional[str] = None
    day_of_week: Optional[str] = None
    session: Optional[str] = None
    volatility: Optional[float] = None
    atr: Optional[float] = None
    trend_characteristics: Optional[str] = None
    news_context: Optional[str] = None
    indicator_values_entry: dict[str, Any] = field(default_factory=dict)
    indicator_values_exit: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def total_realized_pnl(self) -> float:
        """Realized P&L including financing."""
        if self.exit_price is None:
            return 0.0
        if self.direction == "LONG":
            base = (self.exit_price - self.entry_price) * self.position_size * 100000
        else:
            base = (self.entry_price - self.exit_price) * self.position_size * 100000
        return base - self.commission + self.financing

    @property
    def is_winner(self) -> bool:
        if self.exit_price is None:
            return False
        if self.direction == "LONG":
            return self.exit_price > self.entry_price
        return self.exit_price < self.entry_price

    @property
    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        # Prefer the realized P&L provided by the broker when available.
        if self.realized_pl is not None:
            return self.realized_pl
        if self.direction == "LONG":
            return (self.exit_price - self.entry_price) * self.position_size * 100000
        return (self.entry_price - self.exit_price) * self.position_size * 100000

    @property
    def risk_per_pip(self) -> float:
        return self.position_size * 100000 * 0.0001


@dataclass
class TradeFailureAnalysis(DataclassMixin):
    trade_id: str
    failure_category: FailureCategory
    confidence: ConfidenceLevel
    description: str
    contributing_factors: list[str] = field(default_factory=list)
    counterfactual: str = ""
    historical_comparison: str = ""
    recommended_action: str = ""
    verdict: str = ""
    research_implication: str = ""
    evidence: list[str] = field(default_factory=list)


@dataclass
class StrategyHealthScore(DataclassMixin):
    score: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    grade: str = "F"
    recommendations: list[str] = field(default_factory=list)
    health_status: HealthStatus = HealthStatus.INSUFFICIENT_DATA
    explanation: str = ""

    def compute_grade(self) -> str:
        if self.score >= 90:
            return "A+"
        elif self.score >= 80:
            return "A"
        elif self.score >= 70:
            return "B"
        elif self.score >= 60:
            return "C"
        elif self.score >= 50:
            return "D"
        return "F"


@dataclass
class PerformanceMetrics(DataclassMixin):
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    loss_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    avg_r_multiple: float = 0.0
    median_r_multiple: float = 0.0
    avg_rr_ratio: float = 0.0
    payoff_ratio: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    avg_duration_minutes: float = 0.0
    median_duration_minutes: float = 0.0
    recovery_factor: float = 0.0
    trade_frequency_per_day: float = 0.0
    avg_mfe_r: float = 0.0
    avg_mae_r: float = 0.0


@dataclass
class Alert(DataclassMixin):
    severity: AlertSeverity = AlertSeverity.WARNING
    alert_type: str = ""
    description: str = ""
    threshold: float = 0.0
    actual: float = 0.0
    timestamp: datetime = field(default_factory=_now_utc)
    trade_id: Optional[str] = None


@dataclass
class DataQualityMetrics(DataclassMixin):
    total_trades: int = 0
    with_exit: int = 0
    with_mfe: int = 0
    with_mae: int = 0
    with_regime: int = 0
    with_session: int = 0
    with_spread: int = 0
    with_atr: int = 0
    missing_pct: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Diagnostic dimensions (Section 4 of spec)
# ---------------------------------------------------------------------------

class DiagnosticDimension(str, Enum):
    """Primary diagnostic dimensions for trade analysis."""
    SIGNAL = "signal"
    MARKET_REGIME = "market_regime"
    EXECUTION = "execution"
    RISK = "risk"
    TIMING = "timing"
    TRADE_MANAGEMENT = "trade_management"
    UNKNOWN = "unknown"


class DiagnosticLevel(str, Enum):
    """Levels of evidence quality."""
    OBSERVATION = "observation"
    ASSOCIATION = "association"
    HYPOTHESIS = "hypothesis"
    CONCLUSION = "conclusion"


# ---------------------------------------------------------------------------
# Multi-factor trade diagnostic (Section 4)
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticFactor(DataclassMixin):
    """A single contributing or protective factor for a trade."""
    dimension: DiagnosticDimension
    label: str
    level: DiagnosticLevel = DiagnosticLevel.OBSERVATION
    description: str = ""
    supporting_evidence: list[str] = field(default_factory=list)
    sample_size: int = 0
    effect_size: float = 0.0
    confidence: float = 0.0


@dataclass
class TradeDiagnostic(DataclassMixin):
    """Multi-factor diagnostic for a single trade (replaces single-category model)."""
    trade_id: str
    outcome: str = ""
    primary_diagnosis: str = ""
    primary_dimension: DiagnosticDimension = DiagnosticDimension.UNKNOWN
    contributing_factors: list[DiagnosticFactor] = field(default_factory=list)
    protective_factors: list[DiagnosticFactor] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence_level: DiagnosticLevel = DiagnosticLevel.OBSERVATION
    sample_sizes: dict[str, int] = field(default_factory=dict)
    statistical_support: dict[str, Any] = field(default_factory=dict)
    counterfactuals: list[dict[str, Any]] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Evidence Package (Section 6)
# ---------------------------------------------------------------------------

@dataclass
class EvidencePackage(DataclassMixin):
    """Structured evidence package for a single trade."""
    trade_id: str
    trade: Optional[dict[str, Any]] = None
    baseline: dict[str, float] = field(default_factory=dict)
    similar_trades: dict[str, Any] = field(default_factory=dict)
    regime: dict[str, Any] = field(default_factory=dict)
    execution: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] = field(default_factory=dict)
    anomalies: list[dict[str, Any]] = field(default_factory=list)
    historical_comparisons: list[dict[str, Any]] = field(default_factory=list)
    counterfactuals: list[dict[str, Any]] = field(default_factory=list)
    statistical_tests: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Similar-trade result (Section 7)
# ---------------------------------------------------------------------------

@dataclass
class SimilarTradeResult(DataclassMixin):
    """Result of similar-trade lookup for a given trade."""
    trade_id: str
    definition_of_similar: str = ""
    n_matches: int = 0
    win_rate: float = 0.0
    expectancy_r: float = 0.0
    median_mae: float = 0.0
    median_mfe: float = 0.0
    outcome_distribution: dict[str, int] = field(default_factory=dict)
    matching_trade_ids: list[str] = field(default_factory=list)
    sample_size_warning: str = ""


# ---------------------------------------------------------------------------
# Counterfactual result (Section 8)
# ---------------------------------------------------------------------------

@dataclass
class CounterfactualResult(DataclassMixin):
    """A single counterfactual scenario for a trade."""
    scenario: str = ""
    estimated_outcome_r: Optional[float] = None
    methodology: str = "estimated"
    confidence: float = 0.0
    data_available: bool = False
    notes: str = ""


# ---------------------------------------------------------------------------
# Critic assessment (Section 13)
# ---------------------------------------------------------------------------

@dataclass
class CriticAssessment(DataclassMixin):
    """Challenges a finding or conclusion with bias-awareness checks."""
    finding: str = ""
    challenges: list[str] = field(default_factory=list)
    sample_size_concern: bool = False
    independence_concern: bool = False
    survivorship_bias_concern: bool = False
    look_ahead_bias_concern: bool = False
    overfitting_concern: bool = False
    multiple_testing_concern: bool = False
    out_of_sample_concern: bool = False
    economic_meaningfulness: str = ""
    alternative_explanations: list[str] = field(default_factory=list)
    initial_confidence: float = 0.0
    adjusted_confidence: float = 0.0
    status: str = ""


# ---------------------------------------------------------------------------
# Hypothesis with lifecycle (Section 11)
# ---------------------------------------------------------------------------

class HypothesisStatus(str, Enum):
    PROPOSED = "proposed"
    TESTING = "testing"
    SUPPORTED = "supported"
    WEAKENED = "weakened"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


@dataclass
class ResearchHypothesis(DataclassMixin):
    """A tracked hypothesis with lifecycle status."""
    hypothesis: str
    date_created: str = ""
    evidence: str = ""
    sample_size: int = 0
    test_methodology: str = ""
    expected_effect: str = ""
    result: str = ""
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    confidence_interval: Optional[dict[str, float]] = None
    notes: str = ""
