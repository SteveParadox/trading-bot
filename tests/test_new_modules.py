from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from forex_agent.data.schemas import (
    DiagnosticDimension,
    DiagnosticLevel,
    TradeDiagnostic,
    TradeRecord,
    SimilarTradeResult,
    EvidencePackage,
    CriticAssessment,
    CounterfactualResult,
    ResearchHypothesis,
    HypothesisStatus,
)
from forex_agent.data.ingestion import compute_r_multiple


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_trade(**overrides) -> TradeRecord:
    defaults = dict(
        trade_id="T001",
        symbol="EURUSD",
        direction="LONG",
        entry_price=1.1000,
        exit_price=1.1050,
        stop_loss=1.0950,
        take_profit=1.1100,
        position_size=1.0,
        account_balance=10000.0,
        risk_amount=50.0,
        spread_at_entry=1.5,
        slippage_pips=0.0,
        entry_time=datetime(2025, 1, 15, 10, 0),
        exit_time=datetime(2025, 1, 15, 12, 0),
    )
    defaults.update(overrides)
    return TradeRecord(**defaults)


def _make_losing_trade(**overrides) -> TradeRecord:
    defaults = dict(
        trade_id="T002",
        symbol="EURUSD",
        direction="LONG",
        entry_price=1.1000,
        exit_price=1.0950,
        stop_loss=1.0950,
        take_profit=1.1100,
        position_size=1.0,
        account_balance=10000.0,
        risk_amount=50.0,
        realized_pl=-50.0,
        spread_at_entry=1.5,
        entry_time=datetime(2025, 1, 16, 10, 0),
        exit_time=datetime(2025, 1, 16, 11, 0),
    )
    defaults.update(overrides)
    return TradeRecord(**defaults)


def test_trade_analyst_drawdown_uses_account_equity() -> None:
    from forex_agent.agent.analyst import TradeAnalyst

    analyst = TradeAnalyst("", "")
    trades = [
        _make_losing_trade(trade_id="L1", account_balance=10_000.0),
        _make_losing_trade(trade_id="L2", account_balance=10_000.0),
    ]

    metrics = analyst._compute_metrics(trades)

    assert metrics.max_drawdown == pytest.approx(100.0)
    assert metrics.max_drawdown_pct == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Statistics enhanced tests
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_basic(self):
        from forex_agent.analysis.statistics_enhanced import bootstrap_confidence_interval
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = bootstrap_confidence_interval(data, n_bootstrap=1000, seed=42)
        assert "statistic" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["ci_lower"] <= result["statistic"] <= result["ci_upper"]
        assert abs(result["statistic"] - 3.0) < 0.01

    def test_empty(self):
        from forex_agent.analysis.statistics_enhanced import bootstrap_confidence_interval
        result = bootstrap_confidence_interval(np.array([]))
        assert result["statistic"] == 0.0

    def test_single_value(self):
        from forex_agent.analysis.statistics_enhanced import bootstrap_confidence_interval
        result = bootstrap_confidence_interval(np.array([5.0]))
        assert result["statistic"] == 5.0
        assert result["ci_lower"] == 5.0


class TestPermutationTest:
    def test_identical_groups(self):
        from forex_agent.analysis.statistics_enhanced import permutation_test
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = permutation_test(a, b, n_permutations=500, seed=42)
        assert result["p_value"] > 0.5
        assert abs(result["observed_diff"]) < 0.01

    def test_different_groups(self):
        from forex_agent.analysis.statistics_enhanced import permutation_test
        a = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = permutation_test(a, b, n_permutations=500, seed=42)
        assert result["p_value"] < 0.05
        assert result["observed_diff"] > 0


class TestMultipleTesting:
    def test_bh_correction(self):
        from forex_agent.analysis.statistics_enhanced import benjamini_hochberg
        p_vals = [0.01, 0.02, 0.03, 0.04, 0.50]
        result = benjamini_hochberg(p_vals, alpha=0.05)
        assert len(result) == 5
        assert all("adjusted_p" in r for r in result)
        assert all("significant" in r for r in result)

    def test_bonferroni(self):
        from forex_agent.analysis.statistics_enhanced import bonferroni_correction
        p_vals = [0.01, 0.02, 0.03]
        result = bonferroni_correction(p_vals, alpha=0.05)
        assert len(result) == 3
        # With Bonferroni: 0.01*3=0.03 (sig), 0.02*3=0.06 (not sig)
        assert result[0]["significant"]
        assert not result[1]["significant"]


# ---------------------------------------------------------------------------
# Schemas tests
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_trade_diagnostic_roundtrip(self):
        diag = TradeDiagnostic(
            trade_id="T001",
            outcome="loss",
            primary_diagnosis="elevated spread",
            primary_dimension=DiagnosticDimension.EXECUTION,
            confidence=0.65,
            evidence_level=DiagnosticLevel.OBSERVATION,
        )
        d = diag.to_dict()
        assert d["trade_id"] == "T001"
        assert d["outcome"] == "loss"
        assert d["confidence"] == 0.65

    def test_evidence_package_roundtrip(self):
        pkg = EvidencePackage(
            trade_id="T001",
            baseline={"win_rate": 0.55, "expectancy": 0.15},
            confidence=0.8,
        )
        d = pkg.to_dict()
        assert d["trade_id"] == "T001"
        assert d["baseline"]["win_rate"] == 0.55

    def test_hypothesis_status(self):
        h = ResearchHypothesis(
            hypothesis="Edge degrades in high vol",
            status=HypothesisStatus.PROPOSED,
        )
        assert h.status == HypothesisStatus.PROPOSED
        h.status = HypothesisStatus.SUPPORTED
        assert h.status.value == "supported"

    def test_critic_assessment(self):
        c = CriticAssessment(
            finding="High vol causes losses",
            initial_confidence=0.7,
            adjusted_confidence=0.42,
            status="inconclusive",
        )
        d = c.to_dict()
        assert d["adjusted_confidence"] == 0.42


# ---------------------------------------------------------------------------
# Similar-trade tests
# ---------------------------------------------------------------------------

class TestSimilarTrades:
    def test_finds_same_symbol_and_direction(self):
        from forex_agent.analysis.similarity import find_similar_trades
        target = _make_trade(trade_id="TARGET")
        others = [_make_trade(trade_id=f"S{i}") for i in range(10)]
        result = find_similar_trades(target, others)
        assert result.n_matches == 10
        assert result.win_rate == 1.0  # all winners in fixture

    def test_no_trades(self):
        from forex_agent.analysis.similarity import find_similar_trades
        target = _make_trade()
        result = find_similar_trades(target, [])
        assert result.n_matches == 0
        assert "no closed" in result.sample_size_warning.lower() or result.n_matches == 0

    def test_mixed_outcomes(self):
        from forex_agent.analysis.similarity import find_similar_trades
        target = _make_trade(trade_id="TARGET")
        winners = [_make_trade(trade_id=f"W{i}", exit_price=1.1050) for i in range(5)]
        losers = [_make_trade(trade_id=f"L{i}", exit_price=1.0950, symbol="GBPUSD") for i in range(5)]
        result = find_similar_trades(target, winners + losers)
        assert result.n_matches == 5  # only EURUSD winners


# ---------------------------------------------------------------------------
# Critic tests
# ---------------------------------------------------------------------------

class TestCritic:
    def test_small_sample_downgrades(self):
        from forex_agent.analysis.critic import assess_finding
        trades = [_make_trade(trade_id=f"T{i}") for i in range(5)]
        assessment = assess_finding("Some finding", trades, initial_confidence=0.8)
        assert assessment.adjusted_confidence < 0.8
        assert assessment.sample_size_concern

    def test_large_sample_stable(self):
        from forex_agent.analysis.critic import assess_finding
        trades = [_make_trade(trade_id=f"T{i}", entry_time=datetime(2025, 1, 1, i % 24, 0)) for i in range(100)]
        assessment = assess_finding("Some finding", trades, initial_confidence=0.8)
        # Should not be drastically downgraded
        assert assessment.adjusted_confidence > 0.6

    def test_multiple_tests_increases_concern(self):
        from forex_agent.analysis.critic import assess_finding
        trades = [_make_trade(trade_id=f"T{i}") for i in range(50)]
        assessment = assess_finding("Some finding", trades, initial_confidence=0.8, tests_conducted=10)
        assert assessment.overfitting_concern
        assert assessment.multiple_testing_concern


# ---------------------------------------------------------------------------
# Counterfactual tests
# ---------------------------------------------------------------------------

class TestCounterfactuals:
    def test_loss_gets_no_trade_counterfactual(self):
        from forex_agent.analysis.evidence import _build_counterfactuals
        trade = _make_losing_trade()
        cf = _build_counterfactuals(trade)
        assert any("Avoid trade" in c.scenario for c in cf)

    def test_counterfactual_fields(self):
        from forex_agent.analysis.evidence import _build_counterfactuals
        trade = _make_losing_trade()
        cf = _build_counterfactuals(trade)
        for c in cf:
            assert hasattr(c, "scenario")
            assert hasattr(c, "methodology")
            assert c.methodology in ("estimated", "hypothetical", "historical simulation")


# ---------------------------------------------------------------------------
# Evidence package tests
# ---------------------------------------------------------------------------

class TestEvidencePackage:
    def test_builds_with_trades(self):
        from forex_agent.analysis.evidence import build_evidence_package
        target = _make_trade(trade_id="TARGET")
        others = [_make_trade(trade_id=f"T{i}", entry_time=datetime(2025, 1, 1, i % 24, 0)) for i in range(20)]
        pkg = build_evidence_package(target, others)
        assert pkg.trade_id == "TARGET"
        assert "win_rate" in pkg.baseline
        assert pkg.similar_trades["n"] > 0

    def test_empty_pool(self):
        from forex_agent.analysis.evidence import build_evidence_package
        target = _make_trade(trade_id="TARGET")
        pkg = build_evidence_package(target, [])
        assert pkg.trade_id == "TARGET"
        assert pkg.similar_trades["n"] == 0


# ---------------------------------------------------------------------------
# LLM reasoner tests (template fallback only - no API keys in CI)
# ---------------------------------------------------------------------------

class TestLLMReasoner:
    def _force_template(self, monkeypatch):
        from forex_agent.analysis import llm_reasoner
        monkeypatch.setattr(llm_reasoner, "_call_openai", lambda *a, **k: None)
        monkeypatch.setattr(llm_reasoner, "_call_ollama", lambda *a, **k: None)

    def test_template_fallback_no_keys(self, monkeypatch):
        from forex_agent.analysis.llm_reasoner import generate_llm_explanation
        from forex_agent.data.schemas import (
            TradeDiagnostic, DiagnosticDimension, DiagnosticLevel,
            EvidencePackage, CriticAssessment,
        )
        self._force_template(monkeypatch)
        trade = _make_losing_trade(trade_id="LLM01")
        diagnostic = TradeDiagnostic(
            trade_id="LLM01",
            outcome="loss",
            primary_diagnosis="elevated spread",
            primary_dimension=DiagnosticDimension.EXECUTION,
            confidence=0.6,
            evidence_level=DiagnosticLevel.OBSERVATION,
        )
        evidence = EvidencePackage(trade_id="LLM01", baseline={"win_rate": 0.5})
        critic = CriticAssessment(
            finding="elevated spread",
            initial_confidence=0.6,
            adjusted_confidence=0.45,
            status="inconclusive",
        )

        # With no env keys set, should fall back to template
        result = generate_llm_explanation(trade, diagnostic, evidence, critic)
        assert result["provider"] == "template"
        assert result["fallback"] is True
        assert "Trade Summary" in result["explanation"]
        assert "LLM01" in result["explanation"]
        assert "LOSS" in result["explanation"]

    def test_ollama_adapter_rejects_malformed_success_payload(self, monkeypatch):
        from forex_agent.analysis import llm_reasoner

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"message": {"content": 123}}

        monkeypatch.setattr(llm_reasoner.requests, "post", lambda *args, **kwargs: Response())

        assert llm_reasoner._call_ollama("system", "user") is None

    def test_template_explains_all_sections(self):
        from forex_agent.analysis.llm_reasoner import _template_explanation
        from forex_agent.data.schemas import (
            TradeDiagnostic, DiagnosticDimension, DiagnosticLevel, DiagnosticFactor,
            EvidencePackage, CriticAssessment,
        )
        trade = _make_trade(trade_id="TMPL01")
        diagnostic = TradeDiagnostic(
            trade_id="TMPL01",
            outcome="win",
            primary_diagnosis="favorable regime",
            primary_dimension=DiagnosticDimension.MARKET_REGIME,
            confidence=0.75,
            evidence_level=DiagnosticLevel.ASSOCIATION,
            contributing_factors=[
                DiagnosticFactor(
                    dimension=DiagnosticDimension.MARKET_REGIME,
                    label="regime_alignment",
                    description="Aligned with trending regime",
                ),
            ],
        )
        evidence = EvidencePackage(
            trade_id="TMPL01",
            baseline={"win_rate": 0.55, "expectancy": 0.18},
            similar_trades={"n": 15, "win_rate": 0.6, "expectancy": 0.2},
        )
        result = _template_explanation(trade, diagnostic, evidence)
        assert "Trade Summary" in result
        assert "Primary Diagnosis" in result
        assert "Contributing Factors" in result
        assert "Evidence" in result
        assert "favorable regime" in result


# ---------------------------------------------------------------------------
# LLM research synthesis tests (template fallback only - no API keys in CI)
# ---------------------------------------------------------------------------

class TestResearchSummaryLLM:
    def _force_template(self, monkeypatch):
        from forex_agent.analysis import llm_reasoner
        monkeypatch.setattr(llm_reasoner, "_call_openai", lambda *a, **k: None)
        monkeypatch.setattr(llm_reasoner, "_call_ollama", lambda *a, **k: None)

    def test_template_fallback_no_keys(self, monkeypatch):
        from forex_agent.analysis.llm_reasoner import generate_research_summary
        self._force_template(monkeypatch)
        hypotheses = [
            {
                "hypothesis": "Strategy edge is weaker during asian session",
                "status": "testing",
                "evidence": "Pattern observed 12 times",
                "sample_size": 30,
                "p_value": 0.04,
                "effect_size": 0.35,
            },
            {
                "hypothesis": "Strategy lacks edge on EURUSD",
                "status": "rejected",
                "evidence": "t=1.2, p=0.31",
                "sample_size": 25,
            },
        ]
        findings = [{"text": "Spread ratio above 2x correlates with losses"}]
        decisions = [{"decision": "Stop trading during asian session"}]

        result = generate_research_summary(
            hypotheses=hypotheses, findings=findings, decisions=decisions
        )
        assert result["provider"] == "template"
        assert result["fallback"] is True
        assert "Research Overview" in result["summary"]
        assert "2 hypotheses" in result["summary"]
        assert "1 findings" in result["summary"]
        assert "testing: 1" in result["summary"]
        assert "rejected: 1" in result["summary"]

    def test_template_summarizes_empty_memory(self, monkeypatch):
        from forex_agent.analysis.llm_reasoner import generate_research_summary
        self._force_template(monkeypatch)
        result = generate_research_summary()
        assert result["provider"] == "template"
        assert result["fallback"] is True
        assert "0 hypotheses" in result["summary"]
        assert "No hypotheses recorded." in result["summary"]
