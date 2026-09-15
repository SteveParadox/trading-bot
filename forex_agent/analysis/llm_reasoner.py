from __future__ import annotations

import json
import os
from typing import Any

import requests

from forex_agent.config import load_config
from forex_agent.data.schemas import (
    CriticAssessment,
    EvidencePackage,
    TradeDiagnostic,
    TradeRecord,
)
from forex_agent.log import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a senior quantitative research assistant analyzing forex trades.

You must ONLY use facts from the Evidence Package provided. You MUST NOT:
- Invent statistics, trade data, or market conditions
- Claim causality from correlation
- Override quantitative calculations
- Fabricate confidence levels
- Hallucinate unavailable evidence

Every factual claim must be traceable to a field in the Evidence Package.
If evidence is insufficient, say "Insufficient evidence" or "I don't know."

You must distinguish:
- OBSERVATION: directly measurable fact
- ASSOCIATION: statistical relationship
- HYPOTHESIS: possible explanation
- CONCLUSION: sufficiently supported explanation

Produce a structured analysis in this exact format:

## Trade Summary
What happened?

## Expected Behavior
How does this compare with historically similar trades?

## Primary Diagnosis
What is the most likely explanation?

## Contributing Factors
What else may have mattered?

## Evidence
What quantitative evidence supports the diagnosis?

## Counterfactuals
What alternative conditions were investigated?

## Confidence
How strong is the evidence? (numeric 0-1 and qualitative)

## Alternative Explanations
What else could explain the outcome?

## Critic Assessment
Why might this conclusion be wrong?

## Research Recommendation
What experiment should we run next?
"""


RESEARCH_SYSTEM_PROMPT = """\
You are a senior quantitative researcher synthesizing an ongoing trading-strategy research program.

You MUST ONLY use the research memory provided below. You MUST NOT:
- Invent hypotheses, findings, or decisions
- Fabricate statistics or evidence
- Overstate confidence given the sample sizes
- Recommend strategy changes that contradict the evidence

Base every claim on the stored hypotheses, findings, and decisions.
Where evidence is thin or inconclusive, say so explicitly.

Produce a research synthesis in this exact format:

## Research Overview
What is the current state of the research program?

## Hypothesis Status
How many hypotheses are in each lifecycle state, and which are most important?

## Key Findings
Which findings are supported by the evidence so far?

## Open Questions
What remains unresolved or insufficiently tested?

## Recommended Next Steps
Which experiments or data should be prioritized next?

Be specific and reference hypotheses by their stored text where useful.
"""


def _build_user_message(
    trade: TradeRecord,
    diagnostic: TradeDiagnostic,
    evidence: EvidencePackage,
    critic: CriticAssessment | None = None,
) -> str:
    """Build the user message from the evidence package."""
    sections: list[str] = []

    sections.append("=== EVIDENCE PACKAGE ===")
    sections.append(f"Trade ID: {evidence.trade_id}")
    sections.append(f"Outcome: {diagnostic.outcome}")

    # Trade details
    if evidence.trade:
        t = evidence.trade
        sections.append(
            f"Pair: {t.get('symbol')} | Direction: {t.get('direction')} | "
            f"Entry: {t.get('entry_price')} | SL: {t.get('stop_loss')} | "
            f"TP: {t.get('take_profit', 'N/A')}"
        )
        if t.get("exit_price"):
            sections.append(f"Exit: {t.get('exit_price')} | P&L: {t.get('realized_pl', 'N/A')}")
        sections.append(f"Spread: {t.get('spread_at_entry', 0)} pips | Slippage: {t.get('slippage_pips', 0)} pips")

    # Baseline
    if evidence.baseline:
        b = evidence.baseline
        sections.append(
            f"\nBaseline: WR={b.get('win_rate', 0):.1%}, "
            f"Expectancy={b.get('expectancy', 0):.2f}R, "
            f"Median R={b.get('median_r', 0):.2f}, "
            f"Total trades={b.get('total_trades', 0):.0f}"
        )

    # Similar trades
    s = evidence.similar_trades
    if s.get("n", 0) > 0:
        sections.append(
            f"\nSimilar trades: {s['n']} matches "
            f"(WR={s.get('win_rate', 0):.1%}, "
            f"Expectancy={s.get('expectancy', 0):.2f}R)"
        )
        if s.get("definition"):
            sections.append(f"  Definition: {s['definition']}")
        if s.get("warning"):
            sections.append(f"  WARNING: {s['warning']}")

    # Regime
    if evidence.regime:
        r = evidence.regime
        sections.append(
            f"\nRegime: {r.get('regime', '?')} "
            f"({r.get('n_trades_in_regime', 0)} trades, "
            f"WR={r.get('win_rate', 0):.1%}, "
            f"Exp={r.get('expectancy_r', 0):.2f}R)"
        )

    # Execution
    if evidence.execution:
        e = evidence.execution
        sections.append(
            f"\nExecution: spread={e.get('spread_at_entry', 0):.1f} pips "
            f"(median={e.get('median_spread', 0):.1f}, "
            f"ratio={e.get('spread_ratio', 0):.1f}x)"
        )

    # Timing
    if evidence.timing:
        ti = evidence.timing
        sections.append(
            f"\nTiming: {ti.get('session', '?')} session, "
            f"{ti.get('day_of_week', '?')}, hour {ti.get('hour', '?')}"
        )

    # Anomalies
    if evidence.anomalies:
        sections.append("\nAnomalies:")
        for a in evidence.anomalies:
            sections.append(f"  - {a.get('description', '?')}")

    # Statistical tests
    if evidence.statistical_tests:
        sections.append("\nStatistical tests:")
        for st in evidence.statistical_tests:
            sections.append(
                f"  - {st.get('test', '?')}: "
                f"p={st.get('p_value', '?')}, "
                f"effect={st.get('effect_size', '?')}"
            )

    # Diagnostic
    sections.append(f"\n=== DIAGNOSTIC ===")
    sections.append(f"Primary: {diagnostic.primary_diagnosis}")
    sections.append(f"Dimension: {diagnostic.primary_dimension.value}")
    sections.append(f"Confidence: {diagnostic.confidence:.1%}")
    if diagnostic.contributing_factors:
        sections.append("Contributing:")
        for f in diagnostic.contributing_factors:
            sections.append(f"  [{f.dimension.value}] {f.label}: {f.description}")
    if diagnostic.protective_factors:
        sections.append("Protective:")
        for f in diagnostic.protective_factors:
            sections.append(f"  [{f.dimension.value}] {f.label}: {f.description}")
    if diagnostic.unknowns:
        sections.append(f"Unknowns: {', '.join(diagnostic.unknowns)}")

    # Critic
    if critic:
        sections.append(f"\n=== CRITIC ASSESSMENT ===")
        sections.append(f"Status: {critic.status}")
        sections.append(f"Confidence: {critic.initial_confidence:.1%} -> {critic.adjusted_confidence:.1%}")
        if critic.challenges:
            sections.append("Challenges:")
            for c in critic.challenges:
                sections.append(f"  - {c}")
        sections.append(f"Economic meaningfulness: {critic.economic_meaningfulness}")
        if critic.alternative_explanations:
            sections.append("Alternatives:")
            for a in critic.alternative_explanations:
                sections.append(f"  - {a}")

    sections.append(f"\n=== REQUESTED OUTPUT ===")
    sections.append("Produce the structured analysis following the required format.")
    sections.append("Use ONLY the data above. Do not invent statistics.")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _call_openai(
    system: str,
    user_message: str,
    model: str = "gpt-4o",
    api_key: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 2000,
) -> str | None:
    """Call OpenAI API. Returns response text or None on failure."""
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        return None

    try:
        import openai
        client = openai.OpenAI(api_key=key, timeout=30.0)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not response.choices:
            return None
        content = response.choices[0].message.content
        return content.strip() if isinstance(content, str) and content.strip() else None
    except ImportError:
        logger.debug("openai package not installed; falling back")
        return None
    except Exception as exc:
        logger.warning("OpenAI call failed: %s", exc)
        return None


def _call_ollama(
    system: str,
    user_message: str,
    model: str = "phi3:mini",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.3,
    max_tokens: int = 2000,
) -> str | None:
    """Call Ollama local API. Returns response text or None on failure."""
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        message = data.get("message") if isinstance(data, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        return content.strip() if isinstance(content, str) and content.strip() else None
    except Exception as exc:
        logger.debug("Ollama call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Template fallback (no LLM)
# ---------------------------------------------------------------------------

def _template_explanation(
    trade: TradeRecord,
    diagnostic: TradeDiagnostic,
    evidence: EvidencePackage,
    critic: CriticAssessment | None = None,
) -> str:
    """Template-based explanation when no LLM is available."""
    sections: list[str] = []

    sections.append(f"## Trade Summary")
    if trade.exit_price is not None:
        result = "WIN" if trade.pnl > 0 else "LOSS"
        sections.append(f"[{trade.trade_id}] {trade.symbol} {trade.direction} | {result} | P&L: ${trade.pnl:.2f}")
    else:
        sections.append(f"[{trade.trade_id}] {trade.symbol} {trade.direction} | OPEN")

    sections.append(f"\n## Expected Behavior")
    s = evidence.similar_trades
    if s.get("n", 0) > 0:
        sections.append(
            f"Based on {s['n']} similar trades: "
            f"win rate {s.get('win_rate', 0):.1%}, "
            f"expectancy {s.get('expectancy', 0):.2f}R"
        )
        if s.get("warning"):
            sections.append(f"Warning: {s['warning']}")
    else:
        sections.append("No similar trades found for comparison.")

    sections.append(f"\n## Primary Diagnosis")
    sections.append(f"{diagnostic.primary_diagnosis}")
    sections.append(f"Dimension: {diagnostic.primary_dimension.value}")

    if diagnostic.contributing_factors:
        sections.append(f"\n## Contributing Factors")
        for f in diagnostic.contributing_factors:
            sections.append(f"- [{f.dimension.value}] {f.description}")

    if diagnostic.protective_factors:
        sections.append(f"\n## Protective Factors")
        for f in diagnostic.protective_factors:
            sections.append(f"- [{f.dimension.value}] {f.description}")

    sections.append(f"\n## Evidence")
    sections.append(f"Confidence: {diagnostic.confidence:.1%}")
    sections.append(f"Evidence level: {diagnostic.evidence_level.value}")
    if evidence.statistical_tests:
        for st in evidence.statistical_tests:
            sections.append(f"- {st.get('test')}: p={st.get('p_value')}, effect={st.get('effect_size')}")

    sections.append(f"\n## Counterfactuals")
    if evidence.counterfactuals:
        for cf in evidence.counterfactuals:
            sections.append(f"- {cf.get('scenario', '?')}: {cf.get('notes', '')}")
    else:
        sections.append("No counterfactuals computed.")

    sections.append(f"\n## Confidence")
    sections.append(f"{diagnostic.confidence:.1%}")

    sections.append(f"\n## Alternative Explanations")
    if diagnostic.unknowns:
        for u in diagnostic.unknowns:
            sections.append(f"- {u}")
    else:
        sections.append("No unknowns recorded.")

    if critic:
        sections.append(f"\n## Critic Assessment")
        sections.append(f"Adjusted confidence: {critic.adjusted_confidence:.1%}")
        if critic.challenges:
            for c in critic.challenges:
                sections.append(f"- {c}")
        sections.append(f"Economic meaningfulness: {critic.economic_meaningfulness}")

    sections.append(f"\n## Research Recommendation")
    sections.append("Gather more data before drawing conclusions. Continue monitoring.")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Research memory synthesis
# ---------------------------------------------------------------------------

def _build_research_user_message(
    hypotheses: list[dict[str, Any]] | None = None,
    findings: list[dict[str, Any]] | None = None,
    decisions: list[dict[str, Any]] | None = None,
) -> str:
    """Build the user message from the research memory."""
    sections: list[str] = []

    sections.append("=== RESEARCH MEMORY ===")

    hyp_list = hypotheses or []
    if hyp_list:
        sections.append(f"\nHypotheses ({len(hyp_list)}):")
        for i, h in enumerate(hyp_list, 1):
            sections.append(
                f"  H{i}: {h.get('hypothesis', '?')} "
                f"[{h.get('status', 'unknown')}]"
            )
            if h.get("date_created"):
                sections.append(f"    Created: {h['date_created']}")
            if h.get("evidence"):
                sections.append(f"    Evidence: {h['evidence']}")
            if h.get("result"):
                sections.append(f"    Result: {h['result']}")
            if h.get("p_value") is not None:
                sections.append(
                    f"    p={h['p_value']:.3f}, effect={h.get('effect_size', 0):.2f}, "
                    f"sample={h.get('sample_size', 0)}"
                )
            if h.get("notes"):
                sections.append(f"    Notes: {h['notes']}")
    else:
        sections.append("\nHypotheses: none")

    findings_list = findings or []
    if findings_list:
        sections.append(f"\nFindings ({len(findings_list)}):")
        for i, f in enumerate(findings_list, 1):
            text = f.get("text") or f.get("finding") or f.get("summary") or "?"
            sections.append(f"  F{i}: {text}")
    else:
        sections.append("\nFindings: none")

    decisions_list = decisions or []
    if decisions_list:
        sections.append(f"\nDecisions ({len(decisions_list)}):")
        for i, d in enumerate(decisions_list, 1):
            text = d.get("text") or d.get("decision") or d.get("summary") or "?"
            sections.append(f"  D{i}: {text}")
    else:
        sections.append("\nDecisions: none")

    sections.append("\n=== REQUESTED OUTPUT ===")
    sections.append("Produce the research synthesis following the required format.")
    sections.append("Use ONLY the data above. Do not invent findings.")

    return "\n".join(sections)


def _template_research_summary(
    hypotheses: list[dict[str, Any]] | None = None,
    findings: list[dict[str, Any]] | None = None,
    decisions: list[dict[str, Any]] | None = None,
) -> str:
    """Template-based research synthesis when no LLM is available."""
    hyp_list = hypotheses or []
    findings_list = findings or []
    decisions_list = decisions or []

    sections: list[str] = []

    sections.append("## Research Overview")
    sections.append(
        f"Research memory contains {len(hyp_list)} hypotheses, "
        f"{len(findings_list)} findings, and {len(decisions_list)} decisions."
    )

    sections.append("\n## Hypothesis Status")
    if hyp_list:
        by_status: dict[str, int] = {}
        for h in hyp_list:
            status = h.get("status", "unknown")
            by_status[status] = by_status.get(status, 0) + 1
        for status, count in sorted(by_status.items()):
            sections.append(f"- {status}: {count}")
    else:
        sections.append("- No hypotheses recorded.")

    sections.append("\n## Key Findings")
    if findings_list:
        for i, f in enumerate(findings_list, 1):
            text = f.get("text") or f.get("finding") or f.get("summary") or "?"
            sections.append(f"- {text}")
    else:
        sections.append("- No findings recorded.")

    sections.append("\n## Open Questions")
    open_statuses = {"proposed", "testing", "inconclusive"}
    open_h = [h for h in hyp_list if h.get("status", "") in open_statuses]
    if open_h:
        sections.append(
            f"- {len(open_h)} hypotheses still open "
            "(proposed/testing/inconclusive)."
        )
    else:
        sections.append(
            "- No open hypotheses; all recorded hypotheses have a terminal status."
        )

    sections.append("\n## Recommended Next Steps")
    sections.append(
        "Run the proposed experiments to accumulate statistical evidence "
        "before changing the strategy."
    )

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_llm_explanation(
    trade: TradeRecord,
    diagnostic: TradeDiagnostic,
    evidence: EvidencePackage,
    critic: CriticAssessment | None = None,
) -> dict[str, Any]:
    """Generate trade explanation using LLM (OpenAI primary, Ollama fallback, template last).

    Returns dict with keys:
      - explanation: str (the natural language analysis)
      - provider: str (which provider was used)
      - model: str (which model was used)
      - fallback: bool (whether a fallback was used)
    """
    user_message = _build_user_message(trade, diagnostic, evidence, critic)

    # 1. Try OpenAI
    config = load_config()
    openai_key = config.openai_api_key
    if openai_key:
        openai_model = config.openai_model
        result = _call_openai(SYSTEM_PROMPT, user_message, model=openai_model, api_key=openai_key, temperature=config.llm_temperature, max_tokens=config.llm_max_tokens)
        if result:
            return {
                "explanation": result,
                "provider": "openai",
                "model": openai_model,
                "fallback": False,
            }

    # 2. Try Ollama
    ollama_url = config.ollama_base_url
    ollama_model = config.ollama_model
    result = _call_ollama(SYSTEM_PROMPT, user_message, model=ollama_model, base_url=ollama_url, temperature=config.llm_temperature, max_tokens=config.llm_max_tokens)
    if result:
        return {
            "explanation": result,
            "provider": "ollama",
            "model": ollama_model,
            "fallback": False,
        }

    # 3. Template fallback
    logger.info("No LLM available; using template explanation")
    template = _template_explanation(trade, diagnostic, evidence, critic)
    return {
        "explanation": template,
        "provider": "template",
        "model": "none",
        "fallback": True,
    }


def generate_research_summary(
    hypotheses: list[dict[str, Any]] | None = None,
    findings: list[dict[str, Any]] | None = None,
    decisions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate research synthesis using LLM (OpenAI primary, Ollama fallback, template last).

    Accepts the accumulated research memory (hypotheses, findings, decisions)
    and returns a natural-language synthesis of the research program.

    Returns dict with keys:
      - summary: str (the research synthesis)
      - provider: str (which provider was used)
      - model: str (which model was used)
      - fallback: bool (whether a fallback was used)
    """
    user_message = _build_research_user_message(hypotheses, findings, decisions)

    # 1. Try OpenAI
    config = load_config()
    openai_key = config.openai_api_key
    if openai_key:
        openai_model = config.openai_model
        result = _call_openai(RESEARCH_SYSTEM_PROMPT, user_message, model=openai_model, api_key=openai_key, temperature=config.llm_temperature, max_tokens=config.llm_max_tokens)
        if result:
            return {
                "summary": result,
                "provider": "openai",
                "model": openai_model,
                "fallback": False,
            }

    # 2. Try Ollama
    ollama_url = config.ollama_base_url
    ollama_model = config.ollama_model
    result = _call_ollama(RESEARCH_SYSTEM_PROMPT, user_message, model=ollama_model, base_url=ollama_url, temperature=config.llm_temperature, max_tokens=config.llm_max_tokens)
    if result:
        return {
            "summary": result,
            "provider": "ollama",
            "model": ollama_model,
            "fallback": False,
        }

    # 3. Template fallback
    logger.info("No LLM available; using template research summary")
    template = _template_research_summary(hypotheses, findings, decisions)
    return {
        "summary": template,
        "provider": "template",
        "model": "none",
        "fallback": True,
    }
