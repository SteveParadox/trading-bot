from __future__ import annotations

import json
import os
from typing import Any

import requests

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
        client = openai.OpenAI(api_key=key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except ImportError:
        logger.debug("openai package not installed; falling back")
        return None
    except Exception as exc:
        logger.warning("OpenAI call failed: %s", exc)
        return None


def _call_ollama(
    system: str,
    user_message: str,
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.3,
    max_tokens: int = 2000,
) -> str | None:
    """Call Ollama local API. Returns response text or None on failure."""
    url = f"{base_url}/api/chat"
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
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content")
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
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        result = _call_openai(SYSTEM_PROMPT, user_message, model=openai_model, api_key=openai_key)
        if result:
            return {
                "explanation": result,
                "provider": "openai",
                "model": openai_model,
                "fallback": False,
            }

    # 2. Try Ollama
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1")
    result = _call_ollama(SYSTEM_PROMPT, user_message, model=ollama_model, base_url=ollama_url)
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
