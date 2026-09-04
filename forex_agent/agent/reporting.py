from __future__ import annotations

from typing import Any

from forex_agent.agent.analyst import TradeAnalyst
from forex_agent.agent.explanations import (
    generate_failure_summary,
    generate_research_hypotheses,
    generate_trade_explanation,
)
from forex_agent.agent.hypotheses import propose_experiments
from forex_agent.data.schemas import TradeRecord, TradeFailureAnalysis


def generate_weekly_report(analyst: TradeAnalyst) -> str:
    report = analyst.run_full_analysis()
    sections: list[str] = []

    sections.append("=" * 70)
    sections.append("WEEKLY RESEARCH REPORT")
    sections.append("=" * 70)

    sections.append("")
    sections.append("1. EXECUTIVE SUMMARY")
    sections.append("-" * 40)
    m = report.metrics
    sections.append(
        f"  Total trades: {m.total_trades} | "
        f"Win rate: {m.win_rate:.1%} | "
        f"Expectancy: {m.expectancy:.2f}R"
    )
    sections.append(
        f"  Profit factor: {m.profit_factor:.2f} | "
        f"Total P&L: ${m.total_pnl:.2f} | "
        f"Sharpe: {m.sharpe_ratio:.2f}"
    )
    sections.append(f"  Strategy Health: {report.health.grade} (score: {report.health.score:.0f}/100)")
    if report.health.recommendations:
        sections.append("  Key concerns:")
        for rec in report.health.recommendations[:3]:
            sections.append(f"    - {rec}")

    if report.bootstrap_expectancy_ci:
        ci = report.bootstrap_expectancy_ci
        sections.append(
            f"  Expectancy CI (95%): {ci['statistic']:.2f}R "
            f"[{ci['ci_lower']:.2f}, {ci['ci_upper']:.2f}]"
        )

    sections.append("")
    sections.append("2. STRATEGY PERFORMANCE")
    sections.append("-" * 40)
    sections.append(f"  Win rate:        {m.win_rate:.1%} ({m.winning_trades}W / {m.losing_trades}L)")
    sections.append(f"  Expectancy:      {m.expectancy:.2f}R")
    sections.append(f"  Avg win:         ${m.average_win:.2f}")
    sections.append(f"  Avg loss:        ${m.average_loss:.2f}")
    sections.append(f"  Profit factor:   {m.profit_factor:.2f}")
    sections.append(f"  Avg R:R:         1:{m.avg_rr_ratio:.1f}")
    sections.append(f"  Largest win:     ${m.largest_win:.2f}")
    sections.append(f"  Largest loss:    ${m.largest_loss:.2f}")
    sections.append(f"  Max drawdown:    ${m.max_drawdown:.2f} ({m.max_drawdown_pct:.1%})")
    sections.append(f"  Consec. wins:    {m.consecutive_wins}")
    sections.append(f"  Consec. losses:  {m.consecutive_losses}")
    sections.append(f"  Avg duration:    {m.avg_duration_minutes:.0f} min")
    sections.append(f"  Sharpe ratio:    {m.sharpe_ratio:.2f}")

    sections.append("")
    sections.append("3. WINNER ANALYSIS")
    sections.append("-" * 40)
    if report.winner_analyses:
        qualities: dict[str, int] = {}
        for w in report.winner_analyses:
            q = w.get("quality", "unknown")
            qualities[q] = qualities.get(q, 0) + 1
        for q, count in sorted(qualities.items()):
            sections.append(f"  {q}: {count} trades")
        # Show top winners
        top_winners = sorted(report.winner_analyses, key=lambda x: x.get("r_multiple", 0), reverse=True)[:3]
        for w in top_winners:
            sections.append(
                f"  [{w['trade_id']}] {w['symbol']} {w['direction']}: "
                f"{w.get('r_multiple', 0):.2f}R ({w.get('quality', 'unknown')})"
            )
            if w.get("factors"):
                for f in w["factors"]:
                    sections.append(f"    + {f}")
    else:
        sections.append("  No winners to analyze.")

    sections.append("")
    sections.append("4. BIGGEST FAILURES")
    sections.append("-" * 40)
    significant = sorted(
        [f for f in report.failures if f.failure_category.value != "unknown"],
        key=lambda x: x.trade_id,
    )[:5]
    if significant:
        for fa in significant:
            sections.append(f"  [{fa.trade_id}] {fa.failure_category.value.upper()}: {fa.description}")
            if fa.contributing_factors:
                for factor in fa.contributing_factors[:2]:
                    sections.append(f"    - {factor}")
    else:
        sections.append("  No significant failures beyond normal variance.")

    sections.append("")
    sections.append("5. FAILURE CATEGORIES")
    sections.append("-" * 40)
    if report.failures:
        sections.append(generate_failure_summary(report.failures))
    else:
        sections.append("  No failures to categorize.")

    sections.append("")
    sections.append("6. RECURRING PATTERNS")
    sections.append("-" * 40)
    if report.recurring_patterns:
        for pat in report.recurring_patterns:
            sections.append(f"  [{pat.get('confidence', '?')}] {pat.get('pattern', 'unknown')}")
            sections.append(f"    Count: {pat.get('count', 0)} | {pat.get('description', '')}")
    else:
        sections.append("  No statistically supported recurring patterns detected.")

    sections.append("")
    sections.append("7. REGIME ANALYSIS")
    sections.append("-" * 40)
    if report.regime_analysis:
        for regime, data in report.regime_analysis.items():
            if isinstance(data, dict):
                count = data.get("count", 0)
                closed = data.get("closed", 0)
                wr = data.get("win_rate", 0.0)
                exp = data.get("expectancy", 0.0)
                pnl = data.get("total_pnl", 0.0)
                if closed > 0:
                    sections.append(f"  {regime}: {count} trades, {closed} closed, WR={wr:.0%}, Exp={exp:.2f}R, P&L=${pnl:.2f}")
                else:
                    sections.append(f"  {regime}: {count} trades, no closed data")
            else:
                sections.append(f"  {regime}: {data}")
    else:
        sections.append("  No regime data available.")

    sections.append("")
    sections.append("8. RISK ANALYSIS")
    sections.append("-" * 40)
    if report.risk_analysis:
        for key, value in report.risk_analysis.items():
            if key != "status":
                if isinstance(value, float):
                    sections.append(f"  {key}: {value:.4f}")
                else:
                    sections.append(f"  {key}: {value}")
    else:
        sections.append("  No risk data available.")

    sections.append("")
    sections.append("9. EXECUTION ANALYSIS")
    sections.append("-" * 40)
    if report.execution_analysis:
        for key, value in report.execution_analysis.items():
            if key != "status":
                if isinstance(value, float):
                    sections.append(f"  {key}: {value:.2f}")
                else:
                    sections.append(f"  {key}: {value}")
    else:
        sections.append("  No execution data available.")

    sections.append("")
    sections.append("10. HYPOTHESES")
    sections.append("-" * 40)
    hypotheses = generate_research_hypotheses(report.recurring_patterns)
    for i, hyp in enumerate(hypotheses, 1):
        sections.append(f"  H{i}: {hyp['hypothesis']}")
        sections.append(f"    Reason: {hyp['reason']}")
        sections.append(f"    Overfitting risk: {hyp['overfitting_risk']}")

    sections.append("")
    sections.append("11. RECOMMENDED EXPERIMENTS")
    sections.append("-" * 40)
    experiments = propose_experiments(report)
    if experiments:
        for i, exp in enumerate(experiments, 1):
            sections.append(f"  Experiment {i}: {exp.hypothesis}")
            sections.append(f"    Metric: {exp.metric}")
            sections.append(f"    Test: {exp.statistical_test}")
            sections.append(f"    Min sample: {exp.min_sample_size}")
            sections.append(f"    Acceptance: {exp.acceptance_criteria}")
            sections.append(f"    Overfitting risk: {exp.overfitting_risk}")
            sections.append(f"    OOS plan: {exp.out_of_sample_plan}")
    else:
        sections.append("  No experiments recommended at this time.")

    sections.append("")
    sections.append("=" * 70)

    return "\n".join(sections)


def generate_trade_report(trade: TradeRecord, analysis: TradeFailureAnalysis) -> str:
    return generate_trade_explanation(trade, analysis)


def generate_monthly_report(analyst: TradeAnalyst) -> str:
    report = analyst.run_full_analysis()
    sections: list[str] = []

    sections.append("=" * 70)
    sections.append("MONTHLY RESEARCH REPORT")
    sections.append("=" * 70)

    m = report.metrics
    sections.append("")
    sections.append("SUMMARY")
    sections.append("-" * 40)
    sections.append(
        f"  {m.total_trades} trades | "
        f"WR: {m.win_rate:.1%} | "
        f"Exp: {m.expectancy:.2f}R | "
        f"PF: {m.profit_factor:.2f}"
    )
    sections.append(
        f"  Total P&L: ${m.total_pnl:.2f} | "
        f"Max DD: {m.max_drawdown_pct:.1%} | "
        f"Sharpe: {m.sharpe_ratio:.2f}"
    )
    sections.append(f"  Health: {report.health.grade} ({report.health.score:.0f}/100)")

    if report.recurring_patterns:
        sections.append("")
        sections.append("KEY PATTERNS")
        sections.append("-" * 40)
        for pat in report.recurring_patterns:
            sections.append(f"  {pat.get('pattern', 'unknown')}: {pat.get('description', '')}")

    if report.health.recommendations:
        sections.append("")
        sections.append("RECOMMENDATIONS")
        sections.append("-" * 40)
        for rec in report.health.recommendations:
            sections.append(f"  - {rec}")

    experiments = propose_experiments(report)
    if experiments:
        sections.append("")
        sections.append("EXPERIMENTS TO CONSIDER")
        sections.append("-" * 40)
        for exp in experiments[:3]:
            sections.append(f"  {exp.hypothesis}")
            sections.append(f"    Test: {exp.statistical_test} | Min n: {exp.min_sample_size}")

    sections.append("")
    sections.append("=" * 70)
    return "\n".join(sections)


def generate_health_dashboard(analyst: TradeAnalyst) -> dict[str, Any]:
    health = analyst.get_strategy_health()
    anomalies = analyst.get_anomalies()
    regime = analyst.get_regime_analysis()
    metrics = analyst._compute_metrics(analyst.trades)

    return {
        "health_grade": health.grade,
        "health_score": health.score,
        "health_components": health.components,
        "recommendations": health.recommendations,
        "metrics": {
            "total_trades": metrics.total_trades,
            "win_rate": metrics.win_rate,
            "expectancy": metrics.expectancy,
            "profit_factor": metrics.profit_factor,
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "sharpe_ratio": metrics.sharpe_ratio,
            "total_pnl": metrics.total_pnl,
            "consecutive_losses": metrics.consecutive_losses,
        },
        "anomalies": anomalies,
        "regime_analysis": regime,
    }
