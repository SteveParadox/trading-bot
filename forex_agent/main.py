from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="forex_agent",
        description="Forex Forward-Test Monitoring & Trade-Failure Analysis Agent",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("analyze", help="Run full analysis and print report")

    trade_parser = sub.add_parser("trade", help="Analyze a specific trade")
    trade_parser.add_argument("trade_id", help="Trade ID to analyze")

    sub.add_parser("health", help="Show strategy health score")

    sub.add_parser("report", help="Generate weekly research report")

    sub.add_parser("validate", help="Run data validation")

    sub.add_parser("anomalies", help="Run anomaly detection")

    sub.add_parser("experiments", help="Propose research experiments")

    sub.add_parser("monthly", help="Generate monthly report")

    sub.add_parser("dashboard", help="Output health dashboard as JSON")

    sub.add_parser("alerts", help="Show generated alerts")

    # --- New commands (spec upgrade) ---
    diag_parser = sub.add_parser("diagnose", help="Multi-factor diagnostic for a trade")
    diag_parser.add_argument("trade_id", help="Trade ID to diagnose")

    ev_parser = sub.add_parser("evidence", help="Build evidence package for a trade")
    ev_parser.add_argument("trade_id", help="Trade ID")

    sim_parser = sub.add_parser("similar", help="Find similar trades")
    sim_parser.add_argument("trade_id", help="Trade ID")

    crit_parser = sub.add_parser("critique", help="Run critic/anti-bias assessment on a finding")
    crit_parser.add_argument("finding", help="Finding text to critique")

    sub.add_parser("diagnose-all", help="Diagnose all trades (winners and losers)")

    sub.add_parser("research", help="Show research memory summary")

    explain_parser = sub.add_parser("explain", help="LLM-powered trade explanation")
    explain_parser.add_argument("trade_id", help="Trade ID to explain")

    args = parser.parse_args(argv)

    if args.command is None:
        _run_default()
        return 0

    cmd = args.command

    if cmd == "analyze":
        _cmd_analyze()
    elif cmd == "trade":
        _cmd_trade(args.trade_id)
    elif cmd == "health":
        _cmd_health()
    elif cmd == "report":
        _cmd_report()
    elif cmd == "validate":
        _cmd_validate()
    elif cmd == "anomalies":
        _cmd_anomalies()
    elif cmd == "experiments":
        _cmd_experiments()
    elif cmd == "monthly":
        _cmd_monthly()
    elif cmd == "dashboard":
        _cmd_dashboard()
    elif cmd == "alerts":
        _cmd_alerts()
    elif cmd == "diagnose":
        _cmd_diagnose(args.trade_id)
    elif cmd == "evidence":
        _cmd_evidence(args.trade_id)
    elif cmd == "similar":
        _cmd_similar(args.trade_id)
    elif cmd == "critique":
        _cmd_critique(args.finding)
    elif cmd == "diagnose-all":
        _cmd_diagnose_all()
    elif cmd == "research":
        _cmd_research()
    elif cmd == "explain":
        _cmd_explain(args.trade_id)
    else:
        parser.print_help()
        return 1

    return 0


def _build_analyst() -> "TradeAnalyst":
    from fxbot.config import settings_from_env
    from forex_agent.agent.analyst import TradeAnalyst

    settings = settings_from_env()
    return TradeAnalyst(
        database_url=settings.runtime.database_url,
        jsonl_path=settings.runtime.log_jsonl_path,
    )


def _cmd_analyze() -> None:
    analyst = _build_analyst()
    report = analyst.run_full_analysis()

    from forex_agent.agent.explanations import generate_failure_summary

    m = report.metrics
    print(f"Total trades: {m.total_trades}")
    print(f"Win rate: {m.win_rate:.1%}")
    print(f"Expectancy: {m.expectancy:.2f}R")
    print(f"Profit factor: {m.profit_factor:.2f}")
    print(f"Total P&L: ${m.total_pnl:.2f}")
    print(f"Max drawdown: {m.max_drawdown_pct:.1%}")
    if m.sharpe_ratio is not None:
        print(f"Sharpe: {m.sharpe_ratio:.2f}")
    print(f"Health: {report.health.health_status.value.upper()} ({report.health.grade}, score: {report.health.score:.0f}/100)")
    print(f"  {report.health.explanation}")

    if report.bootstrap_expectancy_ci:
        ci = report.bootstrap_expectancy_ci
        print(f"\nExpectancy (bootstrap 95% CI): {ci['statistic']:.2f}R [{ci['ci_lower']:.2f}, {ci['ci_upper']:.2f}]")

    if report.alerts:
        print(f"\nAlerts: {len(report.alerts)}")
        for a in report.alerts:
            print(
                f"  [{a.severity.value.upper()}] {a.alert_type}: {a.description}"
            )

    if report.failures:
        print(f"\nFailure summary ({len(report.failures)} losses):")
        print(generate_failure_summary(report.failures))

    if report.winner_analyses:
        print(f"\nWinner analysis ({len(report.winner_analyses)} winners):")
        qualities: dict[str, int] = {}
        for w in report.winner_analyses:
            q = w.get("quality", "unknown")
            qualities[q] = qualities.get(q, 0) + 1
        for q, count in sorted(qualities.items()):
            print(f"  {q}: {count}")

    if report.anomalies:
        print(f"\nAnomalies detected: {len(report.anomalies)}")
        for a in report.anomalies:
            print(f"  [{a.get('severity', '?')}] {a.get('description', 'unknown')}")

    if report.recurring_patterns:
        print(f"\nRecurring patterns: {len(report.recurring_patterns)}")
        for p in report.recurring_patterns:
            print(f"  {p.get('pattern', 'unknown')}")


def _cmd_trade(trade_id: str) -> None:
    analyst = _build_analyst()
    result = analyst.analyze_single_trade(trade_id)
    if result is None:
        print(f"Trade {trade_id} not found")
        return

    trades = analyst.trades
    trade = next((t for t in trades if t.trade_id == trade_id), None)
    if trade is None:
        print(f"Trade {trade_id} not found in records")
        return

    from forex_agent.agent.explanations import generate_trade_explanation
    print(generate_trade_explanation(trade, result))


def _cmd_health() -> None:
    analyst = _build_analyst()
    health = analyst.get_strategy_health()
    print(f"Health Score: {health.score:.0f}/100")
    print(f"Grade: {health.grade}")
    print(f"Status: {health.health_status.value.upper()}")
    print(f"  {health.explanation}")
    print("\nComponents:")
    for name, value in health.components.items():
        print(f"  {name}: {value:.0f}")
    if health.recommendations:
        print("\nRecommendations:")
        for rec in health.recommendations:
            print(f"  - {rec}")


def _cmd_report() -> None:
    from forex_agent.agent.reporting import generate_weekly_report
    analyst = _build_analyst()
    print(generate_weekly_report(analyst))


def _cmd_monthly() -> None:
    from forex_agent.agent.reporting import generate_monthly_report
    analyst = _build_analyst()
    print(generate_monthly_report(analyst))


def _cmd_validate() -> None:
    from forex_agent.data.ingestion import load_trades_from_jsonl
    from forex_agent.data.validation import validate_trades
    from pathlib import Path

    analyst = _build_analyst()
    path = Path(analyst.jsonl_path)
    if not path.exists():
        print(f"JSONL file not found: {analyst.jsonl_path}")
        return

    trades = load_trades_from_jsonl(str(path))
    report = validate_trades(trades)
    print(f"Total trades: {report.total_trades}")
    print(f"Clean trades: {report.clean_trades}")
    print(f"Errors: {report.error_count}")
    print(f"Warnings: {report.warning_count}")
    if report.issues:
        print("\nIssues:")
        for issue in report.issues:
            print(f"  [{issue.severity}] {issue.trade_id}: {issue.field} - {issue.message}")


def _cmd_anomalies() -> None:
    analyst = _build_analyst()
    anomalies = analyst.get_anomalies()
    if not anomalies:
        print("No anomalies detected")
        return
    print(f"Detected {len(anomalies)} anomalies:")
    for a in anomalies:
        print(f"  [{a.get('severity', '?')}] {a.get('type', 'unknown')}: {a.get('description', '')}")


def _cmd_experiments() -> None:
    from forex_agent.agent.hypotheses import propose_experiments
    analyst = _build_analyst()
    report = analyst.run_full_analysis()
    experiments = propose_experiments(report)
    if not experiments:
        print("No experiments recommended")
        return
    print(f"{len(experiments)} experiment(s) proposed:")
    for i, exp in enumerate(experiments, 1):
        print(f"\n  Experiment {i}: {exp.hypothesis}")
        print(f"    Reason: {exp.reason}")
        print(f"    Metric: {exp.metric}")
        print(f"    Test: {exp.statistical_test}")
        print(f"    Min sample: {exp.min_sample_size}")
        print(f"    Acceptance: {exp.acceptance_criteria}")
        print(f"    Rejection: {exp.rejection_criteria}")
        print(f"    Overfitting risk: {exp.overfitting_risk}")
        print(f"    OOS plan: {exp.out_of_sample_plan}")


def _cmd_dashboard() -> None:
    import json
    from forex_agent.agent.reporting import generate_health_dashboard
    analyst = _build_analyst()
    dashboard = generate_health_dashboard(analyst)
    print(json.dumps(dashboard, indent=2, default=str))


def _cmd_alerts() -> None:
    analyst = _build_analyst()
    alerts = analyst.get_alerts()
    if not alerts:
        print("No active alerts.")
        return
    for alert in alerts:
        print(
            f"[{alert.severity.value.upper()}] {alert.alert_type}: "
            f"{alert.description} "
            f"(threshold: {alert.threshold}, actual: {alert.actual:.2f})"
        )


def _cmd_diagnose(trade_id: str) -> None:
    analyst = _build_analyst()
    diag = analyst.diagnose_trade(trade_id)
    if diag is None:
        print(f"Trade {trade_id} not found")
        return

    print(f"MULTI-FACTOR DIAGNOSIS: {diag.trade_id}")
    print(f"  Outcome: {diag.outcome}")
    print(f"  Primary diagnosis: {diag.primary_diagnosis}")
    print(f"  Primary dimension: {diag.primary_dimension.value}")
    print(f"  Confidence: {diag.confidence:.1%}")
    print(f"  Evidence level: {diag.evidence_level.value}")

    if diag.contributing_factors:
        print("\n  CONTRIBUTING FACTORS:")
        for f in diag.contributing_factors:
            print(f"    [{f.dimension.value}] {f.label}: {f.description}")
            if f.sample_size:
                print(f"      (sample: {f.sample_size}, effect: {f.effect_size:.2f})")

    if diag.protective_factors:
        print("\n  PROTECTIVE FACTORS:")
        for f in diag.protective_factors:
            print(f"    [{f.dimension.value}] {f.label}: {f.description}")

    if diag.observations:
        print("\n  OBSERVATIONS:")
        for obs in diag.observations:
            print(f"    - {obs}")

    if diag.unknowns:
        print("\n  UNKNOWN:")
        for u in diag.unknowns:
            print(f"    - {u}")


def _cmd_evidence(trade_id: str) -> None:
    import json
    analyst = _build_analyst()
    pkg = analyst.build_evidence_package(trade_id)
    if pkg is None:
        print(f"Trade {trade_id} not found")
        return
    print(json.dumps(pkg.to_dict(), indent=2, default=str))


def _cmd_similar(trade_id: str) -> None:
    analyst = _build_analyst()
    result = analyst.find_similar(trade_id)
    if result is None:
        print(f"Trade {trade_id} not found")
        return

    print(f"SIMILAR TRADES: {result.trade_id}")
    print(f"  Definition: {result.definition_of_similar}")
    print(f"  Matches: {result.n_matches}")
    print(f"  Win rate: {result.win_rate:.1%}")
    print(f"  Expectancy: {result.expectancy_r:.2f}R")
    if result.median_mae:
        print(f"  Median MAE: {result.median_mae:.5f}")
    if result.median_mfe:
        print(f"  Median MFE: {result.median_mfe:.5f}")
    if result.outcome_distribution:
        print(f"  Distribution: {result.outcome_distribution}")
    if result.sample_size_warning:
        print(f"  WARNING: {result.sample_size_warning}")


def _cmd_critique(finding: str) -> None:
    analyst = _build_analyst()
    assessment = analyst.critique_finding(finding)

    print(f"CRITIC ASSESSMENT: {assessment.finding}")
    print(f"  Status: {assessment.status}")
    print(f"  Confidence: {assessment.initial_confidence:.1%} -> {assessment.adjusted_confidence:.1%}")

    if assessment.challenges:
        print("\n  CHALLENGES:")
        for c in assessment.challenges:
            print(f"    - {c}")

    if assessment.alternative_explanations:
        print("\n  ALTERNATIVE EXPLANATIONS:")
        for a in assessment.alternative_explanations:
            print(f"    - {a}")

    print(f"\n  Economic meaningfulness: {assessment.economic_meaningfulness}")


def _cmd_diagnose_all() -> None:
    analyst = _build_analyst()
    diagnostics = analyst.analyze_all_trades()
    print(f"Diagnosed {len(diagnostics)} trades")

    # Summary by outcome
    outcomes: dict[str, int] = {}
    for d in diagnostics:
        outcomes[d.outcome] = outcomes.get(d.outcome, 0) + 1
    print("\nBy outcome:")
    for outcome, count in sorted(outcomes.items()):
        print(f"  {outcome}: {count}")

    # Summary by primary dimension
    dims: dict[str, int] = {}
    for d in diagnostics:
        dims[d.primary_dimension.value] = dims.get(d.primary_dimension.value, 0) + 1
    print("\nBy primary dimension:")
    for dim, count in sorted(dims.items(), key=lambda x: -x[1]):
        print(f"  {dim}: {count}")

    # Low confidence trades
    low_conf = [d for d in diagnostics if d.confidence < 0.3]
    if low_conf:
        print(f"\n{len(low_conf)} trades with low confidence (< 30%):")
        for d in low_conf[:5]:
            print(f"  [{d.trade_id}] {d.outcome}: {d.primary_diagnosis} (conf: {d.confidence:.1%})")


def _cmd_research() -> None:
    analyst = _build_analyst()
    summary = analyst.get_research_memory_summary()
    print("RESEARCH MEMORY SUMMARY:")
    print(f"  Total hypotheses: {summary['total_hypotheses']}")
    print(f"  Total findings: {summary['total_findings']}")
    print(f"  Total decisions: {summary['total_decisions']}")
    if summary["by_status"]:
        print("  Hypotheses by status:")
        for status, count in summary["by_status"].items():
            if count > 0:
                print(f"    {status}: {count}")


def _cmd_explain(trade_id: str) -> None:
    analyst = _build_analyst()
    result = analyst.explain_trade(trade_id)
    if result is None:
        print(f"Trade {trade_id} not found")
        return

    provider = result.get("provider", "unknown")
    model = result.get("model", "unknown")
    fallback = result.get("fallback", False)

    print(f"TRADE EXPLANATION (provider: {provider}, model: {model}"
          f"{', template fallback' if fallback else ''})")
    print("=" * 70)
    print(result.get("explanation", "No explanation generated."))
    print("=" * 70)

    # Show critic if present
    critic = result.get("critic", {})
    if critic:
        print(f"\nCritic: {critic.get('status', '?')} "
              f"(confidence {critic.get('adjusted_confidence', 0):.1%})")


def _run_default() -> None:
    print("Forex Agent - Trade Failure Analysis System")
    print("Use 'forex_agent <command>' to run analysis.")
    print("Commands: analyze, trade <id>, health, report, validate, anomalies, experiments")
    print("New: diagnose <id>, evidence <id>, similar <id>, critique <finding>, diagnose-all, research, explain <id>")


if __name__ == "__main__":
    sys.exit(main())
