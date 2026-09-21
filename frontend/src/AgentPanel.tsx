import {
  Brain,
  ChevronLeft,
  ChevronRight,
  FlaskConical,
  HeartPulse,
  History,
  Lightbulb,
  Loader2,
  MessageSquareWarning,
  Radar,
  Search,
  ShieldAlert,
  Sparkles,
  Target,
} from "lucide-react";
import { Fragment, useCallback, useEffect, useState } from "react";
import type { ReactNode } from "react";
import { fetchJson } from "./api";

const ROWS_PER_PAGE = 10;

type AgentTab = "analysis" | "health" | "diagnose" | "explain" | "evidence" | "similar" | "critique" | "experiments" | "research" | "anomalies";

type HealthData = {
  score: number;
  grade: string;
  status: string;
  explanation: string;
  metrics: Record<string, unknown>;
  warnings: string[];
  opportunities: string[];
};

type DiagnosticFactor = {
  dimension: string;
  level: string;
  title: string;
  detail: string;
  weight: number;
};

type DiagnosticData = {
  trade_id: string;
  outcome: string;
  r_multiple: number | null;
  primary_dimension: string;
  primary_diagnosis: string;
  evidence_level: string;
  contributing_factors: DiagnosticFactor[];
  protective_factors: DiagnosticFactor[];
  observations: string[];
  hypotheses: string[];
  unknowns: string[];
  confidence: number;
  raw_metrics: Record<string, unknown>;
};

type EvidenceData = {
  trade_id: string;
  baseline: Record<string, unknown>;
  similar_trades: Record<string, unknown>;
  regime: Record<string, unknown>;
  execution: Record<string, unknown>;
  timing: Record<string, unknown>;
  risk: Record<string, unknown>;
  anomalies: Record<string, unknown>[];
  counterfactuals: Record<string, unknown>[];
  statistical_tests: Record<string, unknown>[];
  confidence: number;
};

type SimilarTrade = {
  trade_id: string;
  similarity_score: number;
  outcome: string;
  r_multiple: number;
  entry_time: string;
  instrument: string;
};

type SimilarData = {
  trade_id: string;
  match_count: number;
  definition: string;
  sample_size_warning: string;
  win_rate: number;
  expectancy_r: number;
  outcome_distribution: Record<string, number>;
  matches: SimilarTrade[];
};

type CritiqueData = {
  finding: string;
  status: string;
  initial_confidence: number;
  adjusted_confidence: number;
  challenges: string[];
  sample_size_warning: boolean;
  independence_warning: boolean;
  survivorship_bias_warning: boolean;
  look_ahead_bias_warning: boolean;
  overfitting_warning: boolean;
  multiple_testing_warning: boolean;
  out_of_sample_instability: boolean;
  economic_meaningfulness: string;
  alternative_explanations: string[];
  recommendation: string;
};

type Experiment = {
  hypothesis: string;
  reason: string;
  metric: string;
  statistical_test: string;
  min_sample_size: number;
  acceptance_criteria: string;
  rejection_criteria: string;
  overfitting_risk: string;
  out_of_sample_plan: string;
};

type ResearchSummary = {
  total_findings: number;
  total_hypotheses: number;
  total_experiments: number;
  total_decisions: number;
  total_trades: number;
  findings: Record<string, unknown>[];
  hypotheses: Record<string, unknown>[];
  recent_decisions: Record<string, unknown>[];
};

type AnalysisReport = {
  total_trades: number;
  metrics: Record<string, unknown>;
  health: HealthData;
  failures: Record<string, unknown>[];
  winner_analyses: Record<string, unknown>[];
  anomalies: Record<string, unknown>[];
  regime_analysis: Record<string, unknown>;
  risk_analysis: Record<string, unknown>;
  execution_analysis: Record<string, unknown>;
  recurring_patterns: Record<string, unknown>[];
  alerts: Record<string, unknown>[];
  bootstrap_expectancy_ci: Record<string, unknown>;
};

type DiagnoseAllData = {
  total: number;
  outcomes: Record<string, number>;
  dimensions: Record<string, number>;
  diagnostics: DiagnosticData[];
};

const TABS: { key: AgentTab; label: string; icon: typeof Brain }[] = [
  { key: "analysis", label: "Analysis", icon: Target },
  { key: "health", label: "Health", icon: HeartPulse },
  { key: "diagnose", label: "Diagnose", icon: Search },
  { key: "explain", label: "Explain", icon: Sparkles },
  { key: "evidence", label: "Evidence", icon: ShieldAlert },
  { key: "similar", label: "Similar", icon: History },
  { key: "critique", label: "Critique", icon: MessageSquareWarning },
  { key: "experiments", label: "Experiments", icon: FlaskConical },
  { key: "anomalies", label: "Anomalies", icon: Radar },
  { key: "research", label: "Research", icon: Lightbulb },
];

export function AgentPanel() {
  const [tab, setTab] = useState<AgentTab>("health");
  const [tradeId, setTradeId] = useState("");
  const [finding, setFinding] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [health, setHealth] = useState<HealthData | null>(null);
  const [diagnostic, setDiagnostic] = useState<DiagnosticData | null>(null);
  const [evidence, setEvidence] = useState<EvidenceData | null>(null);
  const [similar, setSimilar] = useState<SimilarData | null>(null);
  const [critique, setCritique] = useState<CritiqueData | null>(null);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [research, setResearch] = useState<ResearchSummary | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisReport | null>(null);
  const [diagnoseAll, setDiagnoseAll] = useState<DiagnoseAllData | null>(null);
  const [explainResult, setExplainResult] = useState<Record<string, unknown> | null>(null);
  const [anomalies, setAnomalies] = useState<Record<string, unknown>[]>([]);

  const loadHealth = useCallback(async () => {
    setLoading(true); setError("");
    try {
      setHealth(await fetchJson<HealthData>("/api/agent/health"));
    } catch (e: unknown) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setLoading(false); }
  }, []);

  const loadDiagnose = useCallback(async () => {
    if (!tradeId.trim()) return;
    setLoading(true); setError("");
    try {
      setDiagnostic(await fetchJson<DiagnosticData>(`/api/agent/diagnose/${encodeURIComponent(tradeId.trim())}`));
    } catch (e: unknown) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setLoading(false); }
  }, [tradeId]);

  const loadExplain = useCallback(async () => {
    if (!tradeId.trim()) return;
    setLoading(true); setError("");
    try {
      setExplainResult(await fetchJson<Record<string, unknown>>(`/api/agent/explain/${encodeURIComponent(tradeId.trim())}`));
    } catch (e: unknown) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setLoading(false); }
  }, [tradeId]);

  const loadEvidence = useCallback(async () => {
    if (!tradeId.trim()) return;
    setLoading(true); setError("");
    try {
      setEvidence(await fetchJson<EvidenceData>(`/api/agent/evidence/${encodeURIComponent(tradeId.trim())}`));
    } catch (e: unknown) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setLoading(false); }
  }, [tradeId]);

  const loadSimilar = useCallback(async () => {
    if (!tradeId.trim()) return;
    setLoading(true); setError("");
    try {
      setSimilar(await fetchJson<SimilarData>(`/api/agent/similar/${encodeURIComponent(tradeId.trim())}`));
    } catch (e: unknown) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setLoading(false); }
  }, [tradeId]);

  const loadCritique = useCallback(async () => {
    if (!finding.trim()) return;
    setLoading(true); setError("");
    try {
      setCritique(await fetchJson<CritiqueData>("/api/agent/critique", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ finding: finding.trim() }),
      }));
    } catch (e: unknown) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setLoading(false); }
  }, [finding]);

  const loadExperiments = useCallback(async () => {
    setLoading(true); setError("");
    try {
      setExperiments(await fetchJson<Experiment[]>("/api/agent/experiments"));
    } catch (e: unknown) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setLoading(false); }
  }, []);

  const loadResearch = useCallback(async () => {
    setLoading(true); setError("");
    try {
      setResearch(await fetchJson<ResearchSummary>("/api/agent/research"));
    } catch (e: unknown) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setLoading(false); }
  }, []);

  const loadAnalysis = useCallback(async () => {
    setLoading(true); setError("");
    try {
      setAnalysis(await fetchJson<AnalysisReport>("/api/agent/analyze"));
      setDiagnoseAll(await fetchJson<DiagnoseAllData>("/api/agent/diagnose-all"));
    } catch (e: unknown) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setLoading(false); }
  }, []);

  const loadAnomalies = useCallback(async () => {
    setLoading(true); setError("");
    try {
      setAnomalies(await fetchJson<Record<string, unknown>[]>("/api/agent/anomalies"));
    } catch (e: unknown) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => {
    loadHealth().catch(() => undefined);
  }, [loadHealth]);

  const needsTradeId = ["diagnose", "explain", "evidence", "similar"].includes(tab);
  const needsFinding = tab === "critique";

  const runTabAction = useCallback(() => {
    if (needsTradeId && !tradeId.trim()) {
      setError("Enter a trade ID before running this action.");
      return;
    }
    if (needsFinding && !finding.trim()) {
      setError("Describe a finding before running the critique.");
      return;
    }

    switch (tab) {
      case "analysis": loadAnalysis(); break;
      case "health": loadHealth(); break;
      case "diagnose": loadDiagnose(); break;
      case "explain": loadExplain(); break;
      case "evidence": loadEvidence(); break;
      case "similar": loadSimilar(); break;
      case "critique": loadCritique(); break;
      case "experiments": loadExperiments(); break;
      case "anomalies": loadAnomalies(); break;
      case "research": loadResearch(); break;
    }
  }, [tab, tradeId, finding, needsTradeId, needsFinding, loadAnalysis, loadHealth, loadDiagnose, loadExplain, loadEvidence, loadSimilar, loadCritique, loadExperiments, loadResearch, loadAnomalies]);

  return (
    <section className="agentView">
      <div className="agentTabs">
        {TABS.map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            className={`agentTab ${tab === key ? "active" : ""}`}
            onClick={() => { setTab(key); setError(""); }}
          >
            <Icon size={15} />
            {label}
          </button>
        ))}
      </div>

      <div className="agentToolbar">
        {needsTradeId && (
          <input
            value={tradeId}
            onChange={(e) => setTradeId(e.target.value)}
            placeholder="Trade ID (e.g. 12345)"
            className="agentInput"
            onKeyDown={(e) => e.key === "Enter" && runTabAction()}
          />
        )}
        {needsFinding && (
          <input
            value={finding}
            onChange={(e) => setFinding(e.target.value)}
            placeholder="Describe a finding to critique..."
            className="agentInput agentInputWide"
            onKeyDown={(e) => e.key === "Enter" && runTabAction()}
          />
        )}
        <button className="agentRunBtn" onClick={runTabAction} disabled={loading}>
          {loading ? <Loader2 size={16} className="spin" /> : <Target size={16} />}
          {loading ? "Running..." : "Run"}
        </button>
      </div>

      {error && <div className="notice">{error}</div>}

      <div className="agentContent">
        {tab === "analysis" && !analysis && !loading && <EmptyState icon={Target} text="Click Run to load the full strategy analysis" />}
        {tab === "analysis" && analysis && <AnalysisView report={analysis} diagnoseAll={diagnoseAll} />}

        {tab === "health" && !health && !loading && <EmptyState icon={HeartPulse} text="Click Run to load strategy health" />}
        {tab === "health" && health && <HealthView data={health} />}

        {tab === "diagnose" && !diagnostic && !loading && <EmptyState icon={Search} text="Enter a trade ID and click Run" />}
        {tab === "diagnose" && diagnostic && <DiagnosticView data={diagnostic} />}

        {tab === "explain" && !explainResult && !loading && <EmptyState icon={Sparkles} text="Enter a trade ID and click Run" />}
        {tab === "explain" && explainResult && <ExplainView data={explainResult} />}

        {tab === "evidence" && !evidence && !loading && <EmptyState icon={ShieldAlert} text="Enter a trade ID and click Run" />}
        {tab === "evidence" && evidence && <EvidenceView data={evidence} />}

        {tab === "similar" && !similar && !loading && <EmptyState icon={History} text="Enter a trade ID and click Run" />}
        {tab === "similar" && similar && <SimilarView data={similar} />}

        {tab === "critique" && !critique && !loading && <EmptyState icon={MessageSquareWarning} text="Describe a finding and click Run" />}
        {tab === "critique" && critique && <CritiqueView data={critique} />}

        {tab === "experiments" && experiments.length === 0 && !loading && <EmptyState icon={FlaskConical} text="Click Run to load proposed experiments" />}
        {tab === "experiments" && experiments.length > 0 && <ExperimentsView data={experiments} />}

        {tab === "anomalies" && anomalies.length === 0 && !loading && <EmptyState icon={Radar} text="Click Run to scan the journal for anomalies" />}
        {tab === "anomalies" && anomalies.length > 0 && <AnomaliesView data={anomalies} />}

        {tab === "research" && !research && !loading && <EmptyState icon={Lightbulb} text="Click Run to load research memory" />}
        {tab === "research" && research && <ResearchView data={research} />}
      </div>
    </section>
  );
}

function EmptyState({ icon: Icon, text }: { icon: typeof Brain; text: string }) {
  return (
    <div className="agentEmpty">
      <Icon size={32} />
      <span>{text}</span>
    </div>
  );
}

function scoreColor(score: number) {
  if (score >= 0.7) return "goodText";
  if (score >= 0.4) return "warnText";
  return "badText";
}

function healthScoreColor(score: number) {
  if (score >= 70) return "goodText";
  if (score >= 40) return "warnText";
  return "badText";
}

function AnalysisView({ report, diagnoseAll }: { report: AnalysisReport; diagnoseAll: DiagnoseAllData | null }) {
  const m = report.metrics;
  const pf = Number(m.profit_factor ?? 0);
  const exp = Number(m.expectancy ?? 0);
  const wr = Number(m.win_rate ?? 0);
  const [alertsPage, setAlertsPage] = useState(0);
  const [recurringPage, setRecurringPage] = useState(0);
  const [anomaliesPage, setAnomaliesPage] = useState(0);
  const [failuresPage, setFailuresPage] = useState(0);
  const [winnersPage, setWinnersPage] = useState(0);

  useEffect(() => {
    setAlertsPage(0);
    setRecurringPage(0);
    setAnomaliesPage(0);
    setFailuresPage(0);
    setWinnersPage(0);
  }, [report]);

  return (
    <div className="agentResult">
      <div className="agentScoreRow">
        <div className={`agentBigScore ${report.health ? healthScoreColor(report.health.score) : ""}`}>
          {report.health ? report.health.score.toFixed(0) : "n/a"}
          <small>/ 100 health</small>
        </div>
        <div className="agentScoreMeta">
          <span className="agentMetaText">{report.total_trades} trades</span>
          <span className="agentMetaText">WR {wr.toFixed(0)}%</span>
          <span className={`agentMetaText ${exp >= 0 ? "goodText" : "badText"}`}>Exp {exp.toFixed(2)}R</span>
          <span className={`agentMetaText ${pf >= 1 ? "goodText" : "badText"}`}>PF {pf.toFixed(2)}</span>
          {report.health && report.health.grade && <span className="agentMetaText">Grade {report.health.grade}</span>}
        </div>
      </div>

      <div className="agentSection">
        <h3>Metrics</h3>
        <dl className="agentDl">
          <MetricItem label="Total P&L" value={money(Number(m.total_pnl ?? 0))} tone={Number(m.total_pnl ?? 0) >= 0 ? "goodText" : "badText"} />
          <MetricItem label="Average R" value={`${Number(m.avg_r_multiple ?? 0).toFixed(2)}R`} tone={Number(m.avg_r_multiple ?? 0) >= 0 ? "goodText" : "badText"} />
          <MetricItem label="Max Drawdown" value={percent(Number(m.max_drawdown_pct ?? 0))} />
          <MetricItem label="Sharpe" value={Number(m.sharpe_ratio ?? 0).toFixed(2)} />
          <MetricItem label="Sortino" value={Number(m.sortino_ratio ?? 0).toFixed(2)} />
          <MetricItem label="Largest Win" value={money(Number(m.largest_win ?? 0))} tone="goodText" />
          <MetricItem label="Largest Loss" value={money(Number(m.largest_loss ?? 0))} tone="badText" />
          <MetricItem label="Consecutive Wins / Losses" value={`${m.consecutive_wins ?? 0} / ${m.consecutive_losses ?? 0}`} />
          <MetricItem label="Avg Duration" value={`${Number(m.avg_duration_minutes ?? 0).toFixed(0)} min`} />
          <MetricItem label="Freq / Day" value={Number(m.trade_frequency_per_day ?? 0).toFixed(2)} />
          <MetricItem label="Avg MFE / MAE (R)" value={`${Number(m.avg_mfe_r ?? 0).toFixed(2)} / ${Number(m.avg_mae_r ?? 0).toFixed(2)}`} />
          <MetricItem label="Recovery Factor" value={Number(m.recovery_factor ?? 0).toFixed(2)} />
          <MetricItem label="Payoff Ratio" value={Number(m.payoff_ratio ?? 0).toFixed(2)} />
        </dl>
      </div>

      {report.health && report.health.warnings.length > 0 && (
        <div className="agentSection">
          <h3>Warnings</h3>
          <ul>{report.health.warnings.map((w, i) => <li key={i} className="badText">{w}</li>)}</ul>
        </div>
      )}

      {report.bootstrap_expectancy_ci && Object.keys(report.bootstrap_expectancy_ci).length > 0 && (
        <div className="agentSection">
          <h3>Expectancy (Bootstrap 95% CI)</h3>
          <p>
            {Number(report.bootstrap_expectancy_ci.statistic ?? 0).toFixed(3)}R
            <span className="agentMetaText">[{Number(report.bootstrap_expectancy_ci.ci_lower ?? 0).toFixed(3)}, {Number(report.bootstrap_expectancy_ci.ci_upper ?? 0).toFixed(3)}]</span>
          </p>
        </div>
      )}

      {report.alerts && report.alerts.length > 0 && (
        <div className="agentSection">
          <h3>Alerts</h3>
          <PaginatedCards
            items={report.alerts}
            page={alertsPage}
            onPageChange={setAlertsPage}
            renderItem={(a, i) => (
              <div key={i} className="agentFactor">
                <span className="agentFactorDim">{String(a.severity ?? "info")}</span>
                <span className="agentFactorTitle">{String(a.alert_type ?? `Alert ${i + 1}`)}</span>
                <p>{String(a.description ?? "")}</p>
              </div>
            )}
          />
        </div>
      )}

      {report.recurring_patterns && report.recurring_patterns.length > 0 && (
        <div className="agentSection">
          <h3>Recurring Patterns</h3>
          <PaginatedCards
            items={report.recurring_patterns}
            page={recurringPage}
            onPageChange={setRecurringPage}
            renderItem={(p, i) => (
              <div key={i} className="agentFactor">
                <span className="agentFactorDim">{String(p.confidence ?? "")}</span>
                <span className="agentFactorTitle">{String(p.pattern ?? "")}</span>
                <p>{String(p.description ?? "")}</p>
              </div>
            )}
          />
        </div>
      )}

      {report.anomalies && report.anomalies.length > 0 && (
        <div className="agentSection">
          <h3>Anomalies</h3>
          <PaginatedCards
            items={report.anomalies}
            page={anomaliesPage}
            onPageChange={setAnomaliesPage}
            renderItem={(a, i) => (
              <div key={i} className="agentFactor">
                <span className={`agentFactorDim ${a.severity === "high" ? "badText" : a.severity === "medium" ? "warnText" : ""}`}>{String(a.severity ?? "")}</span>
                <span className="agentFactorTitle">{String(a.type ?? `Anomaly ${i + 1}`)}</span>
                <p>{String(a.description ?? "")}</p>
              </div>
            )}
          />
        </div>
      )}

      {report.failures && report.failures.length > 0 && (
        <div className="agentSection">
          <h3>Failure Analysis ({report.failures.length} losses)</h3>
          <PaginatedTable
            columns={["Trade", "Category", "Confidence", "Recommendation"]}
            rows={report.failures}
            page={failuresPage}
            onPageChange={setFailuresPage}
            renderRow={(f, i) => (
              <tr key={`${f.trade_id ?? ""}-${i}`}>
                <td>{String(f.trade_id ?? "")}</td>
                <td><span className="agentFactorDim">{String(f.failure_category ?? "")}</span></td>
                <td>{String(f.confidence ?? "")}</td>
                <td>{String(f.recommended_action ?? f.description ?? "")}</td>
              </tr>
            )}
          />
        </div>
      )}

      {report.winner_analyses && report.winner_analyses.length > 0 && (
        <div className="agentSection">
          <h3>Winner Analysis ({report.winner_analyses.length})</h3>
          <PaginatedTable
            columns={["Trade", "Pair", "Quality", "R-Multiple", "Regime", "Factors"]}
            rows={report.winner_analyses}
            page={winnersPage}
            onPageChange={setWinnersPage}
            renderRow={(w, i) => (
              <tr key={`${w.trade_id ?? ""}-${i}`}>
                <td>{String(w.trade_id ?? "")}</td>
                <td>{String(w.symbol ?? "")}</td>
                <td><span className="statusPill filled">{String(w.quality ?? "")}</span></td>
                <td className={Number(w.r_multiple ?? 0) >= 0 ? "goodText" : "badText"}>{Number(w.r_multiple ?? 0).toFixed(2)}</td>
                <td>{String(w.regime ?? "")}</td>
                <td>{(w.factors as unknown[] | undefined ?? []).join(", ")}</td>
              </tr>
            )}
          />
        </div>
      )}

      {report.regime_analysis && Object.keys(report.regime_analysis).length > 0 && (
        <div className="agentSection">
          <h3>Regime Analysis</h3>
          {Object.entries(report.regime_analysis).map(([regime, v]) => {
            if (typeof v !== "object" || v === null) return null;
            const data = v as Record<string, unknown>;
            return (
              <div key={regime} className="agentFactor">
                <span className="agentFactorDim">{regime}</span>
                <span className="agentFactorTitle">{Number(data.closed ?? 0)} closed</span>
                <span className="agentFactorLevel">WR {percent(Number(data.win_rate ?? 0))}</span>
                <span className={`agentFactorLevel ${Number(data.expectancy ?? 0) >= 0 ? "goodText" : "badText"}`}>Exp {Number(data.expectancy ?? 0).toFixed(2)}R</span>
                <span className="agentFactorLevel">{money(Number(data.total_pnl ?? 0))}</span>
              </div>
            );
          })}
        </div>
      )}

      {diagnoseAll && diagnoseAll.total > 0 && (
        <div className="agentSection">
          <h3>Diagnostics Breakdown ({diagnoseAll.total} closed trades)</h3>
          <dl className="agentDl">
            {Object.entries(diagnoseAll.outcomes).map(([k, v]) => (
              <Fragment key={k}>
                <dt>{k}</dt>
                <dd>{String(v)}</dd>
              </Fragment>
            ))}
            {Object.entries(diagnoseAll.dimensions).map(([k, v]) => (
              <Fragment key={k}>
                <dt>{k.replace(/_/g, " ")}</dt>
                <dd>{String(v)}</dd>
              </Fragment>
            ))}
          </dl>
        </div>
      )}

      {report.execution_analysis && Object.keys(report.execution_analysis).length > 0 && (
        <div className="agentSection">
          <h3>Execution</h3>
          <dl className="agentDl">
            {Object.entries(report.execution_analysis).map(([k, v]) => (
              typeof v !== "object" && (
                <Fragment key={k}>
                  <dt>{k.replace(/_/g, " ")}</dt>
                  <dd>{typeof v === "number" ? v.toFixed(2) : String(v)}</dd>
                </Fragment>
              )
            ))}
          </dl>
        </div>
      )}

      {report.risk_analysis && Object.keys(report.risk_analysis).length > 0 && (
        <div className="agentSection">
          <h3>Risk</h3>
          <dl className="agentDl">
            {Object.entries(report.risk_analysis).map(([k, v]) => (
              typeof v !== "object" && (
                <Fragment key={k}>
                  <dt>{k.replace(/_/g, " ")}</dt>
                  <dd>{typeof v === "number" ? v.toFixed(2) : String(v)}</dd>
                </Fragment>
              )
            ))}
          </dl>
        </div>
      )}
    </div>
  );
}

function HealthView({ data }: { data: HealthData }) {
  return (
    <div className="agentResult">
      <div className="agentScoreRow">
        <div className={`agentBigScore ${healthScoreColor(data.score)}`}>
          {data.score.toFixed(0)}
          <small>/ 100</small>
        </div>
        <div className="agentScoreMeta">
          <span className={`pill ${healthStatusPill(data.status)}`}>
            {data.status.replace(/_/g, " ")}
          </span>
          {data.grade && <span className="agentMetaText">Grade: {data.grade}</span>}
          {data.score < 40 && <span className="agentMetaText badText">strategy needs attention</span>}
        </div>
      </div>
      {data.explanation && (
        <div className="agentSection">
          <h3>Summary</h3>
          <p>{data.explanation}</p>
        </div>
      )}
      {data.warnings.length > 0 && (
        <div className="agentSection">
          <h3>Warnings</h3>
          <ul>{data.warnings.map((w, i) => <li key={i} className="badText">{w}</li>)}</ul>
        </div>
      )}
      {data.opportunities.length > 0 && (
        <div className="agentSection">
          <h3>Opportunities</h3>
          <ul>{data.opportunities.map((o, i) => <li key={i} className="goodText">{o}</li>)}</ul>
        </div>
      )}
      {Object.keys(data.metrics).length > 0 && (
        <div className="agentSection">
          <h3>Component Scores</h3>
          <dl className="agentDl">
            {Object.entries(data.metrics).map(([k, v]) => (
              <Fragment key={k}>
                <dt>{k.replace(/_/g, " ")}</dt>
                <dd>{typeof v === "number" ? `${v.toFixed(1)}/100` : String(v)}</dd>
              </Fragment>
            ))}
          </dl>
        </div>
      )}
    </div>
  );
}

function DiagnosticView({ data }: { data: DiagnosticData }) {
  return (
    <div className="agentResult">
      <div className="agentScoreRow">
        <div className={`agentBigScore ${scoreColor(data.confidence)}`}>
          {(data.confidence * 100).toFixed(0)}
          <small>confidence</small>
        </div>
        <div className="agentScoreMeta">
          <span className={`pill ${data.outcome === "win" ? "long" : "short"}`}>{data.outcome}</span>
          <span className="agentMetaText">R: {data.r_multiple?.toFixed(2) ?? "n/a"}</span>
          <span className="agentMetaText">Dimension: {data.primary_dimension}</span>
          <span className="agentMetaText">Evidence: {data.evidence_level}</span>
        </div>
      </div>
      <div className="agentSection">
        <h3>Primary Diagnosis</h3>
        <p>{data.primary_diagnosis}</p>
      </div>
      {data.contributing_factors.length > 0 && (
        <div className="agentSection">
          <h3>Contributing Factors</h3>
          {data.contributing_factors.map((f, i) => (
            <div key={i} className="agentFactor">
              <span className="agentFactorDim">{f.dimension}</span>
              <span className="agentFactorTitle">{f.title}</span>
              <span className="agentFactorLevel">{f.level}</span>
              {typeof f.weight === "number" && f.weight !== 0 && (
                <span className="agentFactorLevel">effect {f.weight.toFixed(2)}</span>
              )}
              <p>{f.detail}</p>
            </div>
          ))}
        </div>
      )}
      {data.protective_factors.length > 0 && (
        <div className="agentSection">
          <h3>Protective Factors</h3>
          {data.protective_factors.map((f, i) => (
            <div key={i} className="agentFactor">
              <span className="agentFactorDim">{f.dimension}</span>
              <span className="agentFactorTitle">{f.title}</span>
              <p>{f.detail}</p>
            </div>
          ))}
        </div>
      )}
      {data.observations.length > 0 && (
        <div className="agentSection">
          <h3>Observations</h3>
          <ul>{data.observations.map((o, i) => <li key={i}>{o}</li>)}</ul>
        </div>
      )}
      {data.hypotheses.length > 0 && (
        <div className="agentSection">
          <h3>Hypotheses</h3>
          <ul>{data.hypotheses.map((h, i) => <li key={i}>{h}</li>)}</ul>
        </div>
      )}
      {data.unknowns.length > 0 && (
        <div className="agentSection">
          <h3>Unknowns</h3>
          <ul>{data.unknowns.map((u, i) => <li key={i}>{u}</li>)}</ul>
        </div>
      )}
    </div>
  );
}

function ExplainView({ data }: { data: Record<string, unknown> }) {
  const explanation = data.explanation as string | undefined;
  const provider = data.provider as string | undefined;
  const model = data.model as string | undefined;

  return (
    <div className="agentResult">
      {provider && (
        <div className="agentScoreRow">
          <div className="agentScoreMeta">
            <span className="agentMetaText">{provider}</span>
            {model && <span className="agentMetaText">{model}</span>}
            {Boolean(data.fallback) && <span className="agentMetaText warnText">template fallback</span>}
          </div>
        </div>
      )}
      {explanation && (
        <div className="agentSection">
          <h3>Explanation</h3>
          <pre className="agentPre">{explanation}</pre>
        </div>
      )}
      {(["diagnostic", "evidence", "critic"] as const).map((key) => {
        const val = data[key];
        if (val === undefined || val === null) return null;
        const label = key.charAt(0).toUpperCase() + key.slice(1);
        return (
          <div key={key} className="agentSection">
            <h3>{label}</h3>
            <pre>{JSON.stringify(val, null, 2)}</pre>
          </div>
        );
      })}
    </div>
  );
}

function EvidenceView({ data }: { data: EvidenceData }) {
  const sections: [string, Record<string, unknown> | Record<string, unknown>[]][] = [
    ["Baseline", data.baseline],
    ["Similar Trades", data.similar_trades],
    ["Regime", data.regime],
    ["Execution", data.execution],
    ["Timing", data.timing],
    ["Risk", data.risk],
    ["Statistical Tests", data.statistical_tests],
  ];
  return (
    <div className="agentResult">
      {data.anomalies.length > 0 && (
        <div className="agentSection">
          <h3>Anomalies</h3>
          <ul>
            {data.anomalies.map((a, i) => (
              <li key={i} className="warnText">
                {String(a.description ?? a.type ?? JSON.stringify(a))}
              </li>
            ))}
          </ul>
        </div>
      )}
      {data.counterfactuals.length > 0 && (
        <div className="agentSection">
          <h3>Counterfactuals</h3>
          {data.counterfactuals.map((cf, i) => (
            <div key={i} className="agentFactor">
              <span className="agentFactorTitle">{String(cf.scenario ?? `Scenario ${i + 1}`)}</span>
              <p>{String(cf.notes ?? "")}</p>
            </div>
          ))}
        </div>
      )}
      {sections.map(([label, obj]) => {
        const resolved = Array.isArray(obj) ? flattenArrayObj(obj) : obj;
        return (
          Object.keys(resolved).length > 0 && (
            <div key={label} className="agentSection">
              <h3>{label}</h3>
              <dl className="agentDl">
                {Object.entries(resolved).map(([k, v]) => (
                  <Fragment key={k}>
                    <dt>{k}</dt>
                    <dd>{typeof v === "number" ? v.toFixed(4) : typeof v === "object" ? JSON.stringify(v) : String(v)}</dd>
                  </Fragment>
                ))}
              </dl>
            </div>
          )
        );
      })}
    </div>
  );
}

function flattenArrayObj(items: Record<string, unknown>[]): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  items.forEach((item, i) => {
    Object.entries(item).forEach(([k, v]) => {
      out[`${i}: ${k}`] = v;
    });
  });
  return out;
}

function SimilarView({ data }: { data: SimilarData }) {
  const [matchesPage, setMatchesPage] = useState(0);

  useEffect(() => {
    setMatchesPage(0);
  }, [data]);

  return (
    <div className="agentResult">
      <div className="agentScoreRow">
        <div className="agentBigScore">
          {data.match_count}
          <small>matches</small>
        </div>
        <div className="agentScoreMeta">
          <span className={`agentMetaText ${data.win_rate >= 0.5 ? "goodText" : "badText"}`}>WR {data.win_rate.toFixed(0)}%</span>
          <span className={`agentMetaText ${data.expectancy_r >= 0 ? "goodText" : "badText"}`}>Exp {data.expectancy_r.toFixed(2)}R</span>
        </div>
      </div>
      <div className="agentSection">
        <h3>Match Definition</h3>
        <p>{data.definition}</p>
        {data.sample_size_warning && <p className="warnText">{data.sample_size_warning}</p>}
      </div>
      {Object.keys(data.outcome_distribution).length > 0 && (
        <div className="agentSection">
          <h3>Outcome Distribution</h3>
          <dl className="agentDl">
            {Object.entries(data.outcome_distribution).map(([k, v]) => (
              <Fragment key={k}>
                <dt>{k.replace(/_/g, " ")}</dt>
                <dd>{String(v)}</dd>
              </Fragment>
            ))}
          </dl>
        </div>
      )}
      {data.matches.length > 0 && (
        <div className="agentSection">
          <h3>Matches</h3>
          <PaginatedTable
            columns={["Trade", "Pair", "Outcome", "R-Multiple", "Entry"]}
            rows={data.matches}
            page={matchesPage}
            onPageChange={setMatchesPage}
            renderRow={(m, i) => (
              <tr key={`${m.trade_id}-${i}`}>
                <td>{m.trade_id}</td>
                <td>{m.instrument}</td>
                <td><span className={`pill ${m.outcome === "win" ? "long" : "short"}`}>{m.outcome}</span></td>
                <td className={m.r_multiple >= 0 ? "goodText" : "badText"}>{m.r_multiple.toFixed(2)}</td>
                <td>{m.entry_time ? new Date(m.entry_time).toLocaleString() : "n/a"}</td>
              </tr>
            )}
          />
        </div>
      )}
    </div>
  );
}

function CritiqueView({ data }: { data: CritiqueData }) {
  const warnings: string[] = [];
  if (data.sample_size_warning) warnings.push("Small sample size");
  if (data.independence_warning) warnings.push("Non-independent trades");
  if (data.survivorship_bias_warning) warnings.push("Survivorship bias");
  if (data.look_ahead_bias_warning) warnings.push("Look-ahead bias");
  if (data.overfitting_warning) warnings.push("Overfitting risk");
  if (data.multiple_testing_warning) warnings.push("Multiple-testing inflation");
  if (data.out_of_sample_instability) warnings.push("Out-of-sample instability");

  return (
    <div className="agentResult">
      <div className="agentScoreRow">
        <div className={`agentBigScore ${scoreColor(data.adjusted_confidence)}`}>
          {(data.adjusted_confidence * 100).toFixed(0)}
          <small>adjusted confidence</small>
        </div>
        <div className="agentScoreMeta">
          <span className={`pill ${data.status === "supported" ? "long" : data.status === "inconclusive" ? "warn" : "short"}`}>
            {data.status || "unknown"}
          </span>
          <span className="agentMetaText">Initial: {(data.initial_confidence * 100).toFixed(0)}%</span>
          <span className="agentMetaText">Adjustment: {((data.adjusted_confidence - data.initial_confidence) * 100).toFixed(0)}pp</span>
        </div>
      </div>
      <div className="agentSection">
        <h3>Finding</h3>
        <p>{data.finding}</p>
      </div>
      {warnings.length > 0 && (
        <div className="agentSection">
          <h3>Warnings</h3>
          <ul>{warnings.map((w, i) => <li key={i} className="badText">{w}</li>)}</ul>
        </div>
      )}
      {data.challenges.length > 0 && (
        <div className="agentSection">
          <h3>Challenges</h3>
          <ul>{data.challenges.map((c, i) => <li key={i}>{c}</li>)}</ul>
        </div>
      )}
      {data.economic_meaningfulness && (
        <div className="agentSection">
          <h3>Economic Meaningfulness</h3>
          <p>{data.economic_meaningfulness}</p>
        </div>
      )}
      {data.alternative_explanations.length > 0 && (
        <div className="agentSection">
          <h3>Alternative Explanations</h3>
          <ul>{data.alternative_explanations.map((a, i) => <li key={i}>{a}</li>)}</ul>
        </div>
      )}
      <div className="agentSection">
        <h3>Recommendation</h3>
        <p>{data.recommendation}</p>
      </div>
    </div>
  );
}

function ExperimentsView({ data }: { data: Experiment[] }) {
  const [page, setPage] = useState(0);

  useEffect(() => {
    setPage(0);
  }, [data]);

  return (
    <div className="agentResult">
      <PaginatedCards
        items={data}
        page={page}
        onPageChange={setPage}
        renderItem={(exp, i) => (
          <div key={i} className="agentExperiment">
            <h3>{exp.hypothesis}</h3>
            <dl className="agentDl">
              <dt>Reason</dt><dd>{exp.reason}</dd>
              <dt>Metric</dt><dd>{exp.metric}</dd>
              <dt>Statistical Test</dt><dd>{exp.statistical_test}</dd>
              <dt>Min Sample Size</dt><dd>{exp.min_sample_size}</dd>
              <dt>Acceptance</dt><dd>{exp.acceptance_criteria}</dd>
              <dt>Rejection</dt><dd>{exp.rejection_criteria}</dd>
              <dt>Overfitting Risk</dt><dd>{exp.overfitting_risk}</dd>
              <dt>Out-of-Sample Plan</dt><dd>{exp.out_of_sample_plan}</dd>
            </dl>
          </div>
        )}
      />
    </div>
  );
}

function ResearchView({ data }: { data: ResearchSummary }) {
  const [decisionsPage, setDecisionsPage] = useState(0);
  const [hypothesesPage, setHypothesesPage] = useState(0);

  useEffect(() => {
    setDecisionsPage(0);
    setHypothesesPage(0);
  }, [data]);

  return (
    <div className="agentResult">
      <div className="agentScoreRow">
        <div className="agentBigScore">
          {data.total_findings}
          <small>findings</small>
        </div>
        <div className="agentBigScore">
          {data.total_hypotheses}
          <small>hypotheses</small>
        </div>
        <div className="agentBigScore">
          {data.total_experiments}
          <small>experiments</small>
        </div>
        <div className="agentBigScore">
          {data.total_decisions}
          <small>decisions</small>
        </div>
      </div>
      {data.recent_decisions.length > 0 && (
        <div className="agentSection">
          <h3>Recent Decisions</h3>
          <PaginatedTable
            columns={["Time", "Type", "Description"]}
            rows={data.recent_decisions}
            page={decisionsPage}
            onPageChange={setDecisionsPage}
            renderRow={(d, i) => (
              <tr key={i}>
                <td>{String(d.timestamp ?? "")}</td>
                <td>{String(d.type ?? "")}</td>
                <td>{String(d.description ?? d.finding ?? d.hypothesis ?? "")}</td>
              </tr>
            )}
          />
        </div>
      )}
      {data.hypotheses.length > 0 && (
        <div className="agentSection">
          <h3>Hypotheses</h3>
          <PaginatedCards
            items={data.hypotheses}
            page={hypothesesPage}
            onPageChange={setHypothesesPage}
            renderItem={(h, i) => (
              <div key={i} className="agentFactor">
                <span className="agentFactorTitle">{String(h.status ?? "")}</span>
                <p>{String(h.hypothesis ?? h.description ?? "")}</p>
              </div>
            )}
          />
        </div>
      )}
    </div>
  );
}

function healthStatusPill(status: string) {
  switch (status) {
    case "healthy":
    case "normal_variance":
      return "long";
    case "watch":
      return "warn";
    case "degraded":
    case "critical":
      return "short";
    default:
      return "neutral";
  }
}

function AnomaliesView({ data }: { data: Record<string, unknown>[] }) {
  const [page, setPage] = useState(0);

  useEffect(() => {
    setPage(0);
  }, [data]);

  return (
    <div className="agentResult">
      {data.length === 0 ? (
        <div className="agentEmpty">
          <Radar size={32} />
          <span>No anomalies detected in the journal</span>
        </div>
      ) : (
        <div className="agentSection">
          <h3>{data.length} Anomalies</h3>
          <PaginatedCards
            items={data}
            page={page}
            onPageChange={setPage}
            renderItem={(a, i) => (
              <div key={i} className="agentFactor">
                <span className={`agentFactorDim ${a.severity === "high" ? "badText" : a.severity === "medium" ? "warnText" : ""}`}>
                  {String(a.severity ?? "info")}
                </span>
                <span className="agentFactorTitle">{String(a.type ?? `Anomaly ${i + 1}`)}</span>
                {typeof a.count === "number" && <span className="agentFactorLevel">{String(a.count)}</span>}
                <p>{String(a.description ?? "")}</p>
              </div>
            )}
          />
        </div>
      )}
    </div>
  );
}

function MetricItem({ label, value, tone = "" }: { label: string; value: string; tone?: string }) {
  return (
    <>
      <dt>{label}</dt>
      <dd className={tone}>{value}</dd>
    </>
  );
}

function PaginatedTable<T>({
  columns,
  rows,
  page,
  onPageChange,
  renderRow,
}: {
  columns: string[];
  rows: T[];
  page: number;
  onPageChange: (page: number) => void;
  renderRow: (item: T, index: number) => ReactNode;
}) {
  const totalPages = Math.max(1, Math.ceil(rows.length / ROWS_PER_PAGE));
  const safePage = Math.min(page, totalPages - 1);
  const start = safePage * ROWS_PER_PAGE;
  const visibleRows = rows.slice(start, start + ROWS_PER_PAGE);

  return (
    <>
      <div className="tableWrap">
        <table className="agentTable">
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column}>{column}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visibleRows.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="empty">No records</td>
              </tr>
            ) : (
              visibleRows.map((item, index) => renderRow(item, start + index))
            )}
          </tbody>
        </table>
      </div>
      {rows.length > ROWS_PER_PAGE && (
        <PaginationControls page={safePage} totalPages={totalPages} itemCount={rows.length} onPageChange={onPageChange} />
      )}
    </>
  );
}

function PaginatedCards<T>({
  items,
  page,
  onPageChange,
  renderItem,
}: {
  items: T[];
  page: number;
  onPageChange: (page: number) => void;
  renderItem: (item: T, index: number) => ReactNode;
}) {
  const totalPages = Math.max(1, Math.ceil(items.length / ROWS_PER_PAGE));
  const safePage = Math.min(page, totalPages - 1);
  const start = safePage * ROWS_PER_PAGE;
  const visibleItems = items.slice(start, start + ROWS_PER_PAGE);

  return (
    <>
      {visibleItems.map((item, index) => renderItem(item, start + index))}
      {items.length > ROWS_PER_PAGE && (
        <PaginationControls page={safePage} totalPages={totalPages} itemCount={items.length} onPageChange={onPageChange} />
      )}
    </>
  );
}

function PaginationControls({
  page,
  totalPages,
  itemCount,
  onPageChange,
}: {
  page: number;
  totalPages: number;
  itemCount: number;
  onPageChange: (page: number) => void;
}) {
  const safePage = Math.min(page, totalPages - 1);
  const start = safePage * ROWS_PER_PAGE;
  const rangeStart = itemCount > 0 ? start + 1 : 0;
  const rangeEnd = Math.min(start + ROWS_PER_PAGE, itemCount);

  const pageNumbers: number[] = [];
  const maxVisible = 5;
  let pageStart = Math.max(0, safePage - Math.floor(maxVisible / 2));
  let pageEnd = Math.min(totalPages, pageStart + maxVisible);
  if (pageEnd - pageStart < maxVisible) {
    pageStart = Math.max(0, pageEnd - maxVisible);
  }
  for (let i = pageStart; i < pageEnd; i++) {
    pageNumbers.push(i);
  }

  return (
    <div className="pagination">
      <span className="paginationInfo">
        {rangeStart}–{rangeEnd} of {itemCount}
      </span>
      <div className="paginationControls">
        <button
          className="paginationBtn"
          disabled={safePage === 0}
          onClick={() => onPageChange(safePage - 1)}
        >
          <ChevronLeft size={16} />
        </button>
        {pageNumbers[0] > 0 && (
          <>
            <button className="paginationPage" onClick={() => onPageChange(0)}>1</button>
            {pageNumbers[0] > 1 && <span className="paginationEllipsis">…</span>}
          </>
        )}
        {pageNumbers.map((i) => (
          <button
            key={i}
            className={`paginationPage ${i === safePage ? "active" : ""}`}
            onClick={() => onPageChange(i)}
          >
            {i + 1}
          </button>
        ))}
        {pageNumbers[pageNumbers.length - 1] < totalPages - 1 && (
          <>
            {pageNumbers[pageNumbers.length - 1] < totalPages - 2 && (
              <span className="paginationEllipsis">…</span>
            )}
            <button className="paginationPage" onClick={() => onPageChange(totalPages - 1)}>
              {totalPages}
            </button>
          </>
        )}
        <button
          className="paginationBtn"
          disabled={safePage >= totalPages - 1}
          onClick={() => onPageChange(safePage + 1)}
        >
          <ChevronRight size={16} />
        </button>
      </div>
    </div>
  );
}

function money(value: number) {
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(value || 0);
}

function percent(value: number) {
  return `${((value || 0) * 100).toFixed(2)}%`;
}
