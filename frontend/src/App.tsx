import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  Bot,
  ChartNoAxesCombined,
  ChevronLeft,
  ChevronRight,
  CirclePause,
  CircleStop,
  Clock3,
  Eye,
  EyeOff,
  KeyRound,
  LayoutDashboard,
  Loader2,
  LockKeyhole,
  Newspaper,
  Play,
  PlugZap,
  RefreshCw,
  Search,
  Settings2,
  ShieldCheck,
  ShieldAlert,
  TrendingUp,
  WalletCards,
  Wifi,
  X,
  Zap,
} from "lucide-react";
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ElementType, ReactNode } from "react";
import {
  ApiError,
  clearStoredApiKey,
  fetchJson,
  isLiveSnapshot,
  readStoredApiKey,
  storeApiKey,
  websocketUrl,
} from "./api";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const AgentPanel = lazy(async () => ({ default: (await import("./AgentPanel")).AgentPanel }));

type BotStatus = {
  state: string;
  reason: string;
  updated_at: string;
  worker_task_running?: boolean;
  demo_only?: boolean;
  live_release_approved?: boolean;
};

type EquityPoint = { timestamp: string; equity: number; balance: number; margin_used: number; open_positions: number };
type Position = { instrument: string; side: string; units: number; avg_price: number; unrealized_pl: number; margin_used: number; price: number | null; estimated_daily_financing: number; updated_at: string };
type Trade = { broker_trade_id: string; instrument: string; side: string; units: number; entry_time: string | null; entry_price: number | null; exit_time: string | null; exit_price: number | null; realized_pl: number; financing: number; state: string; exit_reason: string | null };
type Signal = { id: number; timestamp: string; instrument: string; side: string | null; status: string; reason: string; score: number; entry_price: number | null; stop_loss: number | null; take_profit: number | null; risk_amount: number | null };
type Order = { id: number; client_order_id: string; timestamp: string; instrument: string; side: string; units: number; order_type: string; status: string; broker_order_id: string | null; broker_trade_id: string | null; error: string | null; risk_amount: number };
type Performance = { trade_count: number; open_trade_count: number; win_rate: number; profit_factor: number | null; total_pnl: number; max_drawdown: { amount: number; pct: number }; sharpe_ratio: number; sortino_ratio: number; average_r_multiple: number; financing: number };
type BrokerConfig = { provider?: string; server?: string; configured?: boolean; demo_only?: boolean; terminal_path_configured?: boolean; login_hint?: string };
type ConfigPayload = { instruments?: string[]; dashboard_badge?: string; strategy?: Record<string, unknown>; risk?: Record<string, unknown>; broker?: BrokerConfig };
type Monitoring = { stale_prices: Record<string, number>; stale_scan_age_seconds: number | null; db_lock_retries: number; db_lock_alert: boolean; broker_connected: boolean; unknown_orders: number; risk_halted: boolean; risk_halt_reason: string | null; news_data_age_seconds: number | null; news_data_fresh: boolean | null; clock_health: { status?: string; skew_seconds?: number; reason?: string }; recent_alerts: Array<{ severity?: string; message?: string; timestamp?: string }>; uptime_seconds: number };
type NewsEvent = { event_id: string; name: string; currency: string; impact: string; scheduled_at: string; status: string };
type NewsResponse = { state: string; stale: boolean; last_updated: string | null; age_seconds: number | null; events: NewsEvent[]; event_count: number; blackouts: Record<string, string> };
type LiveSnapshot = { status: BotStatus; positions: Position[]; equity_curve: EquityPoint[]; performance: Performance; recent_trades: Trade[]; recent_signals: Signal[]; config: ConfigPayload };
type DashboardData = { status: BotStatus; positions: Position[]; trades: Trade[]; equity: EquityPoint[]; performance: Performance; config: ConfigPayload; signals: Signal[]; orders: Order[]; monitoring: Monitoring | null; news: NewsResponse | null };
type ConnectionState = "disconnected" | "connecting" | "live" | "polling";
type View = "dashboard" | "agent";
type ControlAction = "start" | "pause" | "stop";

const PAGE_SIZE = 12;
const EMPTY_PERFORMANCE: Performance = { trade_count: 0, open_trade_count: 0, win_rate: 0, profit_factor: null, total_pnl: 0, max_drawdown: { amount: 0, pct: 0 }, sharpe_ratio: 0, sortino_ratio: 0, average_r_multiple: 0, financing: 0 };
const EMPTY_DATA: DashboardData = { status: { state: "offline", reason: "Waiting for a secure connection", updated_at: "" }, positions: [], trades: [], equity: [], performance: EMPTY_PERFORMANCE, config: {}, signals: [], orders: [], monitoring: null, news: null };

function isAuthenticationError(issue: unknown): boolean {
  if (issue instanceof ApiError) return issue.status === 401 || issue.status === 403;
  const message = issue instanceof Error ? issue.message : String(issue);
  return /api key|unauthori[sz]ed|\b401\b|\b403\b/i.test(message);
}

export function App() {
  const [view, setView] = useState<View>("dashboard");
  const [data, setData] = useState<DashboardData>(EMPTY_DATA);
  const [apiKey, setApiKey] = useState(readStoredApiKey);
  const [connectionEnabled, setConnectionEnabled] = useState(Boolean(readStoredApiKey()));
  const [connection, setConnection] = useState<ConnectionState>(readStoredApiKey() ? "connecting" : "disconnected");
  const [authFailed, setAuthFailed] = useState(false);
  const [showKey, setShowKey] = useState(false);
  const [notice, setNotice] = useState("");
  const [syncing, setSyncing] = useState(false);
  const [lastSyncedAt, setLastSyncedAt] = useState<Date | null>(null);
  const [socketNonce, setSocketNonce] = useState(0);
  const [pendingAction, setPendingAction] = useState<ControlAction | null>(null);
  const [controlInFlight, setControlInFlight] = useState<ControlAction | null>(null);
  const [instrumentFilter, setInstrumentFilter] = useState("");
  const [outcomeFilter, setOutcomeFilter] = useState("");
  const [tradesPage, setTradesPage] = useState(0);
  const [signalsPage, setSignalsPage] = useState(0);
  const [ordersPage, setOrdersPage] = useState(0);
  const [chartWindow, setChartWindow] = useState<"All" | "24H" | "7D">("All");
  const filtersRef = useRef({ instrument: "", outcome: "" });

  const handleIssue = useCallback((issue: unknown) => {
    const message = issue instanceof Error ? issue.message : String(issue);
    if (isAuthenticationError(issue)) {
      setAuthFailed(true);
      setConnection("disconnected");
      setNotice("The API key was rejected. Check FX_API_KEY and reconnect.");
      return;
    }
    setNotice(message || "The dashboard could not update. Please try again.");
  }, []);

  const applySnapshot = useCallback((snapshot: LiveSnapshot) => {
    setData((current) => ({ ...current, status: snapshot.status, positions: snapshot.positions, equity: snapshot.equity_curve, performance: snapshot.performance, config: snapshot.config, signals: snapshot.recent_signals, trades: filtersRef.current.instrument || filtersRef.current.outcome ? current.trades : snapshot.recent_trades }));
    setLastSyncedAt(new Date());
  }, []);

  const loadCore = useCallback(async (showSpinner = false) => {
    if (!readStoredApiKey()) return;
    if (showSpinner) setSyncing(true);
    try {
      const results = await Promise.allSettled([
        fetchJson<BotStatus>("/api/status"), fetchJson<Position[]>("/api/positions"), fetchJson<EquityPoint[]>("/api/equity?limit=500"), fetchJson<Performance>("/api/performance"),
        fetchJson<ConfigPayload>("/api/config"), fetchJson<Signal[]>("/api/signals?limit=100"), fetchJson<Order[]>("/api/orders?limit=100"), fetchJson<Monitoring>("/api/monitoring"),
      ]);
      const valueAt = <T,>(index: number): T | undefined => { const result = results[index]; return result.status === "fulfilled" ? result.value as T : undefined; };
      const failures = results.filter((result): result is PromiseRejectedResult => result.status === "rejected");
      if (failures.length === results.length) throw failures[0].reason;
      setData((current) => ({ ...current, status: valueAt<BotStatus>(0) ?? current.status, positions: valueAt<Position[]>(1) ?? current.positions, equity: valueAt<EquityPoint[]>(2) ?? current.equity, performance: valueAt<Performance>(3) ?? current.performance, config: valueAt<ConfigPayload>(4) ?? current.config, signals: valueAt<Signal[]>(5) ?? current.signals, orders: valueAt<Order[]>(6) ?? current.orders, monitoring: valueAt<Monitoring>(7) ?? current.monitoring }));
      setLastSyncedAt(new Date());
      setNotice(failures.length ? "Some operational data is temporarily unavailable. The remaining dashboard data is current." : "");
    } finally { if (showSpinner) setSyncing(false); }
  }, []);

  const loadTrades = useCallback(async () => {
    if (!readStoredApiKey()) return;
    const params = new URLSearchParams({ limit: "300" });
    if (instrumentFilter.trim()) params.set("instrument", instrumentFilter.trim().toUpperCase());
    if (outcomeFilter) params.set("outcome", outcomeFilter);
    const trades = await fetchJson<Trade[]>(`/api/trades?${params.toString()}`, { cache: "no-store" });
    setData((current) => ({ ...current, trades }));
    setTradesPage(0);
  }, [instrumentFilter, outcomeFilter]);

  const loadInitialTrades = useCallback(async () => {
    if (!readStoredApiKey()) return;
    const trades = await fetchJson<Trade[]>("/api/trades?limit=300", { cache: "no-store" });
    setData((current) => ({ ...current, trades }));
    setTradesPage(0);
  }, []);

  const loadNews = useCallback(async () => {
    if (!readStoredApiKey()) return;
    const news = await fetchJson<NewsResponse>("/api/news?limit=4");
    setData((current) => ({ ...current, news }));
  }, []);

  const syncEverything = useCallback(async () => {
    try { await Promise.all([loadCore(true), loadTrades(), loadNews()]); setNotice(""); } catch (issue) { handleIssue(issue); }
  }, [handleIssue, loadCore, loadNews, loadTrades]);

  useEffect(() => { filtersRef.current = { instrument: instrumentFilter, outcome: outcomeFilter }; }, [instrumentFilter, outcomeFilter]);
  useEffect(() => {
    if (!connectionEnabled || authFailed) return;
    Promise.all([loadCore(true), loadInitialTrades(), loadNews()]).catch(handleIssue);
  }, [authFailed, connectionEnabled, handleIssue, loadCore, loadInitialTrades, loadNews]);

  useEffect(() => {
    if (!connectionEnabled || authFailed) { setConnection("disconnected"); return; }
    let socket: WebSocket | null = null;
    let reconnectTimer: number | undefined;
    let disposed = false;
    let backoff = 2_000;
    const connectSocket = () => {
      if (disposed || authFailed) return;
      setConnection("connecting");
      try { socket = new WebSocket(websocketUrl("/ws/live")); } catch { setConnection("polling"); reconnectTimer = window.setTimeout(connectSocket, backoff); return; }
      socket.onopen = () => { backoff = 2_000; setConnection("live"); };
      socket.onmessage = (event) => { try { const payload: unknown = JSON.parse(event.data); if (isLiveSnapshot(payload)) applySnapshot(payload as LiveSnapshot); } catch { /* Ignore malformed stream payloads. */ } };
      socket.onerror = () => socket?.close();
      socket.onclose = (event) => {
        socket = null;
        if (disposed) return;
        if (event.code === 1008) { setAuthFailed(true); setConnection("disconnected"); setNotice("The API key was rejected. Check FX_API_KEY and reconnect."); return; }
        setConnection("polling"); reconnectTimer = window.setTimeout(connectSocket, backoff); backoff = Math.min(backoff * 1.8, 30_000);
      };
    };
    connectSocket();
    return () => { disposed = true; if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer); socket?.close(); };
  }, [applySnapshot, authFailed, connectionEnabled, socketNonce]);

  useEffect(() => {
    if (!connectionEnabled || authFailed || connection === "live") return;
    const interval = window.setInterval(() => loadCore().catch(handleIssue), 5_000);
    return () => window.clearInterval(interval);
  }, [authFailed, connection, connectionEnabled, handleIssue, loadCore]);
  useEffect(() => {
    if (!connectionEnabled || authFailed) return;
    const interval = window.setInterval(() => loadCore().catch(handleIssue), 30_000);
    return () => window.clearInterval(interval);
  }, [authFailed, connectionEnabled, handleIssue, loadCore]);

  const connect = useCallback(() => {
    const key = apiKey.trim();
    if (!key) { setNotice("Enter the FX_API_KEY configured by the backend before connecting."); return; }
    storeApiKey(key); setAuthFailed(false); setConnectionEnabled(true); setConnection("connecting"); setSocketNonce((value) => value + 1);
  }, [apiKey]);

  const forgetKey = useCallback(() => { clearStoredApiKey(); setApiKey(""); setConnectionEnabled(false); setAuthFailed(false); setData(EMPTY_DATA); setNotice("The locally saved key has been removed."); }, []);
  const runControl = useCallback(async (action: ControlAction) => {
    const key = apiKey.trim();
    if (!key) { setNotice("Connect with an API key before sending a control command."); return; }
    setControlInFlight(action);
    try {
      const nextStatus = await fetchJson<BotStatus>(`/api/control/${action}`, { method: "POST", headers: { "Content-Type": "application/json", "X-API-Key": key }, body: JSON.stringify({ reason: `dashboard_${action}` }) });
      setData((current) => ({ ...current, status: nextStatus })); setNotice(""); await loadCore();
    } catch (issue) { handleIssue(issue); } finally { setControlInFlight(null); setPendingAction(null); }
  }, [apiKey, handleIssue, loadCore]);

  const latestEquity = data.equity[data.equity.length - 1];
  const chartData = useMemo(() => filterChartData(data.equity, chartWindow), [chartWindow, data.equity]);
  const closedTrades = useMemo(() => data.trades.filter((trade) => trade.state.toLowerCase() === "closed"), [data.trades]);
  const wins = useMemo(() => closedTrades.filter((trade) => trade.realized_pl + trade.financing > 0).length, [closedTrades]);
  const losses = useMemo(() => closedTrades.filter((trade) => trade.realized_pl + trade.financing < 0).length, [closedTrades]);
  const totalMargin = useMemo(() => data.positions.reduce((sum, position) => sum + position.margin_used, 0), [data.positions]);
  const activeBlackouts = Object.keys(data.news?.blackouts ?? {});
  const statusTone = toneForStatus(data.status.state);
  const canControl = connectionEnabled && !authFailed && Boolean(apiKey.trim());

  return <div className="appShell">
    <header className="appHeader">
      <button className="brand" onClick={() => setView("dashboard")} aria-label="Open trading dashboard"><span className="brandMark"><ChartNoAxesCombined size={20} /></span><span><strong>Northstar FX</strong><small>Forward test terminal</small></span></button>
      <nav className="primaryNav" aria-label="Workspace navigation"><button className={view === "dashboard" ? "active" : ""} onClick={() => setView("dashboard")}><LayoutDashboard size={16} /> Overview</button><button className={view === "agent" ? "active" : ""} onClick={() => setView("agent")}><Bot size={16} /> Analyst</button></nav>
      <div className="headerActions"><ConnectionBadge connection={connection} /><button className="iconButton" title="Refresh dashboard" aria-label="Refresh dashboard" onClick={syncEverything} disabled={!connectionEnabled || syncing}><RefreshCw size={17} className={syncing ? "spin" : ""} /></button></div>
    </header>
    <main className="workspace">
      {notice && <div className="notice" role="alert"><AlertTriangle size={17} /><span>{notice}</span><button className="noticeClose" onClick={() => setNotice("")} aria-label="Dismiss message"><X size={16} /></button></div>}
      {!connectionEnabled ? <ConnectionSetup apiKey={apiKey} showKey={showKey} onApiKeyChange={setApiKey} onShowKeyChange={() => setShowKey((visible) => !visible)} onConnect={connect} connecting={syncing} /> : <>
        <section className="commandBar" aria-label="Bot controls"><div className="commandTitle"><div className={`stateGlyph ${statusTone}`}><Activity size={18} /></div><div><span className="sectionKicker">Execution status</span><h1>{sentenceCase(data.status.state)}</h1><p>{data.status.reason || "No execution reason reported by the service."}</p></div></div><div className="commandMeta"><span><Clock3 size={14} /> {lastSyncedAt ? `Synced ${timeAgo(lastSyncedAt)}` : "Awaiting first sync"}</span><span><ShieldCheck size={14} /> {data.config.dashboard_badge ?? "Forward test"}</span></div><div className="controlActions"><button className="controlButton primary" onClick={() => setPendingAction("start")} disabled={!canControl || controlInFlight !== null || data.status.state.toLowerCase() === "running"}>{controlInFlight === "start" ? <Loader2 size={16} className="spin" /> : <Play size={16} />} Start</button><button className="controlButton" onClick={() => setPendingAction("pause")} disabled={!canControl || controlInFlight !== null || data.status.state.toLowerCase() !== "running"}>{controlInFlight === "pause" ? <Loader2 size={16} className="spin" /> : <CirclePause size={16} />} Pause</button><button className="controlButton destructive" onClick={() => setPendingAction("stop")} disabled={!canControl || controlInFlight !== null || data.status.state.toLowerCase() === "stopped"}>{controlInFlight === "stop" ? <Loader2 size={16} className="spin" /> : <CircleStop size={16} />} Stop</button></div></section>
        <section className="securityStrip"><div className="securityKey"><KeyRound size={16} /><input value={apiKey} onChange={(event) => { setApiKey(event.target.value); setAuthFailed(false); }} type={showKey ? "text" : "password"} placeholder="FX_API_KEY" aria-label="FX API key" onKeyDown={(event) => event.key === "Enter" && connect()} /><button className="inlineIcon" onClick={() => setShowKey((visible) => !visible)} title={showKey ? "Hide API key" : "Show API key"} aria-label={showKey ? "Hide API key" : "Show API key"}>{showKey ? <EyeOff size={16} /> : <Eye size={16} />}</button></div><span className="securityHint"><LockKeyhole size={14} /> Stored only in this browser</span><button className="textButton" onClick={connect} disabled={syncing}>{syncing ? "Connecting…" : "Reconnect"}</button><button className="textButton muted" onClick={forgetKey}>Forget key</button></section>
        {view === "agent" ? <section className="analystWorkspace"><div className="pageHeading"><div><span className="sectionKicker">Research workspace</span><h2>Trade analyst</h2><p>Interrogate outcomes, evidence, anomalies, and experiment proposals without exposing trading controls.</p></div><div className="headingBadge"><Bot size={16} /> Read-only analysis</div></div><Suspense fallback={<div className="loadingPanel"><Loader2 className="spin" size={22} /> Loading analyst tools…</div>}><AgentPanel /></Suspense></section> : <DashboardView data={data} latestEquity={latestEquity} chartData={chartData} chartWindow={chartWindow} onChartWindow={setChartWindow} wins={wins} losses={losses} totalMargin={totalMargin} activeBlackouts={activeBlackouts} instrumentFilter={instrumentFilter} outcomeFilter={outcomeFilter} onInstrumentFilter={setInstrumentFilter} onOutcomeFilter={setOutcomeFilter} onApplyTradeFilters={() => loadTrades().catch(handleIssue)} tradesPage={tradesPage} onTradesPage={setTradesPage} signalsPage={signalsPage} onSignalsPage={setSignalsPage} ordersPage={ordersPage} onOrdersPage={setOrdersPage} />}
      </>}
    </main>
    {pendingAction && <ConfirmationDialog action={pendingAction} running={controlInFlight === pendingAction} onCancel={() => setPendingAction(null)} onConfirm={() => runControl(pendingAction)} />}
  </div>;
}

function ConnectionSetup({ apiKey, showKey, onApiKeyChange, onShowKeyChange, onConnect, connecting }: { apiKey: string; showKey: boolean; onApiKeyChange: (value: string) => void; onShowKeyChange: () => void; onConnect: () => void; connecting: boolean }) {
  return <section className="connectionSetup"><div className="setupCopy"><span className="setupOrb"><Zap size={23} /></span><span className="sectionKicker">Secure operator access</span><h1>One calm place to run your forward test.</h1><p>Connect this dashboard to the local FX service to monitor positions, execution quality, system health, and market-event risk in real time.</p><div className="setupFeatures"><span><Wifi size={15} /> Live stream with polling fallback</span><span><ShieldCheck size={15} /> Authenticated control plane</span><span><Newspaper size={15} /> News blackout visibility</span></div></div><form className="connectCard" onSubmit={(event) => { event.preventDefault(); onConnect(); }}><div className="connectCardIcon"><PlugZap size={22} /></div><h2>Connect the terminal</h2><p>Enter the <code>FX_API_KEY</code> from your backend environment.</p><label htmlFor="setup-api-key">API key</label><div className="passwordField"><KeyRound size={17} /><input id="setup-api-key" value={apiKey} onChange={(event) => onApiKeyChange(event.target.value)} type={showKey ? "text" : "password"} autoComplete="current-password" placeholder="Paste FX_API_KEY" autoFocus /><button type="button" className="inlineIcon" onClick={onShowKeyChange} aria-label={showKey ? "Hide API key" : "Show API key"}>{showKey ? <EyeOff size={17} /> : <Eye size={17} />}</button></div><button className="connectButton" type="submit" disabled={!apiKey.trim() || connecting}>{connecting ? <Loader2 size={17} className="spin" /> : <PlugZap size={17} />}{connecting ? "Verifying connection…" : "Connect securely"}</button><small><LockKeyhole size={13} /> Kept in your browser’s local storage; never sent anywhere except the configured FX API.</small></form></section>;
}

function DashboardView({ data, latestEquity, chartData, chartWindow, onChartWindow, wins, losses, totalMargin, activeBlackouts, instrumentFilter, outcomeFilter, onInstrumentFilter, onOutcomeFilter, onApplyTradeFilters, tradesPage, onTradesPage, signalsPage, onSignalsPage, ordersPage, onOrdersPage }: { data: DashboardData; latestEquity: EquityPoint | undefined; chartData: Array<EquityPoint & { label: string }>; chartWindow: "All" | "24H" | "7D"; onChartWindow: (window: "All" | "24H" | "7D") => void; wins: number; losses: number; totalMargin: number; activeBlackouts: string[]; instrumentFilter: string; outcomeFilter: string; onInstrumentFilter: (value: string) => void; onOutcomeFilter: (value: string) => void; onApplyTradeFilters: () => void; tradesPage: number; onTradesPage: (page: number) => void; signalsPage: number; onSignalsPage: (page: number) => void; ordersPage: number; onOrdersPage: (page: number) => void }) {
  const broker = data.config.broker;
  const hasRiskAlert = Boolean(data.monitoring?.risk_halted || data.monitoring?.db_lock_alert || Object.keys(data.monitoring?.stale_prices ?? {}).length);
  return <>
    <section className="pageHeading overviewHeading"><div><span className="sectionKicker">Live portfolio overview</span><h2>Make decisions with the whole picture.</h2><p>Performance, position risk, operational health, and upcoming event exposure—synchronized from the same service.</p></div><div className="equityChip"><WalletCards size={18} /><span><small>Current equity</small><strong>{money(latestEquity?.equity ?? 0)}</strong></span></div></section>
    <section className="metricGrid" aria-label="Performance summary"><MetricCard label="Net P&L" value={money(data.performance.total_pnl)} sublabel={`${data.performance.trade_count} total trades`} icon={TrendingUp} tone={data.performance.total_pnl >= 0 ? "positive" : "negative"} /><MetricCard label="Win rate" value={percent(data.performance.win_rate)} sublabel={`${wins} wins · ${losses} losses`} icon={Activity} /><MetricCard label="Profit factor" value={data.performance.profit_factor === null ? "—" : data.performance.profit_factor.toFixed(2)} sublabel="Gross profit / gross loss" icon={ChartNoAxesCombined} tone={Number(data.performance.profit_factor ?? 0) >= 1 ? "positive" : "warning"} /><MetricCard label="Max drawdown" value={percent(data.performance.max_drawdown.pct)} sublabel={money(data.performance.max_drawdown.amount)} icon={ArrowDownRight} tone="warning" /><MetricCard label="Open exposure" value={money(totalMargin)} sublabel={`${data.positions.length} open position${data.positions.length === 1 ? "" : "s"}`} icon={ShieldAlert} tone={hasRiskAlert ? "negative" : "neutral"} /></section>
    <section className="overviewGrid"><article className="panel equityPanel"><div className="panelHeader"><div><span className="sectionKicker">Portfolio</span><h3>Equity curve</h3></div><div className="segmentedControl" aria-label="Equity chart window">{(["All", "24H", "7D"] as const).map((window) => <button key={window} className={chartWindow === window ? "active" : ""} onClick={() => onChartWindow(window)}>{window}</button>)}</div></div><div className="chartWrap">{chartData.length === 0 && <EmptyChart />}<ResponsiveContainer width="100%" height={310}><AreaChart data={chartData} margin={{ top: 18, right: 8, bottom: 0, left: -16 }}><defs><linearGradient id="equityFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor="#63e6be" stopOpacity={0.38} /><stop offset="100%" stopColor="#63e6be" stopOpacity={0} /></linearGradient></defs><CartesianGrid vertical={false} stroke="rgba(148, 163, 184, 0.12)" /><XAxis dataKey="label" axisLine={false} tickLine={false} minTickGap={34} tick={{ fill: "#7d8da5", fontSize: 11 }} /><YAxis axisLine={false} tickLine={false} width={70} tickFormatter={(value) => compactMoney(Number(value))} tick={{ fill: "#7d8da5", fontSize: 11 }} /><Tooltip content={<EquityTooltip />} cursor={{ stroke: "rgba(99,230,190,.45)", strokeWidth: 1 }} /><Area type="monotone" dataKey="equity" stroke="#63e6be" strokeWidth={2.4} fill="url(#equityFill)" /></AreaChart></ResponsiveContainer></div><div className="chartFooter"><span><i className="legendDot teal" /> Equity</span><span>Balance {money(latestEquity?.balance ?? 0)}</span><span>Margin in use {money(latestEquity?.margin_used ?? 0)}</span></div></article>
      <article className="panel systemPanel"><div className="panelHeader"><div><span className="sectionKicker">System health</span><h3>Ready to trade?</h3></div><StatusBadge status={data.monitoring?.broker_connected ? "connected" : "checking"} /></div><div className="healthScore"><div className={`healthRing ${hasRiskAlert ? "attention" : ""}`}><span>{hasRiskAlert ? "!" : "✓"}</span></div><div><strong>{hasRiskAlert ? "Attention needed" : "Operational"}</strong><p>{data.monitoring?.broker_connected ? "Broker connection confirmed" : "Broker status has not been confirmed"}</p></div></div><div className="healthList"><HealthLine label="Broker connection" good={Boolean(data.monitoring?.broker_connected)} detail={broker?.server || "MT5 server"} /><HealthLine label="Price feeds" good={Object.keys(data.monitoring?.stale_prices ?? {}).length === 0} detail={Object.keys(data.monitoring?.stale_prices ?? {}).length ? `${Object.keys(data.monitoring?.stale_prices ?? {}).join(", ")} stale` : "No stale prices"} /><HealthLine label="Risk guard" good={!data.monitoring?.risk_halted} detail={data.monitoring?.risk_halt_reason || "No active halt"} /><HealthLine label="News feed" good={data.monitoring?.news_data_fresh !== false} detail={data.monitoring?.news_data_fresh === false ? "Data needs refresh" : "Calendar current"} /></div><div className="systemFooter"><span><Settings2 size={14} /> {broker?.configured ? "Broker configured" : "Broker needs setup"}</span><span>{broker?.demo_only ? "Demo account" : "Live account"}</span></div></article></section>
    <section className="lowerOverviewGrid"><article className="panel positionsPanel"><div className="panelHeader"><div><span className="sectionKicker">Live exposure</span><h3>Open positions</h3></div><span className="countBadge">{data.positions.length}</span></div>{data.positions.length ? <div className="positionList">{data.positions.map((position) => { const share = totalMargin ? (position.margin_used / totalMargin) * 100 : 0; return <div className="positionRow" key={`${position.instrument}-${position.side}`}><div className="positionTop"><div><strong>{position.instrument}</strong><SideBadge side={position.side} /></div><strong className={position.unrealized_pl >= 0 ? "positiveText" : "negativeText"}>{money(position.unrealized_pl)}</strong></div><div className="exposureTrack"><span className={position.side.toLowerCase() === "short" ? "short" : "long"} style={{ width: `${Math.max(4, Math.min(share, 100))}%` }} /></div><div className="positionBottom"><span>{number(position.units)} units · {price(position.avg_price)}</span><span>{money(position.margin_used)} margin</span></div></div>; })}</div> : <EmptyState icon={WalletCards} title="No open positions" text="New positions from the MT5 journal will appear here." />}</article>
      <article className="panel eventPanel"><div className="panelHeader"><div><span className="sectionKicker">Market calendar</span><h3>Event risk</h3></div><span className={activeBlackouts.length ? "riskBadge active" : "riskBadge"}>{activeBlackouts.length ? `${activeBlackouts.length} blackout${activeBlackouts.length > 1 ? "s" : ""}` : "Clear"}</span></div>{activeBlackouts.length > 0 && <div className="blackoutNotice"><ShieldAlert size={16} /><span>Trading restricted for {activeBlackouts.join(", ")}.</span></div>}{data.news?.events.length ? <div className="eventList">{data.news.events.map((event) => <div className="eventItem" key={event.event_id}><span className={`impactDot ${impactTone(event.impact)}`} /><div><strong>{event.name}</strong><small>{event.currency} · {event.impact} impact</small></div><time>{formatShortDate(event.scheduled_at)}</time></div>)}</div> : <EmptyState icon={Newspaper} title="No calendar data" text="Refresh to check the economic calendar." />}</article>
      <article className="panel outcomePanel"><div className="panelHeader"><div><span className="sectionKicker">Closed outcomes</span><h3>Win / loss mix</h3></div></div><div className="outcomeContent"><ResponsiveContainer width={162} height={162}><PieChart><Pie data={[{ name: "Wins", value: wins }, { name: "Losses", value: losses || 0.0001 }]} dataKey="value" innerRadius={52} outerRadius={70} startAngle={90} endAngle={-270} stroke="none" paddingAngle={5}>{["#63e6be", "#ff7b8f"].map((color) => <Cell key={color} fill={color} />)}</Pie></PieChart></ResponsiveContainer><div className="outcomeLegend"><div><i className="legendDot teal" /><span>Wins</span><strong>{wins}</strong></div><div><i className="legendDot coral" /><span>Losses</span><strong>{losses}</strong></div><p>{wins + losses ? `${((wins / (wins + losses)) * 100).toFixed(0)}% of closed trades profitable` : "Awaiting closed trades"}</p></div></div></article></section>
    <section className="panel tablePanel"><div className="tableTitlebar"><div><span className="sectionKicker">Journal</span><h3>Trade history</h3></div><div className="tableFilters"><label className="searchField"><Search size={15} /><input value={instrumentFilter} onChange={(event) => onInstrumentFilter(event.target.value)} placeholder="Filter pair" aria-label="Filter trades by pair" /></label><select value={outcomeFilter} onChange={(event) => onOutcomeFilter(event.target.value)} aria-label="Filter trades by outcome"><option value="">All outcomes</option><option value="win">Wins</option><option value="loss">Losses</option></select><button className="filterButton" onClick={onApplyTradeFilters}><RefreshCw size={15} /> Apply</button></div></div><DataTable columns={["Trade", "Pair", "Side", "Entry", "Exit", "P&L", "State"]} rows={data.trades} page={tradesPage} onPageChange={onTradesPage} empty="No trades are in the journal yet." renderRow={(trade) => { const pnl = trade.realized_pl + trade.financing; return <tr key={trade.broker_trade_id}><td className="mono">{trade.broker_trade_id}</td><td><strong>{trade.instrument}</strong></td><td><SideBadge side={trade.side} /></td><td><span>{price(trade.entry_price)}</span><small>{formatDate(trade.entry_time)}</small></td><td><span>{price(trade.exit_price)}</span><small>{formatDate(trade.exit_time)}</small></td><td className={pnl >= 0 ? "positiveText" : "negativeText"}>{money(pnl)}</td><td><StatusBadge status={trade.state} /></td></tr>; }} /></section>
    <section className="activityGrid"><article className="panel tablePanel compactTable"><div className="panelHeader"><div><span className="sectionKicker">Decision log</span><h3>Recent signals</h3></div><span className="countBadge">{data.signals.length}</span></div><DataTable columns={["Time", "Pair", "Decision", "Score"]} rows={data.signals} page={signalsPage} onPageChange={onSignalsPage} empty="No signals have been recorded." renderRow={(signal) => <tr key={signal.id}><td>{formatDate(signal.timestamp)}</td><td><strong>{signal.instrument}</strong></td><td><StatusBadge status={signal.status} /></td><td>{Number(signal.score).toFixed(2)}</td></tr>} /></article><article className="panel tablePanel compactTable"><div className="panelHeader"><div><span className="sectionKicker">Execution log</span><h3>Recent orders</h3></div><span className="countBadge">{data.orders.length}</span></div><DataTable columns={["Time", "Pair", "Side", "Status", "Risk"]} rows={data.orders} page={ordersPage} onPageChange={onOrdersPage} empty="No orders have been recorded." renderRow={(order) => <tr key={order.client_order_id}><td>{formatDate(order.timestamp)}</td><td><strong>{order.instrument}</strong></td><td><SideBadge side={order.side} /></td><td><StatusBadge status={order.status} /></td><td>{money(order.risk_amount)}</td></tr>} /></article></section>
  </>;
}

function MetricCard({ label, value, sublabel, icon: Icon, tone = "neutral" }: { label: string; value: string; sublabel: string; icon: ElementType; tone?: "positive" | "negative" | "warning" | "neutral" }) { return <article className={`metricCard ${tone}`}><div className="metricIcon"><Icon size={18} /></div><span>{label}</span><strong>{value}</strong><small>{sublabel}</small></article>; }
function ConnectionBadge({ connection }: { connection: ConnectionState }) { const content = connection === "live" ? ["live", "Live stream"] : connection === "polling" ? ["polling", "Polling backup"] : connection === "connecting" ? ["connecting", "Connecting"] : ["offline", "Disconnected"]; return <span className={`connectionBadge ${content[0]}`}><i /> {content[1]}</span>; }
function StatusBadge({ status }: { status: string }) { return <span className={`statusBadge ${toneForStatus(status)}`}>{sentenceCase(status || "unknown")}</span>; }
function SideBadge({ side }: { side: string }) { return <span className={`sideBadge ${side.toLowerCase() === "short" || side.toLowerCase() === "sell" ? "short" : "long"}`}>{sentenceCase(side)}</span>; }
function HealthLine({ label, good, detail }: { label: string; good: boolean; detail: string }) { return <div className="healthLine"><span className={good ? "healthIcon good" : "healthIcon bad"}>{good ? "✓" : "!"}</span><div><strong>{label}</strong><small>{detail}</small></div></div>; }
function EmptyChart() { return <div className="emptyChart"><ChartNoAxesCombined size={26} /><strong>Equity history will appear here</strong><span>The chart fills as the journal receives account snapshots.</span></div>; }
function EmptyState({ icon: Icon, title, text }: { icon: ElementType; title: string; text: string }) { return <div className="emptyState"><Icon size={24} /><strong>{title}</strong><span>{text}</span></div>; }
function EquityTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: EquityPoint }> }) { if (!active || !payload?.length) return null; const point = payload[0].payload; return <div className="chartTooltip"><span>{formatDate(point.timestamp)}</span><strong>{money(point.equity)}</strong><small>Balance {money(point.balance)}</small></div>; }
function DataTable<T>({ columns, rows, page, onPageChange, empty, renderRow }: { columns: string[]; rows: T[]; page: number; onPageChange: (page: number) => void; empty: string; renderRow: (item: T) => ReactNode }) { const totalPages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE)); const safePage = Math.min(page, totalPages - 1); const visibleRows = rows.slice(safePage * PAGE_SIZE, (safePage + 1) * PAGE_SIZE); return <><div className="tableScroll"><table><thead><tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr></thead><tbody>{visibleRows.length ? visibleRows.map(renderRow) : <tr><td className="tableEmpty" colSpan={columns.length}>{empty}</td></tr>}</tbody></table></div>{rows.length > PAGE_SIZE && <Pagination page={safePage} totalPages={totalPages} totalRows={rows.length} onChange={onPageChange} />}</>; }
function Pagination({ page, totalPages, totalRows, onChange }: { page: number; totalPages: number; totalRows: number; onChange: (page: number) => void }) { const start = page * PAGE_SIZE + 1; const end = Math.min((page + 1) * PAGE_SIZE, totalRows); return <div className="pagination"><span>{start}–{end} of {totalRows}</span><div><button onClick={() => onChange(page - 1)} disabled={page === 0} aria-label="Previous page"><ChevronLeft size={16} /></button><span>Page {page + 1} / {totalPages}</span><button onClick={() => onChange(page + 1)} disabled={page === totalPages - 1} aria-label="Next page"><ChevronRight size={16} /></button></div></div>; }
function ConfirmationDialog({ action, running, onCancel, onConfirm }: { action: ControlAction; running: boolean; onCancel: () => void; onConfirm: () => void }) { const copy = action === "start" ? ["Start execution?", "The worker will begin evaluating and submitting eligible demo orders.", "Start bot"] : action === "pause" ? ["Pause execution?", "The worker will stop opening new positions until it is started again.", "Pause bot"] : ["Stop execution?", "The worker will stop. Existing broker positions are not closed by this command.", "Stop bot"]; return <div className="modalBackdrop" role="presentation"><section className="confirmDialog" role="dialog" aria-modal="true" aria-labelledby="confirm-title"><div className={`dialogIcon ${action}`}><ShieldAlert size={22} /></div><h2 id="confirm-title">{copy[0]}</h2><p>{copy[1]}</p><div className="dialogActions"><button onClick={onCancel} disabled={running}>Cancel</button><button className={action === "stop" ? "destructive" : "primary"} onClick={onConfirm} disabled={running}>{running ? <Loader2 size={16} className="spin" /> : null}{running ? "Sending…" : copy[2]}</button></div></section></div>; }
function filterChartData(equity: EquityPoint[], window: "All" | "24H" | "7D"): Array<EquityPoint & { label: string }> { const now = Date.now(); const cutoff = window === "24H" ? now - 86_400_000 : window === "7D" ? now - 604_800_000 : 0; return equity.filter((point) => new Date(point.timestamp).getTime() >= cutoff).map((point) => ({ ...point, label: window === "24H" ? new Date(point.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : new Date(point.timestamp).toLocaleDateString([], { month: "short", day: "numeric" }) })); }
function toneForStatus(status: string): "positive" | "negative" | "warning" | "neutral" { const value = status.toLowerCase(); if (["running", "connected", "filled", "open", "approved", "healthy"].some((word) => value.includes(word))) return "positive"; if (["error", "failed", "rejected", "stopped", "halt", "loss"].some((word) => value.includes(word))) return "negative"; if (["pause", "pending", "checking", "warning"].some((word) => value.includes(word))) return "warning"; return "neutral"; }
function impactTone(impact: string): string { const value = impact.toLowerCase(); return value.includes("high") ? "high" : value.includes("medium") ? "medium" : "low"; }
function sentenceCase(value: string): string { return value ? value.replace(/[_-]/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase()) : "Unknown"; }
function money(value: number): string { return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(value || 0); }
function compactMoney(value: number): string { return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", notation: "compact", maximumFractionDigits: 1 }).format(value || 0); }
function number(value: number): string { return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(value || 0); }
function percent(value: number): string { return `${((value || 0) * 100).toFixed(2)}%`; }
function price(value: number | null): string { return value === null || value === undefined || Number.isNaN(value) ? "—" : Number(value).toFixed(value > 20 ? 3 : 5); }
function formatDate(value: string | null | undefined): string { if (!value || Number.isNaN(new Date(value).getTime())) return "—"; return new Date(value).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }); }
function formatShortDate(value: string): string { if (Number.isNaN(new Date(value).getTime())) return "—"; return new Date(value).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }); }
function timeAgo(value: Date): string { const seconds = Math.max(0, Math.round((Date.now() - value.getTime()) / 1000)); return seconds < 8 ? "just now" : seconds < 60 ? `${seconds}s ago` : seconds < 3600 ? `${Math.floor(seconds / 60)}m ago` : `${Math.floor(seconds / 3600)}h ago`; }
