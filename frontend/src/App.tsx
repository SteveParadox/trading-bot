import {
  Activity,
  CirclePause,
  CircleStop,
  KeyRound,
  Play,
  RefreshCw,
  ShieldCheck
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

type BotStatus = {
  state: string;
  reason: string;
  updated_at: string;
  worker_task_running?: boolean;
  demo_only?: boolean;
};

type EquityPoint = {
  timestamp: string;
  equity: number;
  balance: number;
  margin_used: number;
  open_positions: number;
};

type Position = {
  instrument: string;
  side: string;
  units: number;
  avg_price: number;
  unrealized_pl: number;
  margin_used: number;
  price: number | null;
  estimated_daily_financing: number;
  updated_at: string;
};

type Trade = {
  broker_trade_id: string;
  instrument: string;
  side: string;
  units: number;
  entry_time: string | null;
  entry_price: number | null;
  exit_time: string | null;
  exit_price: number | null;
  realized_pl: number;
  financing: number;
  state: string;
  exit_reason: string | null;
};

type Performance = {
  trade_count: number;
  open_trade_count: number;
  win_rate: number;
  profit_factor: number | null;
  total_pnl: number;
  max_drawdown: { amount: number; pct: number };
  sharpe_ratio: number;
  sortino_ratio: number;
  average_r_multiple: number;
  financing: number;
};

type ConfigPayload = {
  instruments?: string[];
  dashboard_badge?: string;
  strategy?: Record<string, unknown>;
  risk?: Record<string, unknown>;
  broker?: BrokerConfig;
  runtime?: Record<string, unknown>;
};

type BrokerConfig = {
  provider?: string;
  server?: string;
  configured?: boolean;
  demo_only?: boolean;
  terminal_path_configured?: boolean;
  login_hint?: string;
};

type LiveSnapshot = {
  status: BotStatus;
  positions: Position[];
  equity_curve: EquityPoint[];
  performance: Performance;
  recent_trades: Trade[];
  config: ConfigPayload;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

const emptyPerformance: Performance = {
  trade_count: 0,
  open_trade_count: 0,
  win_rate: 0,
  profit_factor: null,
  total_pnl: 0,
  max_drawdown: { amount: 0, pct: 0 },
  sharpe_ratio: 0,
  sortino_ratio: 0,
  average_r_multiple: 0,
  financing: 0
};

export function App() {
  const [status, setStatus] = useState<BotStatus>({ state: "loading", reason: "", updated_at: "" });
  const [positions, setPositions] = useState<Position[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [equity, setEquity] = useState<EquityPoint[]>([]);
  const [performance, setPerformance] = useState<Performance>(emptyPerformance);
  const [config, setConfig] = useState<ConfigPayload>({});
  const [apiKey, setApiKey] = useState(localStorage.getItem("fx_api_key") ?? "");
  const [instrumentFilter, setInstrumentFilter] = useState("");
  const [outcomeFilter, setOutcomeFilter] = useState("");
  const [connection, setConnection] = useState<"connecting" | "live" | "polling">("connecting");
  const [error, setError] = useState("");

  const applySnapshot = useCallback((snapshot: LiveSnapshot) => {
    setStatus(snapshot.status);
    setPositions(snapshot.positions);
    setEquity(snapshot.equity_curve);
    setPerformance(snapshot.performance);
    setTrades(snapshot.recent_trades);
    setConfig(snapshot.config);
    setConnection("live");
  }, []);

  const load = useCallback(async () => {
    const [statusResponse, positionResponse, tradeResponse, equityResponse, performanceResponse, configResponse] =
      await Promise.all([
        fetch(`${API_BASE}/api/status`),
        fetch(`${API_BASE}/api/positions`),
        fetch(`${API_BASE}/api/trades?limit=200`),
        fetch(`${API_BASE}/api/equity?limit=300`),
        fetch(`${API_BASE}/api/performance`),
        fetch(`${API_BASE}/api/config`)
      ]);
    if (!statusResponse.ok) {
      throw new Error("API status request failed");
    }
    setStatus(await statusResponse.json());
    setPositions(await positionResponse.json());
    setTrades(await tradeResponse.json());
    setEquity(await equityResponse.json());
    setPerformance(await performanceResponse.json());
    setConfig(await configResponse.json());
  }, []);

  useEffect(() => {
    load().catch((issue: unknown) => setError(issue instanceof Error ? issue.message : String(issue)));
  }, [load]);

  useEffect(() => {
    const wsUrl = API_BASE.replace(/^http/, "ws") + "/ws/live";
    const socket = new WebSocket(wsUrl);
    socket.onopen = () => setConnection("live");
    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      if (payload.status && payload.performance) {
        applySnapshot(payload);
      }
    };
    socket.onerror = () => setConnection("polling");
    socket.onclose = () => setConnection("polling");
    return () => socket.close();
  }, [applySnapshot]);

  useEffect(() => {
    if (connection !== "polling") {
      return;
    }
    const id = window.setInterval(() => {
      load().catch(() => undefined);
    }, 5000);
    return () => window.clearInterval(id);
  }, [connection, load]);

  const refreshTrades = useCallback(async () => {
    const params = new URLSearchParams({ limit: "300" });
    if (instrumentFilter.trim()) {
      params.set("instrument", instrumentFilter.trim().toUpperCase());
    }
    if (outcomeFilter) {
      params.set("outcome", outcomeFilter);
    }
    const response = await fetch(`${API_BASE}/api/trades?${params}`);
    setTrades(await response.json());
  }, [instrumentFilter, outcomeFilter]);

  const sendControl = useCallback(
    async (action: "start" | "pause" | "stop") => {
      localStorage.setItem("fx_api_key", apiKey);
      const response = await fetch(`${API_BASE}/api/control/${action}`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-API-Key": apiKey },
        body: JSON.stringify({ reason: `dashboard_${action}` })
      });
      if (!response.ok) {
        setError(action + " failed: check FX_API_KEY");
        return;
      }
      setError("");
      setStatus(await response.json());
    },
    [apiKey]
  );

  const latestEquity = equity.length > 0 ? equity[equity.length - 1] : undefined;
  const chartData = useMemo(
    () =>
      equity.map((point) => ({
        ...point,
        label: new Date(point.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
      })),
    [equity]
  );
  const brokerName = "MetaTrader 5";
  const brokerMode = config.broker?.demo_only ? " demo" : " live";
  const brokerServer = config.broker?.server ? ` / ${String(config.broker.server)}` : "";
  const brokerLabel = `${brokerName}${brokerMode}${brokerServer}`;
  const brokerConfigured = config.broker?.configured ? "Configured" : "Needs setup";
  const terminalConfigured = config.broker?.terminal_path_configured ? "Configured" : "Not configured";

  return (
    <main>
      {/* Ambient glassmorphic background orbs — purely decorative */}
      <div className="ambientLayer" aria-hidden="true">
        <div className="orb orb-1" />
        <div className="orb orb-2" />
        <div className="orb orb-3" />
        <div className="noiseOverlay" />
      </div>

      <header className="topbar">
        <div>
          <div className="eyebrow">
            <ShieldCheck size={16} />
            {config.dashboard_badge ?? "DEMO / FORWARD TEST"}
          </div>
          <h1>MT5 FX Forward Test</h1>
        </div>
        <div className="statusCluster">
          <span className={`statusDot ${status.state}`}></span>
          <span className="statusText">{status.state}</span>
          <span className="connection">{connection}</span>
        </div>
      </header>

      {error && <div className="notice">{error}</div>}

      <section className="controlBand">
        <div className="capitalBlock">
          <span>Equity</span>
          <strong>{money(latestEquity?.equity ?? 0)}</strong>
          <small>Updated {formatDate(status.updated_at)}</small>
        </div>
        <div className="keyBox">
          <KeyRound size={17} />
          <input
            value={apiKey}
            onChange={(event) => setApiKey(event.target.value)}
            type="password"
            placeholder="FX_API_KEY"
          />
        </div>
        <div className="buttonRow">
          <button title="Start bot" onClick={() => sendControl("start")}>
            <Play size={18} />
            Start
          </button>
          <button title="Pause bot" onClick={() => sendControl("pause")}>
            <CirclePause size={18} />
            Pause
          </button>
          <button title="Stop bot" className="danger" onClick={() => sendControl("stop")}>
            <CircleStop size={18} />
            Stop
          </button>
        </div>
      </section>

      <section className="statsGrid">
        <Stat label="Total P&L" value={money(performance.total_pnl)} tone={performance.total_pnl >= 0 ? "good" : "bad"} />
        <Stat label="Win Rate" value={percent(performance.win_rate)} />
        <Stat label="Profit Factor" value={performance.profit_factor === null ? "n/a" : performance.profit_factor.toFixed(2)} />
        <Stat label="Max Drawdown" value={percent(performance.max_drawdown.pct)} tone="warn" />
        <Stat label="Sharpe" value={performance.sharpe_ratio.toFixed(2)} />
        <Stat label="Avg R" value={performance.average_r_multiple.toFixed(2)} />
        <Stat label="Trades" value={String(performance.trade_count)} />
        <Stat label="Financing" value={money(performance.financing)} />
      </section>

      <section className="mainGrid">
        <div className="panel equityPanel">
          <div className="panelHeader">
            <h2>Equity Curve</h2>
            <Activity size={18} />
          </div>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={chartData} margin={{ left: 4, right: 16, top: 12, bottom: 6 }}>
              <CartesianGrid stroke="#e6ebef" vertical={false} />
              <XAxis dataKey="label" tickLine={false} axisLine={false} minTickGap={32} />
              <YAxis tickLine={false} axisLine={false} width={72} tickFormatter={(value) => `$${Number(value).toFixed(0)}`} />
              <Tooltip formatter={(value) => money(Number(value))} labelFormatter={(_, rows) => formatDate(rows?.[0]?.payload?.timestamp)} />
              <Line type="monotone" dataKey="equity" stroke="#0f766e" strokeWidth={2.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="panel configPanel">
          <div className="panelHeader">
            <h2>Strategy Parameters</h2>
          </div>
          <dl>
            {config.instruments && <Metric name="Pairs" value={config.instruments.join(", ")} />}
            <Metric name="Broker" value={brokerLabel} />
            <Metric name="MT5 Terminal" value={terminalConfigured} />
            <Metric name="Account" value={config.broker?.login_hint || brokerConfigured} />
            <Metric name="Entry TF" value={String(config.strategy?.entry_timeframe ?? "")} />
            <Metric name="HTF" value={String(config.strategy?.htf_timeframe ?? "")} />
            <Metric name="ADX" value={`${config.strategy?.adx_min ?? ""} / ${config.strategy?.htf_adx_min ?? ""}`} />
            <Metric name="Risk/Trade" value={percent(Number(config.risk?.risk_per_trade_pct ?? 0))} />
            <Metric name="Daily Halt" value={percent(Number(config.risk?.max_daily_loss_pct ?? 0))} />
            <Metric name="Max Gross" value={percent(Number(config.risk?.max_gross_exposure_pct ?? 0))} />
          </dl>
        </div>
      </section>

      <section className="panel">
        <div className="panelHeader">
          <h2>Open Positions</h2>
        </div>
        <Table
          columns={["Pair", "Side", "Units", "Avg", "Mark", "Live P&L", "Margin", "Est. Rollover"]}
          empty="No open MT5 positions"
        >
          {positions.map((position) => (
            <tr key={position.instrument}>
              <td>{position.instrument}</td>
              <td><span className={`pill ${position.side.toLowerCase()}`}>{position.side}</span></td>
              <td>{number(position.units)}</td>
              <td>{price(position.avg_price)}</td>
              <td>{position.price ? price(position.price) : "n/a"}</td>
              <td className={position.unrealized_pl >= 0 ? "goodText" : "badText"}>{money(position.unrealized_pl)}</td>
              <td>{money(position.margin_used)}</td>
              <td>{money(position.estimated_daily_financing)}</td>
            </tr>
          ))}
        </Table>
      </section>

      <section className="panel">
        <div className="historyHeader">
          <h2>Trade History</h2>
          <div className="filters">
            <input value={instrumentFilter} onChange={(event) => setInstrumentFilter(event.target.value)} placeholder="Pair" />
            <select value={outcomeFilter} onChange={(event) => setOutcomeFilter(event.target.value)}>
              <option value="">All</option>
              <option value="win">Wins</option>
              <option value="loss">Losses</option>
            </select>
            <button title="Refresh history" onClick={refreshTrades}>
              <RefreshCw size={17} />
              Refresh
            </button>
          </div>
        </div>
        <Table
          columns={["Trade", "Pair", "Side", "Units", "Entry", "Exit", "P&L", "Financing", "State"]}
          empty="No trades journaled yet"
        >
          {trades.map((trade) => {
            const pnl = trade.realized_pl + trade.financing;
            return (
              <tr key={trade.broker_trade_id}>
                <td>{trade.broker_trade_id}</td>
                <td>{trade.instrument}</td>
                <td>{trade.side}</td>
                <td>{number(trade.units)}</td>
                <td>{price(trade.entry_price)}<small>{formatDate(trade.entry_time)}</small></td>
                <td>{price(trade.exit_price)}<small>{formatDate(trade.exit_time)}</small></td>
                <td className={pnl >= 0 ? "goodText" : "badText"}>{money(pnl)}</td>
                <td>{money(trade.financing)}</td>
                <td>{trade.state}</td>
              </tr>
            );
          })}
        </Table>
      </section>
    </main>
  );
}

function Stat({ label, value, tone = "" }: { label: string; value: string; tone?: string }) {
  return (
    <div className={`stat ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Metric({ name, value }: { name: string; value: string }) {
  return (
    <>
      <dt>{name}</dt>
      <dd>{value}</dd>
    </>
  );
}

function Table({ columns, empty, children }: { columns: string[]; empty: string; children: ReactNode }) {
  const rows = Array.isArray(children) ? children.filter(Boolean) : children;
  const isEmpty = Array.isArray(rows) ? rows.length === 0 : !rows;
  return (
    <div className="tableWrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {isEmpty ? (
            <tr>
              <td colSpan={columns.length} className="empty">{empty}</td>
            </tr>
          ) : (
            rows
          )}
        </tbody>
      </table>
    </div>
  );
}

function money(value: number) {
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(value || 0);
}

function number(value: number) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(value || 0);
}

function percent(value: number) {
  return `${((value || 0) * 100).toFixed(2)}%`;
}

function price(value: number | null) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/a";
  }
  return Number(value).toFixed(value > 20 ? 3 : 5);
}

function formatDate(value: string | null | undefined) {
  if (!value) {
    return "n/a";
  }
  return new Date(value).toLocaleString([], {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  });
}