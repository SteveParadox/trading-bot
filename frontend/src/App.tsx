import {
  ArrowDownRight,
  ArrowUpRight,
  ChevronLeft,
  ChevronRight,
  CirclePause,
  CircleStop,
  KeyRound,
  Play,
  RefreshCw,
  Radar,
  ShieldCheck,
  ShoppingCart
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import {
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  PieChart,
  Pie,
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

type Signal = {
  id: number;
  timestamp: string;
  instrument: string;
  side: string | null;
  status: string;
  reason: string;
  score: number;
  entry_price: number | null;
  stop_loss: number | null;
  take_profit: number | null;
  risk_amount: number | null;
};

type Order = {
  id: number;
  client_order_id: string;
  timestamp: string;
  instrument: string;
  side: string;
  units: number;
  order_type: string;
  status: string;
  broker_order_id: string | null;
  broker_trade_id: string | null;
  error: string | null;
  risk_amount: number;
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
  gross_profit?: number;
  gross_loss?: number;
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
  recent_signals: Signal[];
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

const ROWS_PER_PAGE = 15;

export function App() {
  const [status, setStatus] = useState<BotStatus>({ state: "loading", reason: "", updated_at: "" });
  const [positions, setPositions] = useState<Position[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [equity, setEquity] = useState<EquityPoint[]>([]);
  const [performance, setPerformance] = useState<Performance>(emptyPerformance);
  const [config, setConfig] = useState<ConfigPayload>({});
  const [signals, setSignals] = useState<Signal[]>([]);
  const [orders, setOrders] = useState<Order[]>([]);
  const [refreshingTrades, setRefreshingTrades] = useState(false);
  const [apiKey, setApiKey] = useState(localStorage.getItem("fx_api_key") ?? "");
  const [instrumentFilter, setInstrumentFilter] = useState("");
  const [outcomeFilter, setOutcomeFilter] = useState("");
  const [connection, setConnection] = useState<"connecting" | "live" | "polling">("connecting");
  const [error, setError] = useState("");
  const [tradesPage, setTradesPage] = useState(0);
  const [signalsPage, setSignalsPage] = useState(0);
  const [ordersPage, setOrdersPage] = useState(0);
  const [chartTimeframe, setChartTimeframe] = useState<"1H" | "1D" | "1W">("1D");

  const filtersRef = useRef({ instrument: "", outcome: "" });

  const applySnapshot = useCallback((snapshot: LiveSnapshot) => {
    setStatus(snapshot.status);
    setPositions(snapshot.positions);
    setEquity(snapshot.equity_curve);
    setPerformance(snapshot.performance);
    if (snapshot.recent_signals) {
      setSignals(snapshot.recent_signals);
    }
    if (!filtersRef.current.instrument.trim() && !filtersRef.current.outcome) {
      setTrades(snapshot.recent_trades);
    }
    setConfig(snapshot.config);
    setConnection("live");
  }, []);

  const load = useCallback(async () => {
    const [statusResponse, positionResponse, tradeResponse, equityResponse, performanceResponse, configResponse, signalsResponse, ordersResponse] =
      await Promise.all([
        fetch(`${API_BASE}/api/status`),
        fetch(`${API_BASE}/api/positions`),
        fetch(`${API_BASE}/api/trades?limit=300`),
        fetch(`${API_BASE}/api/equity?limit=300`),
        fetch(`${API_BASE}/api/performance`),
        fetch(`${API_BASE}/api/config`),
        fetch(`${API_BASE}/api/signals?limit=100`),
        fetch(`${API_BASE}/api/orders?limit=100`)
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
    setSignals(await signalsResponse.json());
    setOrders(await ordersResponse.json());
  }, []);

  useEffect(() => {
    load().catch((issue: unknown) => setError(issue instanceof Error ? issue.message : String(issue)));
  }, [load]);

  useEffect(() => {
    filtersRef.current = { instrument: instrumentFilter, outcome: outcomeFilter };
  }, [instrumentFilter, outcomeFilter]);

  // Primary live channel: WebSocket with automatic reconnection.
  useEffect(() => {
    let socket: WebSocket | null = null;
    let reconnectTimer: number | undefined;
    let disposed = false;

    const connect = () => {
      if (disposed) {
        return;
      }
      const wsUrl = API_BASE.replace(/^http/, "ws") + "/ws/live";
      socket = new WebSocket(wsUrl);
      socket.onopen = () => {
        setConnection("live");
        if (reconnectTimer !== undefined) {
          window.clearTimeout(reconnectTimer);
          reconnectTimer = undefined;
        }
      };
      socket.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          if (payload.status && payload.performance) {
            applySnapshot(payload);
          }
        } catch {
          // ignore malformed message
        }
      };
      socket.onerror = () => {
        socket?.close();
      };
      socket.onclose = () => {
        setConnection("polling");
        socket = null;
        if (!disposed) {
          reconnectTimer = window.setTimeout(connect, 3000);
        }
      };
    };

    connect();
    return () => {
      disposed = true;
      if (reconnectTimer !== undefined) {
        window.clearTimeout(reconnectTimer);
      }
      socket?.close();
    };
  }, [applySnapshot]);

  // Polling fallback: active whenever the WebSocket is not connected (live).
  useEffect(() => {
    if (connection === "live") {
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
    setRefreshingTrades(true);
    try {
      const response = await fetch(`${API_BASE}/api/trades?${params}`, { cache: "no-store" });
      if (!response.ok) {
        throw new Error("Trade history request failed");
      }
      const latestTrades = (await response.json()) as Trade[];
      setTrades(latestTrades);
      setTradesPage(0);
      setError("");
    } finally {
      setRefreshingTrades(false);
    }
  }, [instrumentFilter, outcomeFilter]);

  useEffect(() => {
    refreshTrades().catch((issue: unknown) => {
      setError(issue instanceof Error ? issue.message : String(issue));
    });
  }, [refreshTrades]);

  useEffect(() => {
    setTradesPage(0);
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

  const chartData = useMemo(() => {
    if (chartTimeframe === "1H") {
      return equity.map((point) => ({
        ...point,
        label: new Date(point.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
      }));
    }

    const bucketKey = (ts: Date): string => {
      if (chartTimeframe === "1D") {
        return `${ts.getFullYear()}-${String(ts.getMonth() + 1).padStart(2, "0")}-${String(ts.getDate()).padStart(2, "0")}`;
      }
      const d = new Date(ts);
      d.setDate(d.getDate() - d.getDay());
      return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
    };

    const buckets = new Map<string, EquityPoint>();
    for (const point of equity) {
      const key = bucketKey(new Date(point.timestamp));
      buckets.set(key, point);
    }

    return Array.from(buckets.values()).map((point) => {
      const d = new Date(point.timestamp);
      if (chartTimeframe === "1D") {
        return { ...point, label: d.toLocaleDateString([], { month: "short", day: "numeric" }) };
      }
      return { ...point, label: `W/o ${d.toLocaleDateString([], { month: "short", day: "numeric" })}` };
    });
  }, [equity, chartTimeframe]);

  const winCount = useMemo(() => {
    return trades.filter((t) => t.state === "closed" && (t.realized_pl + t.financing) > 0).length;
  }, [trades]);
  const lossCount = useMemo(() => {
    return trades.filter((t) => t.state === "closed" && (t.realized_pl + t.financing) < 0).length;
  }, [trades]);

  const brokerName = "MetaTrader 5";
  const brokerMode = config.broker?.demo_only ? " demo" : " live";
  const brokerServer = config.broker?.server ? ` / ${String(config.broker.server)}` : "";
  const brokerLabel = `${brokerName}${brokerMode}${brokerServer}`;
  const brokerConfigured = config.broker?.configured ? "Configured" : "Needs setup";
  const terminalConfigured = config.broker?.terminal_path_configured ? "Configured" : "Not configured";

  return (
    <main>
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
        <Stat label="Total P&L" value={money(performance.total_pnl)} tone={performance.total_pnl >= 0 ? "good" : "bad"} trend={performance.total_pnl >= 0 ? "up" : "down"} />
        <Stat label="Win Rate" value={percent(performance.win_rate)} />
        <Stat label="Profit Factor" value={performance.profit_factor === null ? "n/a" : performance.profit_factor.toFixed(2)} />
        <Stat label="Max Drawdown" value={percent(performance.max_drawdown.pct)} tone="warn" />
        <Stat label="Sharpe" value={performance.sharpe_ratio.toFixed(2)} />
        <Stat label="Sortino" value={performance.sortino_ratio.toFixed(2)} />
        <Stat label="Trades" value={String(performance.trade_count)} />
        <Stat label="Financing" value={money(performance.financing)} />
      </section>

      <section className="chartRow">
        <div className="panel equityPanel">
          <div className="panelHeader">
            <h2>Equity Curve</h2>
            <div className="chartTimeframeGroup">
              {(["1H", "1D", "1W"] as const).map((tf) => (
                <button
                  key={tf}
                  className={`chartTimeframeBtn ${chartTimeframe === tf ? "active" : ""}`}
                  onClick={() => setChartTimeframe(tf)}
                >
                  {tf}
                </button>
              ))}
            </div>
          </div>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={chartData} margin={{ left: 4, right: 16, top: 12, bottom: 6 }}>
              <defs>
                <linearGradient id="equityGradient" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#06d6a0" />
                  <stop offset="100%" stopColor="#4cc9f0" />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="rgba(148,163,184,0.08)" vertical={false} />
              <XAxis dataKey="label" tickLine={false} axisLine={false} minTickGap={32} tick={{ fill: "#7b8fa3", fontSize: 10 }} />
              <YAxis tickLine={false} axisLine={false} width={72} tick={{ fill: "#7b8fa3", fontSize: 10 }} tickFormatter={(value) => `$${Number(value).toFixed(0)}`} />
              <Tooltip
                formatter={(value) => money(Number(value))}
                labelFormatter={(_, rows) => formatDate(rows?.[0]?.payload?.timestamp)}
                contentStyle={{
                  border: "1px solid rgba(255,255,255,0.12)",
                  borderRadius: 10,
                  background: "rgba(12,20,35,0.92)",
                  backdropFilter: "blur(16px)",
                  color: "#f0f6ff",
                  fontSize: 12,
                  boxShadow: "0 12px 40px rgba(0,0,0,0.4)"
                }}
              />
              <Line type="monotone" dataKey="equity" stroke="url(#equityGradient)" strokeWidth={2.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="donutPanel">
          <div className="panelHeader">
            <h2>Win / Loss</h2>
          </div>
          <div className="donutChart">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={[
                    { name: "Wins", value: winCount },
                    { name: "Losses", value: lossCount }
                  ]}
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={80}
                  paddingAngle={4}
                  dataKey="value"
                  strokeWidth={0}
                >
                  <Cell fill="#06d6a0" />
                  <Cell fill="#ef476f" />
                </Pie>
                <Tooltip
                  formatter={(value: number, name: string) => [String(value), name]}
                  contentStyle={{
                    border: "1px solid rgba(255,255,255,0.12)",
                    borderRadius: 10,
                    background: "rgba(12,20,35,0.92)",
                    backdropFilter: "blur(16px)",
                    color: "#f0f6ff",
                    fontSize: 12,
                    boxShadow: "0 12px 40px rgba(0,0,0,0.4)"
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
            <div className="donutLegend">
              <span className="donutLegendItem"><span className="dotGreen" />{winCount} wins</span>
              <span className="donutLegendItem"><span className="dotRed" />{lossCount} losses</span>
            </div>
          </div>
        </div>
      </section>

      <section className="mainGrid">
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

        <div className="panel exposurePanel">
          <div className="panelHeader">
            <h2>Position Exposure</h2>
          </div>
          {positions.length === 0 ? (
            <div className="emptyExposure">No open positions</div>
          ) : (
            <div className="exposureList">
              {positions.map((position) => {
                const totalMargin = positions.reduce((sum, p) => sum + p.margin_used, 0);
                const pct = totalMargin > 0 ? (position.margin_used / totalMargin) * 100 : 0;
                return (
                  <div key={position.instrument} className="exposureRow">
                    <div className="exposureInfo">
                      <span className="exposurePair">{position.instrument}</span>
                      <span className={`pill ${position.side.toLowerCase()}`}>{position.side}</span>
                    </div>
                    <div className="exposureBarTrack">
                      <div
                        className={`exposureBarFill ${position.side.toLowerCase()}`}
                        style={{ width: `${Math.min(pct, 100)}%` }}
                      />
                    </div>
                    <div className="exposureMeta">
                      <span>{money(position.margin_used)}</span>
                      <span className={position.unrealized_pl >= 0 ? "goodText" : "badText"}>
                        {money(position.unrealized_pl)}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
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
            <button title="Refresh history" onClick={refreshTrades} disabled={refreshingTrades}>
              <RefreshCw size={17} />
              {refreshingTrades ? "Refreshing" : "Refresh"}
            </button>
          </div>
        </div>
        <PaginatedTable
          columns={["Trade", "Pair", "Side", "Units", "Entry", "Exit", "P&L", "Financing", "State"]}
          empty="No trades journaled yet"
          rows={trades}
          page={tradesPage}
          onPageChange={setTradesPage}
          renderRow={(trade) => {
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
          }}
        />
      </section>

      <section className="dualPanels">
        <div className="panel">
          <div className="panelHeader">
            <h2><Radar size={16} /> Signals</h2>
            <span className="panelBadge">{signals.length}</span>
          </div>
          <PaginatedTable
            columns={["Time", "Pair", "Side", "Status", "Reason", "Score", "Entry", "SL", "TP"]}
            empty="No signals recorded yet"
            rows={signals}
            page={signalsPage}
            onPageChange={setSignalsPage}
            renderRow={(signal) => (
              <tr key={signal.id}>
                <td>{formatDate(signal.timestamp)}</td>
                <td>{signal.instrument}</td>
                <td>{signal.side ? <span className={`pill ${signal.side.toLowerCase()}`}>{signal.side}</span> : "—"}</td>
                <td><span className={`statusPill ${signal.status}`}>{signal.status}</span></td>
                <td>{signal.reason}</td>
                <td>{signal.score.toFixed(2)}</td>
                <td>{signal.entry_price ? price(signal.entry_price) : "—"}</td>
                <td>{signal.stop_loss ? price(signal.stop_loss) : "—"}</td>
                <td>{signal.take_profit ? price(signal.take_profit) : "—"}</td>
              </tr>
            )}
          />
        </div>

        <div className="panel">
          <div className="panelHeader">
            <h2><ShoppingCart size={16} /> Orders</h2>
            <span className="panelBadge">{orders.length}</span>
          </div>
          <PaginatedTable
            columns={["Time", "Pair", "Side", "Units", "Type", "Status", "Risk", "Error"]}
            empty="No orders journaled yet"
            rows={orders}
            page={ordersPage}
            onPageChange={setOrdersPage}
            renderRow={(order) => (
              <tr key={order.client_order_id}>
                <td>{formatDate(order.timestamp)}</td>
                <td>{order.instrument}</td>
                <td><span className={`pill ${order.side.toLowerCase()}`}>{order.side}</span></td>
                <td>{number(order.units)}</td>
                <td>{order.order_type}</td>
                <td><span className={`statusPill ${order.status}`}>{order.status}</span></td>
                <td>{money(order.risk_amount)}</td>
                <td className={order.error ? "badText" : ""}>{order.error || "—"}</td>
              </tr>
            )}
          />
        </div>
      </section>
    </main>
  );
}

function Stat({ label, value, tone = "", trend }: { label: string; value: string; tone?: string; trend?: "up" | "down" }) {
  return (
    <div className={`stat ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      {trend && (
        <span className={`trendBadge ${trend === "up" ? "trendUp" : "trendDown"}`}>
          {trend === "up" ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}
        </span>
      )}
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

function PaginatedTable<T>({
  columns,
  empty,
  rows,
  page,
  onPageChange,
  renderRow
}: {
  columns: string[];
  empty: string;
  rows: T[];
  page: number;
  onPageChange: (page: number) => void;
  renderRow: (item: T) => ReactNode;
}) {
  const totalPages = Math.max(1, Math.ceil(rows.length / ROWS_PER_PAGE));
  const safePage = Math.min(page, totalPages - 1);
  const start = safePage * ROWS_PER_PAGE;
  const visibleRows = rows.slice(start, start + ROWS_PER_PAGE);
  const rangeStart = rows.length > 0 ? start + 1 : 0;
  const rangeEnd = Math.min(start + ROWS_PER_PAGE, rows.length);

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
    <>
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
            {visibleRows.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="empty">{empty}</td>
              </tr>
            ) : (
              visibleRows.map((item) => renderRow(item))
            )}
          </tbody>
        </table>
      </div>
      {rows.length > ROWS_PER_PAGE && (
        <div className="pagination">
          <span className="paginationInfo">
            {rangeStart}–{rangeEnd} of {rows.length}
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
      )}
    </>
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
