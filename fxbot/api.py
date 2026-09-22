"""FastAPI service for the MT5 FX forward-test platform."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from fxbot.analytics import live_snapshot, performance_summary
from fxbot.config import FxBotSettings, ensure_runtime_dirs, settings_from_env
from fxbot.forward import ForwardTestWorker
from fxbot.journal import StructuredJournal, row_to_dict
from fxbot.market_hours import can_trade
from fxbot.models import BotRunState
from fxbot.monitoring import OperationalMonitor
from fxbot.security import SlidingWindowRateLimiter, redact

log = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

_DEFAULT_API_KEYS = {"", "change-this-demo-control-key", "your_api_key_here", "changeme"}


class ControlRequest(BaseModel):
    reason: str = ""


class CritiqueRequest(BaseModel):
    finding: str
    trade_ids: list[str] | None = None
    initial_confidence: float = 0.7


class LiveBroadcaster:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._clients.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(websocket)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            clients = list(self._clients)
        for websocket in clients:
            try:
                await websocket.send_json(payload)
            except RuntimeError:
                await self.disconnect(websocket)


class WorkerController:
    def __init__(
        self,
        *,
        settings: FxBotSettings,
        journal: StructuredJournal,
        broadcaster: LiveBroadcaster,
        monitor: OperationalMonitor | None = None,
    ) -> None:
        self.settings = settings
        self.journal = journal
        self.broadcaster = broadcaster
        self.worker = ForwardTestWorker(
            settings,
            journal=journal,
            publisher=self.broadcaster.broadcast,
            monitor=monitor,
        )
        self.task: asyncio.Task[None] | None = None

    async def start_worker_task(self) -> None:
        if self.task and not self.task.done():
            return
        self.task = asyncio.create_task(self.worker.run_forever())

    async def shutdown(self) -> None:
        self.worker.stop()
        if self.task and not self.task.done():
            await asyncio.wait([self.task], timeout=5.0)
        self.worker.close()

    def set_state(self, state: BotRunState, reason: str) -> dict[str, Any]:
        row = self.journal.set_state(state, reason)
        return row_to_dict(row)


def _validate_startup_security(settings: FxBotSettings) -> None:
    """Fail loudly on obviously insecure defaults so nobody ships them to prod."""
    api_key = settings.runtime.api_key
    if not api_key or api_key in _DEFAULT_API_KEYS:
        log.warning(
            "FX_API_KEY is empty or a known default; all /api/* endpoints "
            "will reject requests. Set a real key in .env before deploying."
        )
    if settings.runtime.bind_host == "0.0.0.0" or settings.runtime.bind_host == "::":
        log.warning(
            "API is binding to %s (all interfaces). "
            "Use FX_API_HOST=127.0.0.1 unless you have a reverse proxy with TLS.",
            settings.runtime.bind_host,
        )


def _app_state_news_gateway(settings: FxBotSettings, monitor: OperationalMonitor) -> "object":
    """Return the shared non-worker NewsGateway for this process.

    Builds it lazily once; subsequent calls reuse the same gateway so the
    throttle/refresh state is shared across dashboard polls instead of each
    request paying for a fresh provider fetch.
    """
    key = settings.runtime.database_url
    from fxbot.news import build_news_gateway

    gateway = _NEWS_GATEWAY_CACHE.get(key)
    if gateway is None:
        gateway = build_news_gateway(
            settings.strategy,
            database_url=settings.runtime.database_url,
            monitor=monitor,
            static_events=settings.news_events,
        )
        _NEWS_GATEWAY_CACHE[key] = gateway
    return gateway


_NEWS_GATEWAY_CACHE: dict[str, object] = {}


def create_app(settings: FxBotSettings | None = None) -> FastAPI:
    resolved_settings = settings or settings_from_env()
    ensure_runtime_dirs(resolved_settings)
    _validate_startup_security(resolved_settings)
    journal = StructuredJournal(resolved_settings.runtime.database_url, resolved_settings.runtime.log_jsonl_path)
    broadcaster = LiveBroadcaster()
    monitor = OperationalMonitor(
        stale_price_threshold_seconds=resolved_settings.runtime.max_price_age_seconds,
        news_data_max_age_seconds=resolved_settings.strategy.news_data_max_age_seconds,
    )
    controller = WorkerController(settings=resolved_settings, journal=journal, broadcaster=broadcaster, monitor=monitor)
    config_payload = _config_payload(resolved_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = resolved_settings
        app.state.journal = journal
        app.state.controller = controller
        app.state.monitor = monitor
        if resolved_settings.runtime.start_worker_with_api:
            await controller.start_worker_task()
        yield
        await controller.shutdown()

    app = FastAPI(
        title="FX Forward Test API",
        version="1.0.0",
        description="MetaTrader 5 FX forward-testing control plane.",
        lifespan=lifespan,
    )
    limiter = SlidingWindowRateLimiter(resolved_settings.runtime.api_rate_limit_per_minute)

    @app.middleware("http")
    async def protect_api(request: Request, call_next):
        # Browser CORS preflight requests do not include the API key.  They
        # must reach CORSMiddleware so it can return the appropriate
        # Access-Control-Allow-* headers; authenticating OPTIONS here causes
        # remote Vercel browsers to fail before the actual request is sent.
        if request.method == "OPTIONS":
            return await call_next(request)
        if request.url.path.startswith("/api/"):
            key = request.headers.get("X-API-Key", "")
            if not resolved_settings.runtime.api_key or key != resolved_settings.runtime.api_key:
                return JSONResponse(status_code=401, content={"detail": "invalid or missing API key"})
            client = request.client.host if request.client else "unknown"
            if not limiter.allow(f"{client}:{key[-8:]}"):
                return JSONResponse(status_code=429, content={"detail": "rate limit exceeded"})
        return await call_next(request)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(dict.fromkeys((resolved_settings.runtime.frontend_origin, *resolved_settings.runtime.cors_origins))),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["X-API-Key", "Content-Type", "ngrok-skip-browser-warning"],
    )

    def require_api_key(x_api_key: str | None = Depends(api_key_header)) -> None:
        if x_api_key != resolved_settings.runtime.api_key:
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    # The /health endpoint is intentionally unauthenticated so load balancers
    # and container orchestrators can probe it.
    @app.get("/health")
    def health() -> dict[str, Any]:
        environment = "mt5-demo" if resolved_settings.broker.demo_only else "mt5"
        return {"ok": True, "environment": environment, "time": datetime.now(timezone.utc).isoformat()}

    # All /api/* read endpoints now require a valid API key.  This closes the
    # gap identified in imp.txt: "Authenticate sensitive read endpoints, not
    # just control POSTs."

    @app.get("/api/status", dependencies=[Depends(require_api_key)])
    def status() -> dict[str, Any]:
        state = row_to_dict(journal.get_state())
        return {
            **state,
            "worker_task_running": bool(controller.task and not controller.task.done()),
            "demo_only": resolved_settings.broker.demo_only,
            "live_release_approved": resolved_settings.runtime.live_release_approved,
        }

    @app.get("/api/positions", dependencies=[Depends(require_api_key)])
    def positions() -> list[dict[str, Any]]:
        return [row_to_dict(row) for row in journal.current_positions()]

    @app.get("/api/trades", dependencies=[Depends(require_api_key)])
    def trades(
        instrument: str | None = None,
        state: str | None = None,
        outcome: str | None = Query(default=None, pattern="^(win|loss)?$"),
        start: str | None = None,
        end: str | None = None,
        limit: int = Query(default=500, ge=1, le=2000),
    ) -> list[dict[str, Any]]:
        rows = journal.filtered_trades(
            instrument=instrument,
            state=state,
            outcome=outcome,
            start=_parse_query_datetime(start),
            end=_parse_query_datetime(end),
            limit=limit,
        )
        return [row_to_dict(row) for row in rows]

    @app.get("/api/equity", dependencies=[Depends(require_api_key)])
    def equity(limit: int = Query(default=500, ge=1, le=5000)) -> list[dict[str, Any]]:
        return [row_to_dict(row) for row in journal.latest_equity(limit=limit)]

    @app.get("/api/signals", dependencies=[Depends(require_api_key)])
    def signals(limit: int = Query(default=200, ge=1, le=1000)) -> list[dict[str, Any]]:
        return [row_to_dict(row) for row in journal.recent_signals(limit=limit)]

    @app.get("/api/orders", dependencies=[Depends(require_api_key)])
    def orders(limit: int = Query(default=200, ge=1, le=1000)) -> list[dict[str, Any]]:
        return [row_to_dict(row) for row in journal.recent_orders(limit=limit)]

    @app.get("/api/ai-deliberations", dependencies=[Depends(require_api_key)])
    def ai_deliberations(limit: int = Query(default=200, ge=1, le=1000)) -> list[dict[str, Any]]:
        """Research/audit records only; this endpoint has no trading controls."""
        return [row_to_dict(row) for row in journal.recent_ai_deliberations(limit=limit)]

    @app.get("/api/performance", dependencies=[Depends(require_api_key)])
    def performance(instrument: str | None = None, start: str | None = None, end: str | None = None) -> dict[str, Any]:
        return performance_summary(
            journal,
            instrument=instrument,
            start=_parse_query_datetime(start),
            end=_parse_query_datetime(end),
        )

    @app.get("/api/config", dependencies=[Depends(require_api_key)])
    def config() -> dict[str, Any]:
        return config_payload

    @app.get("/api/monitoring", dependencies=[Depends(require_api_key)])
    def monitoring() -> dict[str, Any]:
        """Operational health dashboard — stale prices, DB locks, clock, etc."""
        snap = monitor.snapshot()
        return {
            "stale_prices": snap.stale_prices,
            "stale_scan_age_seconds": snap.stale_scan_age_seconds,
            "db_lock_retries": snap.db_lock_retries,
            "db_lock_alert": snap.db_lock_alert,
            "broker_connected": snap.broker_connected,
            "unknown_orders": snap.unknown_orders,
            "risk_halted": snap.risk_halted,
            "risk_halt_reason": snap.risk_halt_reason,
            "news_data_age_seconds": snap.news_data_age_seconds,
            "news_data_fresh": snap.news_data_fresh,
            "clock_health": snap.clock_health,
            "recent_alerts": snap.recent_alerts,
            "uptime_seconds": snap.uptime_seconds,
        }

    @app.get("/api/news", dependencies=[Depends(require_api_key)])
    def news_upcoming(limit: int = Query(default=50, ge=1, le=200)) -> dict[str, Any]:
        """Synchronized economic calendar + live blackout state per instrument.

        Uses the worker's NewsGateway so the dashboard always reflects the same
        events the trading engine relies on. When the worker is not started, a
        single gateway is built lazily and reused so dashboard polls share the
        same throttle (no per-request provider fetches).
        """
        gateway = getattr(controller.worker, "news", None)
        if gateway is None:
            gateway = _app_state_news_gateway(resolved_settings, monitor)
        snapshot = gateway.ensure_current()
        now = datetime.now(timezone.utc)

        def _event_dict(event):
            return {
                "event_id": event.event_id,
                "name": event.name,
                "currency": event.currency,
                "country": event.country,
                "impact": event.impact,
                "impact_score": event.impact_score,
                "scheduled_at": event.starts_at.isoformat(),
                "forecast": event.forecast,
                "previous": event.previous,
                "actual": event.actual,
                "source": event.source,
                "description": event.description,
                "source_url": event.source_url,
                "status": event.status,
                "confidence": event.confidence,
            }

        events = [_event_dict(event) for event in snapshot.events]
        events.sort(key=lambda item: str(item["scheduled_at"]))
        blackout: dict[str, str] = {}
        decisions: dict[str, dict[str, Any]] = {}
        for name in resolved_settings.instruments:
            decision = can_trade(
                name,
                now,
                news_state=snapshot,
                settings=resolved_settings.strategy,
            )
            decisions[name] = decision.as_dict()
            if not decision.allowed and decision.reason.startswith("news_"):
                blackout[name] = decision.reason
        return {
            "state": snapshot.known_state,
            "stale": snapshot.stale,
            "last_updated": None if snapshot.last_updated is None else snapshot.last_updated.isoformat(),
            "age_seconds": snapshot.age_seconds,
            "source": snapshot.source,
            "blackout_impact_score_min": resolved_settings.strategy.news_blackout_impact_score_min,
            "events": events[:limit],
            "event_count": len(events),
            "blackouts": blackout,
            "decisions": decisions,
        }

    @app.post("/api/control/start", dependencies=[Depends(require_api_key)])
    async def start_control(request: ControlRequest | None = None) -> dict[str, Any]:
        await controller.start_worker_task()
        reason = request.reason if request else "manual_start"
        return controller.set_state(BotRunState.RUNNING, reason or "manual_start")

    @app.post("/api/control/pause", dependencies=[Depends(require_api_key)])
    def pause_control(request: ControlRequest | None = None) -> dict[str, Any]:
        reason = request.reason if request else "manual_pause"
        return controller.set_state(BotRunState.PAUSED, reason or "manual_pause")

    @app.post("/api/control/stop", dependencies=[Depends(require_api_key)])
    def stop_control(request: ControlRequest | None = None) -> dict[str, Any]:
        reason = request.reason if request else "manual_stop"
        return controller.set_state(BotRunState.STOPPED, reason or "manual_stop")

    @app.websocket("/ws/live")
    async def live(websocket: WebSocket) -> None:
        supplied_key = websocket.headers.get("X-API-Key") or websocket.query_params.get("api_key", "")
        if not resolved_settings.runtime.api_key or supplied_key != resolved_settings.runtime.api_key:
            await websocket.close(code=1008)
            return
        await broadcaster.connect(websocket)
        try:
            while True:
                await websocket.send_json(live_snapshot(journal, config_payload))
                await asyncio.sleep(2.0)
        except WebSocketDisconnect:
            await broadcaster.disconnect(websocket)

    # ------------------------------------------------------------------
    # Agent endpoints
    # ------------------------------------------------------------------

    def _build_agent():
        from forex_agent.agent.analyst import TradeAnalyst
        return TradeAnalyst(
            database_url=resolved_settings.runtime.database_url,
            jsonl_path=resolved_settings.runtime.log_jsonl_path,
        )

    @app.get("/api/agent/health")
    def agent_health() -> dict[str, Any]:
        analyst = _build_agent()
        health = analyst.get_strategy_health()
        return _normalize_health(health)

    @app.get("/api/agent/analyze")
    def agent_analyze() -> dict[str, Any]:
        analyst = _build_agent()
        report = analyst.run_full_analysis()
        return {
            "total_trades": report.total_trades,
            "metrics": report.metrics.to_dict(),
            "health": _normalize_health(report.health),
            "failures": [f.to_dict() for f in report.failures],
            "winner_analyses": report.winner_analyses,
            "anomalies": report.anomalies,
            "regime_analysis": report.regime_analysis,
            "risk_analysis": report.risk_analysis,
            "execution_analysis": report.execution_analysis,
            "recurring_patterns": report.recurring_patterns,
            "alerts": [a.to_dict() if hasattr(a, "to_dict") else a for a in report.alerts],
            "bootstrap_expectancy_ci": report.bootstrap_expectancy_ci,
        }

    @app.get("/api/agent/diagnose/{trade_id}")
    def agent_diagnose(trade_id: str) -> dict[str, Any]:
        analyst = _build_agent()
        diag = analyst.diagnose_trade(trade_id)
        if diag is None:
            raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")
        return _normalize_diagnostic(diag)

    @app.get("/api/agent/evidence/{trade_id}")
    def agent_evidence(trade_id: str) -> dict[str, Any]:
        analyst = _build_agent()
        pkg = analyst.build_evidence_package(trade_id)
        if pkg is None:
            raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")
        return _normalize_evidence(pkg)

    @app.get("/api/agent/similar/{trade_id}")
    def agent_similar(trade_id: str) -> dict[str, Any]:
        analyst = _build_agent()
        result = analyst.find_similar(trade_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")
        return _normalize_similar(result, analyst.trades)

    @app.post("/api/agent/critique")
    def agent_critique(request: CritiqueRequest) -> dict[str, Any]:
        analyst = _build_agent()
        assessment = analyst.critique_finding(
            request.finding,
            request.trade_ids,
            request.initial_confidence,
        )
        return _normalize_critique(assessment)

    @app.get("/api/agent/explain/{trade_id}")
    def agent_explain(trade_id: str) -> dict[str, Any]:
        analyst = _build_agent()
        result = analyst.explain_trade(trade_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")
        return result

    @app.get("/api/agent/anomalies")
    def agent_anomalies() -> list[dict[str, Any]]:
        analyst = _build_agent()
        return analyst.get_anomalies()

    @app.get("/api/agent/experiments")
    def agent_experiments() -> list[dict[str, Any]]:
        from forex_agent.agent.hypotheses import propose_experiments
        analyst = _build_agent()
        report = analyst.run_full_analysis()
        experiments = propose_experiments(report)
        return [
            {
                "hypothesis": e.hypothesis,
                "reason": e.reason,
                "metric": e.metric,
                "statistical_test": e.statistical_test,
                "min_sample_size": e.min_sample_size,
                "acceptance_criteria": e.acceptance_criteria,
                "rejection_criteria": e.rejection_criteria,
                "overfitting_risk": e.overfitting_risk,
                "out_of_sample_plan": e.out_of_sample_plan,
            }
            for e in experiments
        ]

    @app.get("/api/agent/research")
    def agent_research() -> dict[str, Any]:
        from forex_agent.agent.research_memory import ResearchMemory
        from forex_agent.config import load_config
        analyst = _build_agent()
        trades = analyst.trades
        config = load_config()
        mem = ResearchMemory(config.research_memory_path)
        return _build_research_summary(mem, trades)

    @app.get("/api/agent/diagnose-all")
    def agent_diagnose_all() -> dict[str, Any]:
        analyst = _build_agent()
        diagnostics = analyst.analyze_all_trades()
        outcomes: dict[str, int] = {}
        dims: dict[str, int] = {}
        for d in diagnostics:
            outcomes[d.outcome] = outcomes.get(d.outcome, 0) + 1
            dims[d.primary_dimension.value] = dims.get(d.primary_dimension.value, 0) + 1
        return {
            "total": len(diagnostics),
            "outcomes": outcomes,
            "dimensions": dims,
            "diagnostics": [_normalize_diagnostic(d) for d in diagnostics[:100]],
        }

    dashboard_dir = Path("frontend/dist")
    if dashboard_dir.exists():
        app.mount("/", StaticFiles(directory=dashboard_dir, html=True), name="dashboard")

    return app


# --------------------------------------------------------------------------
# Agent response normalizers
#
# The agent domain models carry rich, nested dataclasses. These helpers map
# them onto a flat, frontend-friendly contract consumed by AgentPanel.tsx so
# the UI does not need to know about internal schema shapes.
# --------------------------------------------------------------------------


def _normalize_diagnostic(diag: Any) -> dict[str, Any]:
    d = diag.to_dict()
    r_multiple = getattr(diag, "r_multiple", None)
    if r_multiple is None:
        r_multiple = None
    return {
        "trade_id": d.get("trade_id", ""),
        "outcome": d.get("outcome", ""),
        "r_multiple": r_multiple,
        "primary_dimension": d.get("primary_dimension", "unknown"),
        "primary_diagnosis": d.get("primary_diagnosis", ""),
        "confidence": d.get("confidence", 0.0),
        "evidence_level": d.get("evidence_level", "observation"),
        "contributing_factors": [_normalize_factor(f) for f in d.get("contributing_factors", [])],
        "protective_factors": [_normalize_factor(f) for f in d.get("protective_factors", [])],
        "observations": d.get("observations", []),
        "hypotheses": d.get("hypotheses", []),
        "unknowns": d.get("unknowns", []),
        "sample_sizes": d.get("sample_sizes", {}),
        "statistical_support": d.get("statistical_support", {}),
        "counterfactuals": d.get("counterfactuals", []),
    }


def _normalize_factor(f: dict[str, Any] | None) -> dict[str, Any]:
    f = f or {}
    return {
        "dimension": f.get("dimension", "unknown"),
        "level": f.get("level", "observation"),
        "title": f.get("label", ""),
        "detail": f.get("description", ""),
        "weight": f.get("effect_size", 0.0),
        "sample_size": f.get("sample_size", 0),
    }


def _normalize_health(h: Any) -> dict[str, Any]:
    d = h.to_dict()
    status = d.get("health_status", "insufficient_data")
    metrics = d.get("components", {})
    return {
        "score": d.get("score", 0.0),
        "grade": d.get("grade", "F"),
        "status": status,
        "explanation": d.get("explanation", ""),
        "metrics": metrics,
        "warnings": [r for r in d.get("recommendations", []) if "low" in r or "below" in r or "exceed" in r or "negativ" in r],
        "opportunities": [r for r in d.get("recommendations", []) if "low" not in r and "below" not in r and "exceed" not in r and "negativ" not in r],
    }


def _normalize_evidence(pkg: Any) -> dict[str, Any]:
    d = pkg.to_dict()
    anomalies = d.get("anomalies", [])
    return {
        "trade_id": d.get("trade_id", ""),
        "baseline": d.get("baseline", {}),
        "similar_trades": d.get("similar_trades", {}),
        "regime": d.get("regime", {}),
        "execution": d.get("execution", {}),
        "timing": d.get("timing", {}),
        "risk": d.get("risk", {}),
        "anomalies": anomalies,
        "counterfactuals": d.get("counterfactuals", []),
        "statistical_tests": d.get("statistical_tests", []),
        "confidence": d.get("confidence", 0.0),
    }


def _normalize_similar(result: Any, trades: list[Any]) -> dict[str, Any]:
    from forex_agent.data.ingestion import compute_r_multiple

    d = result.to_dict()
    by_id = {t.trade_id: t for t in trades}
    matches: list[dict[str, Any]] = []
    for tid in d.get("matching_trade_ids", []):
        t = by_id.get(tid)
        if t is None:
            continue
        r = compute_r_multiple(t)
        matches.append({
            "trade_id": tid,
            "similarity_score": 1.0,
            "outcome": "win" if (t.pnl > 0 if t.exit_price is not None else False) else ("open" if t.exit_price is None else "loss"),
            "r_multiple": r if r is not None else 0.0,
            "entry_time": t.entry_time.isoformat() if t.entry_time is not None else "",
            "instrument": t.symbol,
        })
    return {
        "trade_id": d.get("trade_id", ""),
        "match_count": d.get("n_matches", 0),
        "definition": d.get("definition_of_similar", ""),
        "sample_size_warning": d.get("sample_size_warning", ""),
        "win_rate": d.get("win_rate", 0.0),
        "expectancy_r": d.get("expectancy_r", 0.0),
        "outcome_distribution": d.get("outcome_distribution", {}),
        "matches": matches,
    }


def _normalize_critique(assessment: Any) -> dict[str, Any]:
    d = assessment.to_dict()
    return {
        "finding": d.get("finding", ""),
        "status": d.get("status", ""),
        "initial_confidence": d.get("initial_confidence", 0.0),
        "adjusted_confidence": d.get("adjusted_confidence", 0.0),
        "challenges": d.get("challenges", []),
        "sample_size_warning": d.get("sample_size_concern", False),
        "independence_warning": d.get("independence_concern", False),
        "survivorship_bias_warning": d.get("survivorship_bias_concern", False),
        "look_ahead_bias_warning": d.get("look_ahead_bias_concern", False),
        "overfitting_warning": d.get("overfitting_concern", False),
        "multiple_testing_warning": d.get("multiple_testing_concern", False),
        "out_of_sample_instability": d.get("out_of_sample_concern", False),
        "economic_meaningfulness": d.get("economic_meaningfulness", ""),
        "alternative_explanations": d.get("alternative_explanations", []),
        "recommendation": d.get("status", ""),
    }


def _build_research_summary(mem: Any, trades: list[Any]) -> dict[str, Any]:
    findings = [{"id": i, **f} for i, f in enumerate(mem.get_findings())]
    hypotheses = [
        {
            "hypothesis": h.hypothesis,
            "status": h.status.value,
            "date_created": getattr(h, "date_created", ""),
            "result": getattr(h, "result", ""),
            "sample_size": getattr(h, "sample_size", 0),
        }
        for h in mem.get_hypotheses()
    ]
    recent_decisions = [
        {
            "timestamp": d.get("timestamp") or d.get("date", ""),
            "type": d.get("type", "decision"),
            "description": d.get("description", "") or d.get("finding", "") or d.get("hypothesis", ""),
        }
        for d in mem.get_decisions()[-20:]
    ]
    return {
        "total_findings": mem.summary().get("total_findings", len(findings)),
        "total_hypotheses": mem.summary().get("total_hypotheses", len(hypotheses)),
        "total_experiments": mem.summary().get("total_experiments", 0),
        "total_decisions": mem.summary().get("total_decisions", len(recent_decisions)),
        "total_trades": len(trades),
        "findings": findings,
        "hypotheses": hypotheses,
        "recent_decisions": recent_decisions,
    }


def _config_payload(settings: FxBotSettings) -> dict[str, Any]:
    payload = asdict(settings)
    # ``asdict`` includes nested credentials. The config endpoint is useful to
    # operators, but it must never become a secret-disclosure endpoint.
    strategy_payload = dict(payload.get("strategy") or {})
    strategy_payload.pop("news_api_key", None)
    strategy_payload.pop("news_api_endpoint", None)
    strategy_payload["news_api_key_configured"] = bool(settings.strategy.news_api_key)
    strategy_payload["news_api_endpoint_configured"] = bool(settings.strategy.news_api_endpoint)
    payload["strategy"] = strategy_payload
    ai_payload = dict(payload.get("ai") or {})
    ai_payload.pop("api_key", None)
    ai_payload.pop("endpoint", None)
    ai_payload["api_key_configured"] = bool(settings.ai.api_key)
    ai_payload["endpoint_configured"] = bool(settings.ai.endpoint)
    payload["ai"] = ai_payload
    payload["broker"] = {
        "provider": settings.broker.provider,
        "server": redact(settings.broker.server),
        "configured": settings.broker.configured,
        "login_hint": _login_hint(str(settings.broker.login or "")),
        "terminal_path_configured": bool(settings.broker.terminal_path),
        "portable": settings.broker.portable,
        "timeout_ms": settings.broker.timeout_ms,
        "demo_only": settings.broker.demo_only,
        "deviation_points": settings.broker.deviation_points,
        "magic_number": settings.broker.magic_number,
        "order_filling": settings.broker.order_filling,
        "symbol_map": settings.broker.symbol_map,
    }
    payload["runtime"] = {
        "database_url": settings.runtime.database_url,
        "loop_interval_seconds": settings.runtime.loop_interval_seconds,
        "log_jsonl_path": settings.runtime.log_jsonl_path,
        "frontend_origin": settings.runtime.frontend_origin,
        "start_worker_with_api": settings.runtime.start_worker_with_api,
        "bind_host": settings.runtime.bind_host,
        "api_port": settings.runtime.api_port,
        "api_rate_limit_per_minute": settings.runtime.api_rate_limit_per_minute,
    }
    payload["demo_only"] = settings.broker.demo_only
    payload["security"] = {
        "api_key_configured": bool(settings.runtime.api_key),
        "live_release_approved": settings.runtime.live_release_approved,
        "bind_host": settings.runtime.bind_host,
        "cors_origins": list(dict.fromkeys((settings.runtime.frontend_origin, *settings.runtime.cors_origins))),
    }
    payload["dashboard_badge"] = "MT5 DEMO / FORWARD TEST" if settings.broker.demo_only else "MT5 / FORWARD TEST"
    return _jsonable_payload(payload)


def _login_hint(login: str) -> str:
    if not login:
        return ""
    return f"...{login[-4:]}" if len(login) > 4 else "configured"


def _parse_query_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid datetime: {value}") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _jsonable_payload(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_payload(item) for item in value]
    return value


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = settings_from_env()
    uvicorn.run("fxbot.api:app", host=settings.runtime.bind_host, port=settings.runtime.api_port, reload=False)
