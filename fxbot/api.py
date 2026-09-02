"""FastAPI service for the MT5 FX forward-test platform."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from fxbot.analytics import live_snapshot, performance_summary
from fxbot.config import FxBotSettings, ensure_runtime_dirs, settings_from_env
from fxbot.forward import ForwardTestWorker
from fxbot.journal import StructuredJournal, row_to_dict
from fxbot.models import BotRunState

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class ControlRequest(BaseModel):
    reason: str = ""


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
    ) -> None:
        self.settings = settings
        self.journal = journal
        self.broadcaster = broadcaster
        self.worker = ForwardTestWorker(settings, journal=journal, publisher=self.broadcaster.broadcast)
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


def create_app(settings: FxBotSettings | None = None) -> FastAPI:
    resolved_settings = settings or settings_from_env()
    ensure_runtime_dirs(resolved_settings)
    journal = StructuredJournal(resolved_settings.runtime.database_url, resolved_settings.runtime.log_jsonl_path)
    broadcaster = LiveBroadcaster()
    controller = WorkerController(settings=resolved_settings, journal=journal, broadcaster=broadcaster)
    config_payload = _config_payload(resolved_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = resolved_settings
        app.state.journal = journal
        app.state.controller = controller
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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[resolved_settings.runtime.frontend_origin, "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def require_api_key(x_api_key: str | None = Depends(api_key_header)) -> None:
        if x_api_key != resolved_settings.runtime.api_key:
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    @app.get("/health")
    def health() -> dict[str, Any]:
        environment = "mt5-demo" if resolved_settings.broker.demo_only else "mt5"
        return {"ok": True, "environment": environment, "time": datetime.now(timezone.utc).isoformat()}

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        state = row_to_dict(journal.get_state())
        return {
            **state,
            "worker_task_running": bool(controller.task and not controller.task.done()),
            "demo_only": resolved_settings.broker.demo_only,
        }

    @app.get("/api/positions")
    def positions() -> list[dict[str, Any]]:
        return [row_to_dict(row) for row in journal.current_positions()]

    @app.get("/api/trades")
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

    @app.get("/api/equity")
    def equity(limit: int = Query(default=500, ge=1, le=5000)) -> list[dict[str, Any]]:
        return [row_to_dict(row) for row in journal.latest_equity(limit=limit)]

    @app.get("/api/signals")
    def signals(limit: int = Query(default=200, ge=1, le=1000)) -> list[dict[str, Any]]:
        return [row_to_dict(row) for row in journal.recent_signals(limit=limit)]

    @app.get("/api/orders")
    def orders(limit: int = Query(default=200, ge=1, le=1000)) -> list[dict[str, Any]]:
        return [row_to_dict(row) for row in journal.recent_orders(limit=limit)]

    @app.get("/api/performance")
    def performance(instrument: str | None = None, start: str | None = None, end: str | None = None) -> dict[str, Any]:
        return performance_summary(
            journal,
            instrument=instrument,
            start=_parse_query_datetime(start),
            end=_parse_query_datetime(end),
        )

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        return config_payload

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
        await broadcaster.connect(websocket)
        try:
            while True:
                await websocket.send_json(live_snapshot(journal, config_payload))
                await asyncio.sleep(2.0)
        except WebSocketDisconnect:
            await broadcaster.disconnect(websocket)

    dashboard_dir = Path("frontend/dist")
    if dashboard_dir.exists():
        app.mount("/", StaticFiles(directory=dashboard_dir, html=True), name="dashboard")

    return app


def _config_payload(settings: FxBotSettings) -> dict[str, Any]:
    payload = asdict(settings)
    payload["broker"] = {
        "provider": settings.broker.provider,
        "server": settings.broker.server,
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
    }
    payload["demo_only"] = settings.broker.demo_only
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

    uvicorn.run("fxbot.api:app", host="127.0.0.1", port=8000, reload=False)
