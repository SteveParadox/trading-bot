# FX Forward-Test Dashboard

React dashboard for the MT5 FX forward-test API. It supports live updates over
WebSocket, polling fallback, authenticated controls, and the trade-analysis
workspace.

## Run locally

Start the backend from the repository root:

```powershell
uvicorn fxbot.api:app --host 127.0.0.1 --port 8000
```

Then start the Vite development server:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`, enter the `FX_API_KEY` configured for the
backend, then select **Connect**. The dashboard saves the key only in the
current browser's local storage and sends it as `X-API-Key` for API calls.

## Deployment and API URL

`npm run build` produces `frontend/dist`. When FastAPI serves that directory,
the dashboard uses the page's own origin automatically; it does not point to a
visitor's localhost. For a separately hosted frontend, set this at build time:

```powershell
$env:VITE_API_BASE_URL = "https://api.example.com"
npm run build
```

The API must allow the frontend origin through `FX_FRONTEND_ORIGIN` or
`FX_CORS_ORIGINS`.

## Interface contract

The dashboard consumes `/api/status`, `/api/positions`, `/api/trades`,
`/api/equity`, `/api/performance`, `/api/config`, `/api/signals`,
`/api/orders`, the `/ws/live` stream, and `/api/agent/*` analysis endpoints.
Requests time out after 15 seconds with a user-facing connectivity message;
when the WebSocket is unavailable, the dashboard falls back to five-second
polling.
