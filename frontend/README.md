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

Use the **White / Black** button in the header to switch the dashboard and
Analyst workspace between light and dark themes, including charts and dialogs.
The selected theme is saved in this browser and restored before the page
renders. Black is the default if no theme has been saved or browser storage
is unavailable. The toggle also works on the API-key connection screen.

The theme is initialized by `frontend/public/theme-init.js`, which Vite copies
into `frontend/dist/theme-init.js` as part of the production build. This is a
same-origin external script, so it works with a Content Security Policy that
allows `script-src 'self'` without requiring `unsafe-inline`.

## Frontend verification and deployment

Run this from the repository root before deploying a FastAPI-hosted dashboard:

```powershell
python scripts/build_frontend.py
```

The build script uses locked npm dependencies, executes theme regression tests,
generates production assets, and verifies the compiled JS, CSS and theme script.
With existing `node_modules`, pass `--skip-install`.

The FastAPI service serves the prebuilt `frontend/dist` directory directly.
**Deploying source-only changes does not update that directory.** Rebuild it
on the machine/image where FastAPI will run, or use the `fx-dashboard-dist`
artifact produced by the **Frontend validation** GitHub Actions workflow.
Ensure `dist/index.html`, `dist/assets/` and `dist/theme-init.js` are
present together. For a Vercel-hosted frontend, configure Vercel to build
from `frontend/`, run `npm run build`, and serve `dist/`.

CI runs the Node theme initialization regression tests, a production build,
and Chromium smoke tests for theme switching, persistence, responsiveness and
light-mode primary-button contrast. These checks do not issue trading orders.

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
