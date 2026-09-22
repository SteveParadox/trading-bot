/**
 * Use the API origin explicitly during local Vite development, but default to
 * the current origin for the production bundle. The FastAPI application serves
 * `frontend/dist` itself, so a hard-coded localhost URL breaks remote deploys.
 */
const configuredApiBase = import.meta.env.VITE_API_BASE_URL?.trim();
const API_BASE = (configuredApiBase || (import.meta.env.DEV ? "http://127.0.0.1:8000" : window.location.origin)).replace(/\/$/, "");
const DEFAULT_TIMEOUT_MS = 15_000;

function isNgrokEndpoint(apiBase: string): boolean {
  try {
    const hostname = new URL(apiBase).hostname.toLowerCase();
    return hostname.endsWith(".ngrok-free.app") || hostname.endsWith(".ngrok-free.dev") || hostname.endsWith(".ngrok.io");
  } catch {
    return false;
  }
}

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export type LiveSnapshot = {
  status: Record<string, unknown>;
  positions: unknown[];
  equity_curve: unknown[];
  performance: Record<string, unknown>;
  recent_trades: unknown[];
  recent_signals: unknown[];
  config: Record<string, unknown>;
};

export function apiUrl(path: string): string {
  return `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
}

export function websocketUrl(path: string): string {
  const key = readStoredApiKey();
  const base = `${API_BASE.replace(/^http/, "ws")}${path.startsWith("/") ? path : `/${path}`}`;
  if (!key) {
    return base;
  }
  const separator = base.includes("?") ? "&" : "?";
  return `${base}${separator}api_key=${encodeURIComponent(key)}`;
}

export async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers ?? {});
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), DEFAULT_TIMEOUT_MS);
  const signal =
    init?.signal && typeof AbortSignal.any === "function"
      ? AbortSignal.any([init.signal, controller.signal])
      : init?.signal ?? controller.signal;

  // Always attach the stored API key. The backend requires X-API-Key on every
  // /api/* endpoint, not just control POSTs. Callers may still pass their own
  // X-API-Key; fetchJson will not override an explicit value.
  if (!headers.has("X-API-Key")) {
    const storedKey = readStoredApiKey();
    if (storedKey) {
      headers.set("X-API-Key", storedKey);
    }
  }

  // ngrok's free browser-warning page otherwise returns a 200 response
  // without the backend's CORS headers. This header tells ngrok to forward
  // browser API calls directly to FastAPI.
  if (isNgrokEndpoint(API_BASE) && !headers.has("ngrok-skip-browser-warning")) {
    headers.set("ngrok-skip-browser-warning", "true");
  }

  try {
    const response = await fetch(apiUrl(path), { ...init, headers, signal });
    const body = await response.text();
    let payload: unknown = null;

    if (body) {
      try {
        payload = JSON.parse(body);
      } catch {
        payload = null;
      }
    }

    if (!response.ok) {
      const detail =
        payload && typeof payload === "object" && "detail" in payload
          ? String(payload.detail)
          : body.trim() || response.statusText || `Request failed (${response.status})`;
      throw new ApiError(detail, response.status);
    }

    if (payload === null) {
      throw new Error(`API returned an empty or invalid JSON response for ${path}`);
    }
    return payload as T;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error(`The API did not respond within ${DEFAULT_TIMEOUT_MS / 1000} seconds.`);
    }
    if (error instanceof TypeError) {
      throw new Error("Unable to reach the FX API. Check that the backend service is running and the API URL is correct.");
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

export function isLiveSnapshot(value: unknown): value is LiveSnapshot {
  if (!value || typeof value !== "object") {
    return false;
  }
  const snapshot = value as Partial<LiveSnapshot>;
  return Boolean(
    snapshot.status &&
      typeof snapshot.status === "object" &&
      snapshot.performance &&
      typeof snapshot.performance === "object" &&
      Array.isArray(snapshot.positions) &&
      Array.isArray(snapshot.equity_curve) &&
      Array.isArray(snapshot.recent_trades) &&
      Array.isArray(snapshot.recent_signals) &&
      snapshot.config &&
      typeof snapshot.config === "object",
  );
}

export function readStoredApiKey(): string {
  try {
    return window.localStorage.getItem("fx_api_key") ?? "";
  } catch {
    return "";
  }
}

export function storeApiKey(value: string): void {
  try {
    if (value) {
      window.localStorage.setItem("fx_api_key", value);
    } else {
      window.localStorage.removeItem("fx_api_key");
    }
  } catch {
    // Storage can be unavailable in private or restricted browser contexts.
  }
}

export function clearStoredApiKey(): void {
  try {
    window.localStorage.removeItem("fx_api_key");
  } catch {
    // Storage can be unavailable in private or restricted browser contexts.
  }
}
