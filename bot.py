"""Bybit USDT perpetual futures trading bot.

The bot trades only when lower-timeframe signals agree with the higher-timeframe
trend, then sizes positions from stop distance instead of blindly allocating a
large chunk of balance. It still cannot guarantee profits; the goal is to make
bad trades easier to skip and failed execution easier to survive.
"""

# FEATURE_SUMMARY
# Printed at startup by log_feature_summary(). Active values come from config.py:
# PARTIAL_TP_ENABLED, TRAILING_STOP_ENABLED, RSI_DIVERGENCE_FILTER,
# HTF_REQUIRE_MOMENTUM_CANDLE, VOLUME_SPIKE_FILTER, FUNDING_RATE_FILTER,
# SYMBOL_RANKING_ENABLED, FRESH_CROSS_FILTER, TP1_QTY_PCT, TP2_MULTIPLIER,
# BREAKEVEN_BUFFER, TRAIL_ATR_MULTIPLIER, HTF_ADX_MIN,
# VOLUME_SPIKE_MULTIPLIER, MAX_SIGNALS_PER_CYCLE, CROSS_LOOKBACK.

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from functools import lru_cache
from typing import Any, Callable
from uuid import uuid4

import pandas as pd
from pybit.unified_trading import HTTP

from config import (
    ADX_MIN,
    API_KEY,
    API_RETRIES,
    API_RETRY_DELAY,
    API_SECRET,
    ATR_SL_MULTIPLIER,
    BREAKEVEN_BUFFER,
    CANDLE_LIMIT,
    COOLDOWN_CANDLES,
    CROSS_LOOKBACK,
    DEFAULT_TP_DISTANCE,
    DRY_RUN,
    EMERGENCY_CLOSE_ON_PROTECTION_FAILURE,
    FRESH_CROSS_FILTER,
    FUNDING_RATE_FILTER,
    FUNDING_RATE_MAX_LONG,
    FUNDING_RATE_MIN_SHORT,
    HTF_ADX_MIN,
    HTF_REQUIRE_MOMENTUM_CANDLE,
    HTF_TIMEFRAME,
    LEVERAGE,
    LOOP_INTERVAL,
    MAX_ATR_PCT,
    MAX_CONSECUTIVE_LOSSES,
    MAX_DAILY_LOSS_PCT,
    MAX_ENTRY_DEVIATION_PCT,
    MAX_OPEN_POSITIONS,
    MAX_SIGNALS_PER_CYCLE,
    MAX_SPREAD_BPS,
    MAX_STOP_DISTANCE_PCT,
    MAX_TRADE_USDT,
    MIN_ATR_PCT,
    MIN_BALANCE_USDT,
    MIN_RISK_REWARD,
    MIN_STOP_DISTANCE_PCT,
    ORDER_LINK_PREFIX,
    PAPER_BALANCE_USDT,
    PARTIAL_TP_ENABLED,
    POSITION_CONFIRM_DELAY,
    POSITION_CONFIRM_RETRIES,
    POST_TRADE_COOLDOWN,
    RISK_PER_TRADE_PCT,
    RSI_DIVERGENCE_FILTER,
    RSI_DIVERGENCE_LOOKBACK,
    STOP_MODE,
    SYMBOL_RANKING_ENABLED,
    SYMBOLS,
    TESTNET,
    TIMEFRAME,
    TP1_QTY_PCT,
    TP2_MULTIPLIER,
    TP_DISTANCE,
    TP_MODE,
    TRAIL_MIN_MOVE_TICKS,
    TRAIL_ATR_MULTIPLIER,
    TRAIL_UPDATE_MIN_SECONDS,
    TRAILING_STOP_ENABLED,
    VOLUME_SPIKE_FILTER,
    VOLUME_SPIKE_MULTIPLIER,
)
from indicators import (
    SignalDecision,
    SignalScoreDetails,
    TrendDecision,
    calculate_indicators,
    calculate_signal_score_details,
    get_htf_trend,
    get_signal,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("bybit_bot")


session = HTTP(
    testnet=TESTNET,
    api_key=API_KEY,
    api_secret=API_SECRET,
    timeout=30,
)


@dataclass(frozen=True)
class InstrumentInfo:
    qty_step: float
    min_qty: float
    tick_size: float
    min_notional: float = 5.0


@dataclass(frozen=True)
class Wallet:
    equity: float
    available: float


@dataclass(frozen=True)
class MarketSnapshot:
    last: float
    bid: float
    ask: float
    spread_bps: float
    funding_rate: float | None = None

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last


@dataclass(frozen=True)
class ExitPlan:
    stop_loss: float
    take_profit: float
    risk_distance: float
    reward_distance: float
    risk_reward: float


@dataclass(frozen=True)
class PositionSize:
    qty: float
    notional: float
    margin_required: float
    risk_usdt: float


@dataclass(frozen=True)
class ProtectionOrders:
    sl_order_id: str | None
    tp1_order_id: str | None
    tp2_order_id: str | None
    tp1_qty: float
    tp2_qty: float
    tp1_price: float
    tp2_price: float | None
    partial_enabled: bool


@dataclass(frozen=True)
class TradeCandidate:
    symbol: str
    signal: str
    row: pd.Series
    last_close: float
    score_details: SignalScoreDetails
    funding_rate: float | None
    funding_reason: str

    @property
    def score(self) -> float:
        return self.score_details.score


@dataclass(frozen=True)
class FundingDecision:
    allowed: bool
    rate: float | None
    reason: str


cooldown_until: dict[str, float] = {}
daily_state: dict[str, Any] = {
    "date": None,
    "start_equity": None,
    "halted": False,
}
# In-memory trade management state. If the bot restarts while positions are
# open, partial TP/trailing state is lost; inspect open orders manually.
trail_state: dict[str, dict[str, Any]] = {}
partial_tp_state: dict[str, dict[str, Any]] = {}

ACTIVE_ORDER_STATUSES = {"New", "Untriggered", "PartiallyFilled"}


def api_request(
    label: str,
    func: Callable[..., dict[str, Any]],
    *,
    retries: int = API_RETRIES,
    retry_delay: float = API_RETRY_DELAY,
    **kwargs: Any,
) -> dict[str, Any]:
    """Call Bybit with response validation and bounded retries."""
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = func(**kwargs)
            ret_code = response.get("retCode")
            if ret_code not in (None, 0, "0"):
                ret_msg = response.get("retMsg", "unknown error")
                raise RuntimeError(f"{label} failed: retCode={ret_code} {ret_msg}")
            return response
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                break
            log.warning("%s failed on attempt %s/%s: %s", label, attempt, retries, exc)
            time.sleep(retry_delay * attempt)
    raise RuntimeError(f"{label} failed after {retries} attempt(s): {last_exc}")


def decimal_step(value: float, step: float, rounding: str = ROUND_DOWN) -> float:
    d_value = Decimal(str(value))
    d_step = Decimal(str(step))
    if d_step <= 0:
        raise ValueError("step must be positive")
    return float((d_value / d_step).to_integral_value(rounding=rounding) * d_step)


def to_exchange_str(value: float) -> str:
    text = f"{value:.12f}".rstrip("0").rstrip(".")
    return text if text else "0"


def order_link_id(symbol: str, kind: str) -> str:
    clean_prefix = "".join(ch for ch in ORDER_LINK_PREFIX if ch.isalnum())[:8]
    return f"{clean_prefix}-{symbol[:8]}-{kind[:3]}-{uuid4().hex[:10]}"


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert exchange values to finite floats without trusting the payload shape."""
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return converted if math.isfinite(converted) else default


def safe_int(value: Any, default: int = 0) -> int:
    """Convert exchange values to ints without raising on malformed fields."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def format_rate_pct(rate: float) -> str:
    return f"{rate * 100:+.4f}%"


def timeframe_seconds(interval: str) -> int:
    """Convert Bybit intervals such as '5', '5m', '1h', or 'D' to seconds."""
    value = str(interval).strip().lower()
    if value.isdigit():
        seconds = int(value) * 60
    elif value in {"d", "1d"}:
        seconds = 24 * 60 * 60
    elif value.endswith("m") and value[:-1].isdigit():
        seconds = int(value[:-1]) * 60
    elif value.endswith("h") and value[:-1].isdigit():
        seconds = int(value[:-1]) * 60 * 60
    else:
        raise ValueError(f"unsupported timeframe interval: {interval}")
    if seconds <= 0:
        raise ValueError(f"timeframe interval must be positive: {interval}")
    return seconds


def log_feature_summary() -> None:
    log.info("Feature summary:")
    log.info(
        "  Partial TP: %s | TP1 %.0f%% | TP2 x%.2f | breakeven buffer %s",
        PARTIAL_TP_ENABLED,
        TP1_QTY_PCT * 100,
        TP2_MULTIPLIER,
        format_rate_pct(BREAKEVEN_BUFFER),
    )
    log.info(
        "  Trailing stop: %s | ATR multiplier %.2f | min interval %ss | min move %s tick(s)",
        TRAILING_STOP_ENABLED,
        TRAIL_ATR_MULTIPLIER,
        TRAIL_UPDATE_MIN_SECONDS,
        TRAIL_MIN_MOVE_TICKS,
    )
    log.info(
        "  Filters: RSI divergence=%s (lookback %s) | volume spike=%s (%.2fx) | funding=%s",
        RSI_DIVERGENCE_FILTER,
        RSI_DIVERGENCE_LOOKBACK,
        VOLUME_SPIKE_FILTER,
        VOLUME_SPIKE_MULTIPLIER,
        FUNDING_RATE_FILTER,
    )
    log.info(
        "  HTF: ADX min %.1f | momentum candle required=%s | fresh cross=%s (lookback %s)",
        HTF_ADX_MIN,
        HTF_REQUIRE_MOMENTUM_CANDLE,
        FRESH_CROSS_FILTER,
        CROSS_LOOKBACK,
    )
    log.info(
        "  Ranking: enabled=%s | max signals/cycle=%s",
        SYMBOL_RANKING_ENABLED,
        MAX_SIGNALS_PER_CYCLE,
    )
    log.info(
        "  Restart note: partial TP/trailing state is in memory; inspect open orders after restart."
    )


def fetch_klines(symbol: str, interval: str) -> pd.DataFrame:
    """Fetch candles and return oldest-to-newest numeric OHLCV data."""
    response = api_request(
        f"{symbol} kline {interval}",
        session.get_kline,
        category="linear",
        symbol=symbol,
        interval=interval,
        limit=CANDLE_LIMIT,
    )
    raw = response.get("result", {}).get("list", [])
    if not raw:
        raise RuntimeError(f"[{symbol}] no candles returned for interval {interval}")

    df = pd.DataFrame(
        raw,
        columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
    )
    numeric_cols = ["open", "high", "low", "close", "volume", "turnover"]
    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["timestamp"] = pd.to_datetime(
        pd.to_numeric(df["timestamp"], errors="coerce"),
        unit="ms",
        errors="coerce",
    )
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if len(df) < 40:
        raise RuntimeError(f"[{symbol}] not enough valid candles: {len(df)}")
    return df


@lru_cache(maxsize=128)
def get_instrument_info(symbol: str) -> InstrumentInfo:
    response = api_request(
        f"{symbol} instrument info",
        session.get_instruments_info,
        category="linear",
        symbol=symbol,
    )
    instruments = response.get("result", {}).get("list", [])
    if not instruments:
        raise RuntimeError(f"[{symbol}] instrument info not found")

    info = instruments[0]
    lot = info.get("lotSizeFilter") or {}
    price = info.get("priceFilter") or {}
    qty_step = safe_float(lot.get("qtyStep"))
    min_qty = safe_float(lot.get("minOrderQty"))
    tick_size = safe_float(price.get("tickSize"))
    min_notional = safe_float(lot.get("minNotionalValue"), 5.0)
    if qty_step <= 0 or min_qty <= 0 or tick_size <= 0:
        raise RuntimeError(f"[{symbol}] malformed instrument filters: {info}")
    return InstrumentInfo(
        qty_step=qty_step,
        min_qty=min_qty,
        tick_size=tick_size,
        min_notional=max(0.0, min_notional),
    )


def get_market_snapshot(symbol: str) -> MarketSnapshot:
    response = api_request(
        f"{symbol} ticker",
        session.get_tickers,
        category="linear",
        symbol=symbol,
    )
    tickers = response.get("result", {}).get("list", [])
    if not tickers:
        raise RuntimeError(f"[{symbol}] ticker not found")

    ticker = tickers[0]
    last = safe_float(ticker.get("lastPrice"))
    bid = safe_float(ticker.get("bid1Price"), last)
    ask = safe_float(ticker.get("ask1Price"), last)
    funding_raw = ticker.get("fundingRate")
    funding_rate = safe_float(funding_raw, float("nan")) if funding_raw not in (None, "") else None
    if funding_rate is not None and not math.isfinite(funding_rate):
        funding_rate = None
    if last <= 0 or bid <= 0 or ask <= 0:
        raise RuntimeError(f"[{symbol}] invalid ticker values: {ticker}")
    if ask < bid:
        raise RuntimeError(f"[{symbol}] inverted ticker spread: bid={bid} ask={ask}")

    mid = (bid + ask) / 2
    spread_bps = ((ask - bid) / mid) * 10_000 if mid > 0 else float("inf")
    return MarketSnapshot(
        last=last,
        bid=bid,
        ask=ask,
        spread_bps=spread_bps,
        funding_rate=funding_rate,
    )


def get_funding_rate(symbol: str, snapshot: MarketSnapshot | None = None) -> float | None:
    """Fetch the current funding rate, returning None on failure so trading can continue."""
    if snapshot and snapshot.funding_rate is not None:
        return snapshot.funding_rate

    try:
        response = api_request(
            f"{symbol} funding rate",
            session.get_funding_rate_history,
            category="linear",
            symbol=symbol,
            limit=1,
        )
        records = response.get("result", {}).get("list", [])
        if not records:
            return None
        rate_raw = records[0].get("fundingRate")
        if rate_raw in (None, ""):
            return None
        funding_rate = safe_float(rate_raw, float("nan"))
        return funding_rate if math.isfinite(funding_rate) else None
    except Exception as exc:
        log.warning("[%s] Funding rate unavailable; continuing without funding filter: %s", symbol, exc)
        return None


def check_funding_rate(
    symbol: str,
    signal: str,
    snapshot: MarketSnapshot | None = None,
    known_rate: float | None = None,
) -> FundingDecision:
    """Evaluate the funding filter and fail open when the rate cannot be fetched."""
    if not FUNDING_RATE_FILTER:
        return FundingDecision(True, None, "disabled")

    funding_rate = known_rate if known_rate is not None else get_funding_rate(symbol, snapshot)
    if funding_rate is None:
        log.warning("[%s] Funding rate filter skipped: rate unavailable", symbol)
        return FundingDecision(True, None, "unavailable")

    if signal == "LONG" and funding_rate > FUNDING_RATE_MAX_LONG:
        log.info(
            "[%s] Funding rate filter: rate=%s > max=%s; skipping LONG",
            symbol,
            format_rate_pct(funding_rate),
            format_rate_pct(FUNDING_RATE_MAX_LONG),
        )
        return FundingDecision(False, funding_rate, "long_rate_too_high")
    if signal == "SHORT" and funding_rate < FUNDING_RATE_MIN_SHORT:
        log.info(
            "[%s] Funding rate filter: rate=%s < min=%s; skipping SHORT",
            symbol,
            format_rate_pct(funding_rate),
            format_rate_pct(FUNDING_RATE_MIN_SHORT),
        )
        return FundingDecision(False, funding_rate, "short_rate_too_low")

    log.info("[%s] Funding rate OK for %s: %s", symbol, signal, format_rate_pct(funding_rate))
    return FundingDecision(True, funding_rate, "ok")


def funding_rate_allows_trade(
    symbol: str,
    signal: str,
    snapshot: MarketSnapshot | None = None,
) -> bool:
    """Compatibility wrapper for code paths that only need allow/deny."""
    return check_funding_rate(symbol, signal, snapshot).allowed


def get_wallet() -> Wallet:
    if DRY_RUN:
        return Wallet(equity=PAPER_BALANCE_USDT, available=PAPER_BALANCE_USDT)

    response = api_request(
        "wallet balance",
        session.get_wallet_balance,
        accountType="UNIFIED",
        coin="USDT",
    )
    accounts = response.get("result", {}).get("list", [])
    if not accounts:
        raise RuntimeError("wallet balance response did not contain an account")
    account = accounts[0]
    coins = account.get("coin", [])
    usdt = next((coin for coin in coins if coin.get("coin") == "USDT"), None)
    if not usdt:
        raise RuntimeError("USDT wallet balance not found")

    equity = safe_float(usdt.get("equity") or account.get("totalEquity"))
    available_raw = (
        usdt.get("availableToWithdraw")
        or usdt.get("availableBalance")
        or account.get("totalAvailableBalance")
        or equity
    )
    available = safe_float(available_raw, equity)
    if equity <= 0 or available < 0:
        raise RuntimeError(f"malformed USDT wallet values: equity={equity} available={available}")
    return Wallet(equity=equity, available=available)


def get_open_position(symbol: str) -> dict[str, Any] | None:
    if DRY_RUN:
        return None

    response = api_request(
        f"{symbol} position",
        session.get_positions,
        category="linear",
        symbol=symbol,
    )
    for position in response.get("result", {}).get("list", []):
        if safe_float(position.get("size")) > 0:
            return position
    return None


def get_open_positions() -> list[dict[str, Any]]:
    if DRY_RUN:
        return []

    try:
        response = api_request(
            "open positions",
            session.get_positions,
            category="linear",
            settleCoin="USDT",
        )
        positions = response.get("result", {}).get("list", [])
        return [
            position
            for position in positions
            if position.get("symbol") in SYMBOLS and safe_float(position.get("size")) > 0
        ]
    except Exception as exc:
        log.warning("Could not fetch all open positions, falling back per symbol: %s", exc)
        positions = []
        for symbol in SYMBOLS:
            try:
                position = get_open_position(symbol)
                if position:
                    positions.append(position)
            except Exception as symbol_exc:
                log.warning("[%s] Could not fetch fallback position: %s", symbol, symbol_exc)
        return positions


def get_open_orders(symbol: str) -> list[dict[str, Any]]:
    if DRY_RUN:
        return []

    orders: list[dict[str, Any]] = []
    for order_filter in (None, "StopOrder"):
        kwargs: dict[str, Any] = {
            "category": "linear",
            "symbol": symbol,
            "openOnly": 0,
        }
        if order_filter:
            kwargs["orderFilter"] = order_filter
        try:
            response = api_request(
                f"{symbol} open orders {order_filter or 'active'}",
                session.get_open_orders,
                **kwargs,
            )
            orders.extend(response.get("result", {}).get("list", []))
        except Exception as exc:
            log.warning("[%s] Could not fetch %s orders: %s", symbol, order_filter, exc)

    deduped: dict[str, dict[str, Any]] = {}
    for order in orders:
        order_id = order.get("orderId")
        if order_id and is_active_order(order):
            deduped[order_id] = order
    return list(deduped.values())


def set_leverage(symbol: str) -> None:
    if DRY_RUN:
        log.info("[%s] DRY_RUN: leverage change skipped", symbol)
        return

    try:
        api_request(
            f"{symbol} set leverage",
            session.set_leverage,
            category="linear",
            symbol=symbol,
            buyLeverage=str(LEVERAGE),
            sellLeverage=str(LEVERAGE),
        )
        log.info("[%s] Leverage set to %sx", symbol, LEVERAGE)
    except Exception as exc:
        err = str(exc)
        if "110043" in err or "leverage not modified" in err:
            log.info("[%s] Leverage already set to %sx", symbol, LEVERAGE)
        else:
            log.warning("[%s] Set leverage failed: %s", symbol, exc)


def place_order(label: str, **kwargs: Any) -> dict[str, Any]:
    """Place one order without automatic retries to avoid duplicate live orders."""
    if DRY_RUN:
        printable = {key: value for key, value in kwargs.items() if key != "category"}
        log.info("DRY_RUN %s order: %s", label, printable)
        return {"result": {"orderId": f"DRY-RUN-{uuid4().hex[:10]}"}}

    return api_request(label, session.place_order, retries=1, **kwargs)


def require_order_id(response: dict[str, Any], label: str) -> str:
    """Extract a live order id or raise so state is not recorded with N/A IDs."""
    order_id = response.get("result", {}).get("orderId")
    if not order_id:
        raise RuntimeError(f"{label} response did not include orderId: {response}")
    return str(order_id)


def order_status(order: dict[str, Any]) -> str:
    """Return a normalized Bybit orderStatus value."""
    return str(order.get("orderStatus") or "")


def is_active_order(order: dict[str, Any]) -> bool:
    """Treat missing orderStatus from open-order endpoints as active."""
    status = order_status(order)
    return not status or status in ACTIVE_ORDER_STATUSES


def get_order_status(symbol: str, order_id: str | None, order_filter: str | None = None) -> str | None:
    """Fetch terminal/active status for a known order id from order history."""
    if not order_id or DRY_RUN:
        return None

    kwargs: dict[str, Any] = {
        "category": "linear",
        "symbol": symbol,
        "orderId": order_id,
    }
    if order_filter:
        kwargs["orderFilter"] = order_filter
    try:
        response = api_request(
            f"{symbol} order status {order_id}",
            session.get_order_history,
            **kwargs,
        )
        orders = response.get("result", {}).get("list", [])
        if not orders:
            log.warning("[%s] Order %s not found in history", symbol, order_id)
            return None
        return order_status(orders[0]) or None
    except Exception as exc:
        log.warning("[%s] Could not fetch order status for %s: %s", symbol, order_id, exc)
        return None


def cancel_order(symbol: str, order: dict[str, Any]) -> None:
    if DRY_RUN:
        return
    order_id = order.get("orderId")
    if not order_id:
        raise RuntimeError(f"[{symbol}] cannot cancel order without orderId: {order}")

    kwargs: dict[str, Any] = {
        "category": "linear",
        "symbol": symbol,
        "orderId": order_id,
    }
    if order.get("orderFilter"):
        kwargs["orderFilter"] = order["orderFilter"]
    api_request(f"{symbol} cancel stale order", session.cancel_order, retries=1, **kwargs)


def cancel_order_by_id(
    symbol: str,
    order_id: str | None,
    *,
    order_filter: str | None = None,
    label: str = "order",
) -> bool:
    if not order_id:
        return False
    if DRY_RUN:
        log.info("[%s] DRY_RUN: cancel %s %s", symbol, label, order_id)
        return True

    kwargs: dict[str, Any] = {
        "category": "linear",
        "symbol": symbol,
        "orderId": order_id,
    }
    if order_filter:
        kwargs["orderFilter"] = order_filter
    try:
        api_request(f"{symbol} cancel {label}", session.cancel_order, retries=1, **kwargs)
        log.info("[%s] Canceled %s order %s", symbol, label, order_id)
        return True
    except Exception as exc:
        log.warning("[%s] Could not cancel %s order %s: %s", symbol, label, order_id, exc)
        return False


def cancel_stale_reduce_only_orders(symbol: str) -> None:
    if DRY_RUN:
        return

    stale_orders = [
        order
        for order in get_open_orders(symbol)
        if str(order.get("reduceOnly", "")).lower() == "true"
    ]
    for order in stale_orders:
        try:
            cancel_order(symbol, order)
            log.info("[%s] Canceled stale reduce-only order %s", symbol, order["orderId"])
        except Exception as exc:
            log.warning("[%s] Could not cancel stale order %s: %s", symbol, order.get("orderId"), exc)


def audit_position_protection(
    symbol: str,
    position: dict[str, Any],
    orders: list[dict[str, Any]] | None = None,
) -> None:
    """Log visible reduce-only protection for an open position."""
    orders = orders if orders is not None else get_open_orders(symbol)
    reduce_only = [
        order for order in orders if str(order.get("reduceOnly", "")).lower() == "true"
    ]
    stop_count = sum(1 for order in reduce_only if order.get("triggerPrice"))
    target_count = sum(
        1
        for order in reduce_only
        if not order.get("triggerPrice") and order.get("orderType") == "Limit"
    )
    side = position.get("side", "?")
    size = position.get("size", "?")
    entry = position.get("avgPrice", "?")
    if stop_count == 0:
        log.warning(
            "[%s] Open %s position %s @ %s has no visible reduce-only stop",
            symbol,
            side,
            size,
            entry,
        )
    elif target_count == 0:
        log.warning(
            "[%s] Open %s position %s @ %s has stop protection but no visible TP",
            symbol,
            side,
            size,
            entry,
        )
    else:
        log.info(
            "[%s] Position open (%s %s @ %s), protection visible: %s stop / %s TP",
            symbol,
            side,
            size,
            entry,
            stop_count,
            target_count,
        )
    if stop_count > 1:
        stop_ids = ", ".join(str(order.get("orderId")) for order in reduce_only if order.get("triggerPrice"))
        log.warning("[%s] Multiple visible reduce-only stops detected: %s", symbol, stop_ids)


def check_daily_loss_limit() -> bool:
    today = str(date.today())

    if daily_state["date"] != today:
        wallet = get_wallet()
        daily_state["date"] = today
        daily_state["start_equity"] = wallet.equity
        daily_state["halted"] = False
        log.info("New trading day | starting equity: %.2f USDT", wallet.equity)
        return False

    if daily_state["halted"]:
        log.warning("Daily loss limit already hit; trading halted until tomorrow")
        return True

    try:
        wallet = get_wallet()
        start = float(daily_state["start_equity"] or 0)
        if start > 0:
            loss_pct = (start - wallet.equity) / start
            if loss_pct >= MAX_DAILY_LOSS_PCT:
                daily_state["halted"] = True
                log.warning(
                    "Daily loss limit hit: %.2f%% drawdown (%.2f USDT)",
                    loss_pct * 100,
                    start - wallet.equity,
                )
                return True
    except Exception as exc:
        log.warning("Could not check daily loss limit: %s", exc)
    return False


def recent_loss_cooldown(symbol: str) -> bool:
    if DRY_RUN:
        return False

    try:
        cooldown_seconds = COOLDOWN_CANDLES * timeframe_seconds(TIMEFRAME)
    except ValueError as exc:
        log.warning("[%s] Loss cooldown disabled: %s", symbol, exc)
        return False
    cutoff_ms = int((time.time() - cooldown_seconds) * 1000)
    try:
        response = api_request(
            f"{symbol} closed pnl",
            session.get_closed_pnl,
            category="linear",
            symbol=symbol,
            limit=1,
        )
        records = response.get("result", {}).get("list", [])
        if not records:
            return False
        last = records[0]
        closed_pnl = safe_float(last.get("closedPnl"))
        updated_time = safe_int(last.get("updatedTime"))
        if updated_time >= cutoff_ms and closed_pnl < 0:
            cooldown_until[symbol] = time.time() + cooldown_seconds
            log.info(
                "[%s] Recent loss %.4f USDT; cooling down for %s seconds",
                symbol,
                closed_pnl,
                cooldown_seconds,
            )
            return True
    except Exception as exc:
        log.warning("[%s] Could not fetch recent closed PnL: %s", symbol, exc)
    return False


def consecutive_loss_lock(symbol: str) -> bool:
    if DRY_RUN or MAX_CONSECUTIVE_LOSSES <= 0:
        return False

    try:
        response = api_request(
            f"{symbol} closed pnl streak",
            session.get_closed_pnl,
            category="linear",
            symbol=symbol,
            limit=MAX_CONSECUTIVE_LOSSES,
        )
        records = response.get("result", {}).get("list", [])
        if len(records) < MAX_CONSECUTIVE_LOSSES:
            return False
        losses = 0
        for record in records:
            if safe_float(record.get("closedPnl")) < 0:
                losses += 1
            else:
                break
        if losses >= MAX_CONSECUTIVE_LOSSES:
            log.warning(
                "[%s] %s consecutive losses; skipping until you review settings",
                symbol,
                losses,
            )
            return True
    except Exception as exc:
        log.warning("[%s] Could not check consecutive losses: %s", symbol, exc)
    return False


def check_cooldown(symbol: str) -> bool:
    now = time.time()
    expiry = cooldown_until.get(symbol)
    if expiry:
        remaining = expiry - now
        if remaining > 0:
            log.info("[%s] Cooldown active: %ss remaining", symbol, int(remaining))
            return True
        del cooldown_until[symbol]

    return recent_loss_cooldown(symbol)


def validate_market_quality(
    symbol: str,
    signal_row: pd.Series,
    last_closed_price: float,
    snapshot: MarketSnapshot,
) -> bool:
    if last_closed_price <= 0 or not math.isfinite(last_closed_price):
        log.info("[%s] Invalid signal close %.8f; skipping", symbol, last_closed_price)
        return False

    if snapshot.spread_bps > MAX_SPREAD_BPS:
        log.info(
            "[%s] Spread %.2fbps exceeds max %.2fbps; skipping",
            symbol,
            snapshot.spread_bps,
            MAX_SPREAD_BPS,
        )
        return False

    atr_pct = float(signal_row.get("atr_pct") or 0)
    if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
        log.info(
            "[%s] ATR %.2f%% outside %.2f%%-%.2f%% range; skipping",
            symbol,
            atr_pct * 100,
            MIN_ATR_PCT * 100,
            MAX_ATR_PCT * 100,
        )
        return False

    entry_deviation = abs(snapshot.mid - last_closed_price) / last_closed_price
    if entry_deviation > MAX_ENTRY_DEVIATION_PCT:
        log.info(
            "[%s] Price moved %.2f%% from signal close; skipping chase entry",
            symbol,
            entry_deviation * 100,
        )
        return False

    return True


def build_exit_plan(
    symbol: str,
    signal: str,
    entry_price: float,
    signal_row: pd.Series,
    info: InstrumentInfo,
) -> ExitPlan | None:
    atr = safe_float(signal_row.get("atr"), float("nan"))

    if STOP_MODE == "atr":
        if not math.isfinite(atr) or atr <= 0:
            log.info("[%s] ATR stop requested but ATR is unavailable", symbol)
            return None
        raw_stop = (
            entry_price - (atr * ATR_SL_MULTIPLIER)
            if signal == "LONG"
            else entry_price + (atr * ATR_SL_MULTIPLIER)
        )
    else:
        ma28 = safe_float(signal_row.get("ma28"), float("nan"))
        if not math.isfinite(ma28) or ma28 <= 0:
            log.info("[%s] Stop plan skipped: MA28 unavailable", symbol)
            return None
        raw_stop = ma28

    if signal == "LONG":
        stop_loss = decimal_step(raw_stop, info.tick_size, ROUND_DOWN)
        if stop_loss >= entry_price:
            log.info("[%s] LONG stop %.8f is not below entry %.8f", symbol, stop_loss, entry_price)
            return None
        risk_distance = entry_price - stop_loss
    else:
        stop_loss = decimal_step(raw_stop, info.tick_size, ROUND_UP)
        if stop_loss <= entry_price:
            log.info("[%s] SHORT stop %.8f is not above entry %.8f", symbol, stop_loss, entry_price)
            return None
        risk_distance = stop_loss - entry_price

    stop_pct = risk_distance / entry_price
    if stop_pct < MIN_STOP_DISTANCE_PCT or stop_pct > MAX_STOP_DISTANCE_PCT:
        log.info(
            "[%s] Stop distance %.2f%% outside %.2f%%-%.2f%% range; skipping",
            symbol,
            stop_pct * 100,
            MIN_STOP_DISTANCE_PCT * 100,
            MAX_STOP_DISTANCE_PCT * 100,
        )
        return None

    if TP_MODE == "fixed":
        reward_distance = TP_DISTANCE.get(symbol, DEFAULT_TP_DISTANCE)
    else:
        reward_distance = risk_distance * MIN_RISK_REWARD
    if reward_distance <= 0:
        log.info("[%s] Reward distance %.8f is invalid; skipping", symbol, reward_distance)
        return None

    if TP_MODE == "fixed":
        raw_take_profit = (
            entry_price + reward_distance
            if signal == "LONG"
            else entry_price - reward_distance
        )
        take_profit = round_target_price(signal, raw_take_profit, info)
    else:
        take_profit = round_min_rr_target_price(signal, entry_price, risk_distance, info)

    if signal == "LONG":
        if take_profit <= entry_price:
            log.info("[%s] LONG take-profit %.8f is not above entry", symbol, take_profit)
            return None
        reward_distance = take_profit - entry_price
    else:
        if take_profit >= entry_price:
            log.info("[%s] SHORT take-profit %.8f is not below entry", symbol, take_profit)
            return None
        reward_distance = entry_price - take_profit

    risk_reward = reward_distance / risk_distance if risk_distance > 0 else 0
    if risk_reward < MIN_RISK_REWARD:
        log.info(
            "[%s] Risk/reward %.2f below minimum %.2f; skipping",
            symbol,
            risk_reward,
            MIN_RISK_REWARD,
        )
        return None

    return ExitPlan(
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_distance=risk_distance,
        reward_distance=reward_distance,
        risk_reward=risk_reward,
    )


def calculate_position_size(
    symbol: str,
    entry_price: float,
    exit_plan: ExitPlan,
    info: InstrumentInfo,
    wallet: Wallet,
) -> PositionSize | None:
    if entry_price <= 0 or not math.isfinite(entry_price):
        log.warning("[%s] Invalid entry price %.8f", symbol, entry_price)
        return None

    if wallet.available < MIN_BALANCE_USDT:
        log.warning(
            "[%s] Available balance %.2f below minimum %.2f USDT",
            symbol,
            wallet.available,
            MIN_BALANCE_USDT,
        )
        return None

    risk_budget = wallet.equity * RISK_PER_TRADE_PCT
    if risk_budget <= 0:
        log.warning("[%s] Risk budget is zero; skipping", symbol)
        return None

    notional_cap = min(MAX_TRADE_USDT, wallet.available * LEVERAGE * 0.90)
    qty_by_risk = risk_budget / exit_plan.risk_distance
    qty_by_notional = notional_cap / entry_price
    raw_qty = min(qty_by_risk, qty_by_notional)
    qty = decimal_step(raw_qty, info.qty_step, ROUND_DOWN)

    if qty < info.min_qty:
        log.warning("[%s] Qty %s below min qty %s", symbol, qty, info.min_qty)
        return None

    notional = qty * entry_price
    if notional < info.min_notional:
        log.warning("[%s] Notional %.2f below exchange %.2f USDT minimum", symbol, notional, info.min_notional)
        return None

    margin_required = (notional / LEVERAGE) * 1.10
    if margin_required > wallet.available:
        log.warning(
            "[%s] Required margin %.2f exceeds available %.2f",
            symbol,
            margin_required,
            wallet.available,
        )
        return None

    risk_usdt = qty * exit_plan.risk_distance
    log.info(
        "[%s] Size %.8g | notional %.2f | margin %.2f | risk %.3f USDT (budget %.3f)",
        symbol,
        qty,
        notional,
        margin_required,
        risk_usdt,
        risk_budget,
    )
    return PositionSize(
        qty=qty,
        notional=notional,
        margin_required=margin_required,
        risk_usdt=risk_usdt,
    )


def wait_for_position(symbol: str) -> dict[str, Any] | None:
    for attempt in range(1, POSITION_CONFIRM_RETRIES + 1):
        position = get_open_position(symbol)
        if position:
            return position
        log.info("[%s] Waiting for position registration (%s/%s)", symbol, attempt, POSITION_CONFIRM_RETRIES)
        time.sleep(POSITION_CONFIRM_DELAY)
    return None


def emergency_close_position(symbol: str, reason: str) -> None:
    if not EMERGENCY_CLOSE_ON_PROTECTION_FAILURE:
        return

    position = get_open_position(symbol)
    if not position:
        return

    side = "Sell" if position.get("side") == "Buy" else "Buy"
    qty = float(position.get("size") or 0)
    if qty <= 0:
        return

    log.error("[%s] Emergency close triggered: %s", symbol, reason)
    try:
        place_order(
            f"{symbol} emergency close",
            category="linear",
            symbol=symbol,
            side=side,
            orderType="Market",
            qty=to_exchange_str(qty),
            reduceOnly=True,
            orderLinkId=order_link_id(symbol, "panic"),
        )
    except Exception as exc:
        log.critical("[%s] Emergency close failed. Manual action required: %s", symbol, exc)


def close_side_for_signal(signal: str) -> str:
    return "Sell" if signal == "LONG" else "Buy"


def stop_trigger_direction(signal: str) -> int:
    return 2 if signal == "LONG" else 1


def round_stop_price(signal: str, raw_price: float, info: InstrumentInfo) -> float:
    return decimal_step(raw_price, info.tick_size, ROUND_DOWN if signal == "LONG" else ROUND_UP)


def round_target_price(signal: str, raw_price: float, info: InstrumentInfo) -> float:
    return decimal_step(raw_price, info.tick_size, ROUND_DOWN if signal == "LONG" else ROUND_UP)


def round_min_rr_target_price(
    signal: str,
    entry_price: float,
    risk_distance: float,
    info: InstrumentInfo,
) -> float:
    """Round an RR target without accidentally dropping below MIN_RISK_REWARD."""
    raw_price = (
        entry_price + (risk_distance * MIN_RISK_REWARD)
        if signal == "LONG"
        else entry_price - (risk_distance * MIN_RISK_REWARD)
    )
    for _ in range(20):
        target = round_target_price(signal, raw_price, info)
        reward = target - entry_price if signal == "LONG" else entry_price - target
        if risk_distance > 0 and reward / risk_distance >= MIN_RISK_REWARD:
            return target
        raw_price = raw_price + info.tick_size if signal == "LONG" else raw_price - info.tick_size
    return round_target_price(signal, raw_price, info)


def tp2_price_from_plan(
    signal: str,
    entry_price: float,
    exit_plan: ExitPlan,
    info: InstrumentInfo,
) -> float:
    raw_price = (
        entry_price + (exit_plan.reward_distance * TP2_MULTIPLIER)
        if signal == "LONG"
        else entry_price - (exit_plan.reward_distance * TP2_MULTIPLIER)
    )
    return round_target_price(signal, raw_price, info)


def place_stop_loss_order(
    symbol: str,
    signal: str,
    qty: float,
    stop_loss: float,
    *,
    label: str = "stop loss",
) -> str | None:
    close_side = close_side_for_signal(signal)
    trigger_direction = stop_trigger_direction(signal)

    response = place_order(
        f"{symbol} {label}",
        category="linear",
        symbol=symbol,
        side=close_side,
        orderType="Market",
        qty=to_exchange_str(qty),
        triggerPrice=to_exchange_str(stop_loss),
        triggerBy="LastPrice",
        triggerDirection=trigger_direction,
        orderFilter="StopOrder",
        reduceOnly=True,
        closeOnTrigger=True,
        orderLinkId=order_link_id(symbol, "sl"),
    )
    order_id = require_order_id(response, f"{symbol} {label}")
    log.info("[%s] %s placed at %.8f | qty %.8g | order %s", symbol, label, stop_loss, qty, order_id or "N/A")
    return order_id


def place_take_profit_order(
    symbol: str,
    signal: str,
    qty: float,
    price: float,
    *,
    label: str,
) -> str | None:
    close_side = "Sell" if signal == "LONG" else "Buy"

    response = place_order(
        f"{symbol} {label}",
        category="linear",
        symbol=symbol,
        side=close_side,
        orderType="Limit",
        qty=to_exchange_str(qty),
        price=to_exchange_str(price),
        timeInForce="GTC",
        reduceOnly=True,
        orderLinkId=order_link_id(symbol, label.replace(" ", "")[:3] or "tp"),
    )
    order_id = require_order_id(response, f"{symbol} {label}")
    log.info("[%s] %s placed at %.8f | qty %.8g | order %s", symbol, label, price, qty, order_id or "N/A")
    return order_id


def split_take_profit_quantities(
    symbol: str,
    qty: float,
    info: InstrumentInfo,
) -> tuple[float, float, bool]:
    """Split position size into exchange-valid TP quantities without leaving dust."""
    if not PARTIAL_TP_ENABLED:
        return qty, 0.0, False

    pct = min(max(TP1_QTY_PCT, 0.01), 0.99)
    tp1_qty = decimal_step(qty * pct, info.qty_step, ROUND_DOWN)
    tp2_qty = decimal_step(qty - tp1_qty, info.qty_step, ROUND_DOWN)
    total_tp_qty = decimal_step(tp1_qty + tp2_qty, info.qty_step, ROUND_DOWN)
    if tp1_qty < info.min_qty or tp2_qty < info.min_qty:
        log.warning(
            "[%s] Partial TP split too small (tp1 %.8g, tp2 %.8g, min %.8g); using single TP",
            symbol,
            tp1_qty,
            tp2_qty,
            info.min_qty,
        )
        return qty, 0.0, False
    if abs(total_tp_qty - qty) > info.qty_step / 2:
        log.warning(
            "[%s] Partial TP split would leave dust (qty %.8g, tp1+tp2 %.8g); using single TP",
            symbol,
            qty,
            total_tp_qty,
        )
        return qty, 0.0, False
    return tp1_qty, tp2_qty, True


def place_protective_orders(
    symbol: str,
    signal: str,
    qty: float,
    entry_price: float,
    exit_plan: ExitPlan,
    info: InstrumentInfo,
) -> ProtectionOrders:
    sl_order_id = place_stop_loss_order(symbol, signal, qty, exit_plan.stop_loss)

    tp1_qty, tp2_qty, partial_enabled = split_take_profit_quantities(symbol, qty, info)
    tp1_price = exit_plan.take_profit
    tp2_price: float | None = None
    tp2_order_id: str | None = None

    if partial_enabled:
        tp2_price = tp2_price_from_plan(signal, entry_price, exit_plan, info)
        if (signal == "LONG" and tp2_price <= tp1_price) or (signal == "SHORT" and tp2_price >= tp1_price):
            log.warning("[%s] TP2 %.8f is not beyond TP1 %.8f; using single TP", symbol, tp2_price, tp1_price)
            partial_enabled = False
            tp1_qty = qty
            tp2_qty = 0.0
            tp2_price = None
        else:
            tp1_order_id = place_take_profit_order(symbol, signal, tp1_qty, tp1_price, label="take profit 1")
            tp2_order_id = place_take_profit_order(symbol, signal, tp2_qty, tp2_price, label="take profit 2")
            log.info(
                "[%s] Partial TP active | TP1 %.8f qty %.8g | TP2 %.8f qty %.8g",
                symbol,
                tp1_price,
                tp1_qty,
                tp2_price,
                tp2_qty,
            )
    if not partial_enabled:
        tp1_order_id = place_take_profit_order(symbol, signal, tp1_qty, tp1_price, label="take profit 1")

    log.info(
        "[%s] Protective exits ready | SL %.8f | TP1 %.8f | RR %.2f",
        symbol,
        exit_plan.stop_loss,
        exit_plan.take_profit,
        exit_plan.risk_reward,
    )
    return ProtectionOrders(
        sl_order_id=sl_order_id,
        tp1_order_id=tp1_order_id,
        tp2_order_id=tp2_order_id,
        tp1_qty=tp1_qty,
        tp2_qty=tp2_qty,
        tp1_price=tp1_price,
        tp2_price=tp2_price,
        partial_enabled=partial_enabled,
    )


def position_signal(position: dict[str, Any]) -> str:
    return "LONG" if position.get("side") == "Buy" else "SHORT"


def position_size(position: dict[str, Any]) -> float:
    """Return finite position size from Bybit payload."""
    return safe_float(position.get("size"))


def clear_position_state(symbol: str) -> None:
    had_state = symbol in partial_tp_state or symbol in trail_state
    partial_tp_state.pop(symbol, None)
    trail_state.pop(symbol, None)
    if had_state:
        log.info("[%s] Cleared in-memory partial/trailing state; no open position found", symbol)


def record_trade_state(
    symbol: str,
    signal: str,
    actual_entry: float,
    actual_qty: float,
    exit_plan: ExitPlan,
    protection: ProtectionOrders,
) -> None:
    trail_state[symbol] = {
        "active": False,
        "signal": signal,
        "sl": exit_plan.stop_loss,
        "sl_order_id": protection.sl_order_id,
        "last_update": 0.0,
    }

    if not protection.partial_enabled:
        partial_tp_state.pop(symbol, None)
        return

    partial_tp_state[symbol] = {
        "signal": signal,
        "initial_qty": actual_qty,
        "remaining_qty": actual_qty,
        "actual_entry": actual_entry,
        "tp1_order_id": protection.tp1_order_id,
        "tp2_order_id": protection.tp2_order_id,
        "sl_order_id": protection.sl_order_id,
        "tp1_qty": protection.tp1_qty,
        "tp2_qty": protection.tp2_qty,
        "breakeven_moved": False,
    }
    log.info(
        "[%s] Partial TP state recorded | TP1 order %s | SL order %s",
        symbol,
        protection.tp1_order_id or "N/A",
        protection.sl_order_id or "N/A",
    )


def replace_stop_order(
    symbol: str,
    signal: str,
    qty: float,
    stop_loss: float,
    old_order_id: str | None,
    *,
    label: str,
) -> str | None:
    """Place the replacement stop first, then cancel the old stop to avoid gaps."""
    try:
        new_order_id = place_stop_loss_order(symbol, signal, qty, stop_loss, label=label)
    except Exception as exc:
        log.error("[%s] Failed to place %s at %.8f: %s", symbol, label, stop_loss, exc)
        emergency_close_position(symbol, f"{label} placement failed")
        return None

    if old_order_id and old_order_id != new_order_id:
        canceled = cancel_order_by_id(symbol, old_order_id, order_filter="StopOrder", label=f"old {label}")
        if not canceled:
            log.error(
                "[%s] New %s %s is live, but old SL %s could not be canceled. Manual duplicate-stop review required.",
                symbol,
                label,
                new_order_id,
                old_order_id,
            )
    return new_order_id


def move_stop_to_breakeven_if_ready(
    symbol: str,
    position: dict[str, Any],
    orders: list[dict[str, Any]],
) -> bool:
    state = partial_tp_state.get(symbol)
    if not state or state.get("breakeven_moved"):
        return False

    info = get_instrument_info(symbol)
    active_order_ids = {order.get("orderId") for order in orders if order.get("orderId")}
    tp1_order_id = state.get("tp1_order_id")
    tp1_still_open = bool(tp1_order_id and tp1_order_id in active_order_ids)
    current_qty = decimal_step(position_size(position), info.qty_step, ROUND_DOWN)
    initial_qty = float(state.get("initial_qty") or 0)
    tp1_qty = float(state.get("tp1_qty") or 0)
    expected_after_tp1 = max(0.0, initial_qty - tp1_qty)
    tp1_size_reduction_confirmed = current_qty <= expected_after_tp1 + (info.qty_step / 2)
    any_size_reduced = current_qty < max(0.0, initial_qty - info.qty_step / 2)

    if tp1_still_open and not any_size_reduced:
        log.debug("[%s] TP1 still open and position size unchanged; breakeven not active yet", symbol)
        return False
    if tp1_still_open and any_size_reduced:
        status = next((order_status(order) for order in orders if order.get("orderId") == tp1_order_id), "")
        log.info(
            "[%s] TP1 order %s still open with status=%s and size reduced to %.8g; waiting for full TP1 fill",
            symbol,
            tp1_order_id,
            status or "active",
            current_qty,
        )
        return False
    if not any_size_reduced:
        log.info("[%s] TP1 order is not open, but position size has not reduced; keeping original SL", symbol)
        return False

    tp1_status = get_order_status(symbol, str(tp1_order_id), order_filter=None)
    if tp1_status and tp1_status != "Filled":
        log.warning(
            "[%s] TP1 order %s is no longer open but status=%s; not moving SL to breakeven",
            symbol,
            tp1_order_id,
            tp1_status,
        )
        return False
    if not tp1_status and not tp1_size_reduction_confirmed:
        log.warning(
            "[%s] TP1 status unavailable and size reduction %.8g -> %.8g is smaller than TP1 qty %.8g; not moving SL",
            symbol,
            initial_qty,
            current_qty,
            tp1_qty,
        )
        return False
    if tp1_status == "Filled" and not tp1_size_reduction_confirmed:
        log.warning(
            "[%s] TP1 status Filled but position size %.8g is above expected %.8g; waiting for position sync",
            symbol,
            current_qty,
            expected_after_tp1,
        )
        return False

    signal = str(state.get("signal") or position_signal(position))
    actual_entry = safe_float(state.get("actual_entry") or position.get("avgPrice"))
    if actual_entry <= 0 or current_qty <= 0:
        log.warning("[%s] Breakeven move skipped: invalid entry %.8f or qty %.8g", symbol, actual_entry, current_qty)
        return False

    raw_breakeven = (
        actual_entry * (1 + BREAKEVEN_BUFFER)
        if signal == "LONG"
        else actual_entry * (1 - BREAKEVEN_BUFFER)
    )
    breakeven_stop = round_stop_price(signal, raw_breakeven, info)
    old_sl_id = str(state.get("sl_order_id") or trail_state.get(symbol, {}).get("sl_order_id") or "")
    new_sl_id = replace_stop_order(
        symbol,
        signal,
        current_qty,
        breakeven_stop,
        old_sl_id or None,
        label="breakeven stop",
    )
    if not new_sl_id:
        return False

    state["breakeven_moved"] = True
    state["remaining_qty"] = current_qty
    state["sl_order_id"] = new_sl_id
    trail_state[symbol] = {
        "active": TRAILING_STOP_ENABLED,
        "signal": signal,
        "sl": breakeven_stop,
        "sl_order_id": new_sl_id,
        "last_update": time.time(),
    }
    log.info(
        "[%s] TP1 confirmed filled; SL moved to breakeven %.8f for remaining qty %.8g",
        symbol,
        breakeven_stop,
        current_qty,
    )
    if TRAILING_STOP_ENABLED:
        log.info("[%s] Trailing stop activated after TP1 | current trail SL %.8f", symbol, breakeven_stop)
    return True


def update_trailing_stop_if_active(symbol: str, position: dict[str, Any]) -> None:
    """Ratchet the active trailing stop after TP1, with API-spam throttles."""
    state = trail_state.get(symbol)
    if not state or not state.get("active"):
        return

    info = get_instrument_info(symbol)
    signal = str(state.get("signal") or position_signal(position))
    current_qty = decimal_step(position_size(position), info.qty_step, ROUND_DOWN)
    if current_qty < info.min_qty:
        log.info("[%s] Trailing stop skipped: remaining qty %.8g below min %.8g", symbol, current_qty, info.min_qty)
        return

    try:
        df = calculate_indicators(fetch_klines(symbol, TIMEFRAME))
        row = df.iloc[-2]
        atr = float(row.get("atr") or 0)
    except Exception as exc:
        log.warning("[%s] Trailing stop skipped: ATR unavailable (%s)", symbol, exc)
        return

    if atr <= 0:
        log.warning("[%s] Trailing stop skipped: ATR %.8f unavailable", symbol, atr)
        return

    try:
        snapshot = get_market_snapshot(symbol)
    except Exception as exc:
        log.warning("[%s] Trailing stop skipped: current price unavailable (%s)", symbol, exc)
        return

    current_price = snapshot.mid
    raw_trail = (
        current_price - (atr * TRAIL_ATR_MULTIPLIER)
        if signal == "LONG"
        else current_price + (atr * TRAIL_ATR_MULTIPLIER)
    )
    new_trail = round_stop_price(signal, raw_trail, info)
    current_trail = float(state.get("sl") or 0)
    min_move = info.tick_size * max(1, TRAIL_MIN_MOVE_TICKS)
    elapsed = time.time() - float(state.get("last_update") or 0)
    if elapsed < TRAIL_UPDATE_MIN_SECONDS:
        log.debug(
            "[%s] Trailing stop throttle active: %.1fs elapsed < %ss",
            symbol,
            elapsed,
            TRAIL_UPDATE_MIN_SECONDS,
        )
        return

    if signal == "LONG":
        if new_trail >= current_price:
            log.info(
                "[%s] Trailing stop invalid for LONG: %.8f >= current %.8f; skipping",
                symbol,
                new_trail,
                current_price,
            )
            return
        if new_trail <= current_trail + min_move:
            log.info(
                "[%s] Trailing stop unchanged | candidate %.8f has not improved current %.8f by %.8f",
                symbol,
                new_trail,
                current_trail,
                min_move,
            )
            return
    else:
        if new_trail <= current_price:
            log.info(
                "[%s] Trailing stop invalid for SHORT: %.8f <= current %.8f; skipping",
                symbol,
                new_trail,
                current_price,
            )
            return
        if current_trail > 0 and new_trail >= current_trail - min_move:
            log.info(
                "[%s] Trailing stop unchanged | candidate %.8f has not improved current %.8f by %.8f",
                symbol,
                new_trail,
                current_trail,
                min_move,
            )
            return

    old_sl_id = str(state.get("sl_order_id") or partial_tp_state.get(symbol, {}).get("sl_order_id") or "")
    new_sl_id = replace_stop_order(
        symbol,
        signal,
        current_qty,
        new_trail,
        old_sl_id or None,
        label="trailing stop",
    )
    if not new_sl_id:
        return

    state["sl"] = new_trail
    state["sl_order_id"] = new_sl_id
    state["last_update"] = time.time()
    if symbol in partial_tp_state:
        partial_tp_state[symbol]["sl_order_id"] = new_sl_id
    log.info(
        "[%s] Trailing stop updated | price %.8f | ATR %.8f | new SL %.8f",
        symbol,
        current_price,
        atr,
        new_trail,
    )


def order_price(order: dict[str, Any]) -> float:
    """Return limit or trigger price from an order payload."""
    return safe_float(order.get("triggerPrice") or order.get("price"))


def reduce_only_orders(orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return active reduce-only orders from a mixed order list."""
    return [order for order in orders if str(order.get("reduceOnly", "")).lower() == "true" and is_active_order(order)]


def stop_orders(orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return visible reduce-only conditional stop orders."""
    return [order for order in reduce_only_orders(orders) if order.get("triggerPrice")]


def target_orders(orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return visible reduce-only limit target orders."""
    return [
        order
        for order in reduce_only_orders(orders)
        if not order.get("triggerPrice") and str(order.get("orderType") or "") == "Limit"
    ]


def choose_primary_stop(signal: str, stops: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the most protective visible stop when restart recovery sees more than one."""
    priced = [order for order in stops if order_price(order) > 0]
    if not priced:
        return stops[0] if stops else None
    if signal == "LONG":
        return max(priced, key=order_price)
    return min(priced, key=order_price)


def recover_position_state_from_orders(
    symbol: str,
    position: dict[str, Any],
    orders: list[dict[str, Any]],
) -> bool:
    """Best-effort recovery after restart without placing duplicate exits."""
    signal = position_signal(position)
    stops = stop_orders(orders)
    targets = target_orders(orders)
    primary_stop = choose_primary_stop(signal, stops)
    current_qty = position_size(position)
    actual_entry = safe_float(position.get("avgPrice"))

    if not primary_stop:
        log.warning(
            "[%s] Restart recovery: open %s position has no visible stop; manual protection required",
            symbol,
            signal,
        )
        return False

    trail_state[symbol] = {
        "active": False,
        "signal": signal,
        "sl": order_price(primary_stop),
        "sl_order_id": primary_stop.get("orderId"),
        "last_update": 0.0,
        "recovered": True,
    }

    if PARTIAL_TP_ENABLED and len(targets) >= 2 and current_qty > 0 and actual_entry > 0:
        sorted_targets = sorted(targets, key=order_price, reverse=(signal == "SHORT"))
        tp1 = sorted_targets[0]
        tp2 = sorted_targets[1]
        partial_tp_state[symbol] = {
            "signal": signal,
            "initial_qty": current_qty,
            "remaining_qty": current_qty,
            "actual_entry": actual_entry,
            "tp1_order_id": tp1.get("orderId"),
            "tp2_order_id": tp2.get("orderId"),
            "sl_order_id": primary_stop.get("orderId"),
            "tp1_qty": safe_float(tp1.get("qty")),
            "tp2_qty": safe_float(tp2.get("qty")),
            "breakeven_moved": False,
            "recovered": True,
        }
        log.warning(
            "[%s] Restart recovery rebuilt partial TP state from visible orders. "
            "Trailing remains inactive until TP1 status/size confirms.",
            symbol,
        )
        return True

    log.warning(
        "[%s] Restart recovery rebuilt stop state only (%s target order(s) visible). "
        "Existing exits will be monitored, but trailing cannot be inferred safely.",
        symbol,
        len(targets),
    )
    return True


def cancel_extra_stop_orders(
    symbol: str,
    orders: list[dict[str, Any]],
    keep_order_id: str | None,
) -> None:
    """Cancel duplicate visible stops once a primary stop id is known."""
    if not keep_order_id:
        return
    for order in stop_orders(orders):
        order_id = str(order.get("orderId") or "")
        if order_id and order_id != keep_order_id:
            try:
                cancel_order_by_id(symbol, order_id, order_filter="StopOrder", label="duplicate stop")
            except Exception as exc:
                log.warning("[%s] Could not cancel duplicate stop %s: %s", symbol, order_id, exc)


def monitor_open_position(symbol: str, position: dict[str, Any]) -> None:
    """Manage an existing position before any new signal work for the symbol."""
    orders = get_open_orders(symbol)
    audit_position_protection(symbol, position, orders)

    if symbol not in partial_tp_state and symbol not in trail_state:
        if not recover_position_state_from_orders(symbol, position, orders):
            return

    keep_sl_id = str(
        trail_state.get(symbol, {}).get("sl_order_id")
        or partial_tp_state.get(symbol, {}).get("sl_order_id")
        or ""
    )
    cancel_extra_stop_orders(symbol, orders, keep_sl_id or None)

    moved_to_breakeven = move_stop_to_breakeven_if_ready(symbol, position, orders)
    if moved_to_breakeven:
        return
    update_trailing_stop_if_active(symbol, position)


def place_entry_with_exits(
    symbol: str,
    signal: str,
    entry_price: float,
    signal_row: pd.Series,
    wallet: Wallet,
) -> None:
    info = get_instrument_info(symbol)
    exit_plan = build_exit_plan(symbol, signal, entry_price, signal_row, info)
    if not exit_plan:
        return

    size = calculate_position_size(symbol, entry_price, exit_plan, info, wallet)
    if not size:
        return

    side = "Buy" if signal == "LONG" else "Sell"
    log.info(
        "[%s] Trade plan %s entry %.8f | SL %.8f | TP1 %.8f | RR %.2f",
        symbol,
        signal,
        entry_price,
        exit_plan.stop_loss,
        exit_plan.take_profit,
        exit_plan.risk_reward,
    )

    if DRY_RUN:
        place_order(
            f"{symbol} entry",
            category="linear",
            symbol=symbol,
            side=side,
            orderType="Market",
            qty=to_exchange_str(size.qty),
            reduceOnly=False,
            orderLinkId=order_link_id(symbol, "entry"),
        )
        protection = place_protective_orders(symbol, signal, size.qty, entry_price, exit_plan, info)
        record_trade_state(symbol, signal, entry_price, size.qty, exit_plan, protection)
        cooldown_until[symbol] = time.time() + POST_TRADE_COOLDOWN
        return

    try:
        entry_response = place_order(
            f"{symbol} entry",
            category="linear",
            symbol=symbol,
            side=side,
            orderType="Market",
            qty=to_exchange_str(size.qty),
            reduceOnly=False,
            orderLinkId=order_link_id(symbol, "entry"),
        )
        entry_order_id = require_order_id(entry_response, f"{symbol} entry")
        log.info(
            "[%s] %s entry sent | qty %.8g | expected %.8f | order %s",
            symbol,
            signal,
            size.qty,
            entry_price,
            entry_order_id,
        )
        cooldown_until[symbol] = time.time() + POST_TRADE_COOLDOWN

        position = wait_for_position(symbol)
        if not position:
            log.error("[%s] Entry sent but position did not register. Check Bybit manually.", symbol)
            return

        actual_entry = safe_float(position.get("avgPrice"), entry_price)
        actual_qty = decimal_step(safe_float(position.get("size"), size.qty), info.qty_step, ROUND_DOWN)
        if actual_entry <= 0 or actual_qty < info.min_qty:
            emergency_close_position(symbol, f"invalid registered position entry={actual_entry} qty={actual_qty}")
            return
        actual_plan = build_exit_plan(symbol, signal, actual_entry, signal_row, info)
        if not actual_plan:
            emergency_close_position(symbol, "could not build protective exits from actual fill")
            return

        try:
            protection = place_protective_orders(symbol, signal, actual_qty, actual_entry, actual_plan, info)
            record_trade_state(symbol, signal, actual_entry, actual_qty, actual_plan, protection)
        except Exception as exc:
            log.error("[%s] Failed to place protective orders: %s", symbol, exc)
            emergency_close_position(symbol, "protective order placement failed")
    except Exception as exc:
        log.error("[%s] Order placement failed: %s", symbol, exc)
        try:
            orphan = get_open_position(symbol)
            if orphan:
                log.error(
                    "[%s] Orphan position detected: side=%s size=%s entry=%s",
                    symbol,
                    orphan.get("side"),
                    orphan.get("size"),
                    orphan.get("avgPrice"),
                )
                emergency_close_position(symbol, "entry/protection flow failed")
        except Exception as inner_exc:
            log.warning("[%s] Could not check for orphan position: %s", symbol, inner_exc)


def log_htf_rejection(symbol: str, decision: TrendDecision) -> None:
    details = decision.details
    reason = decision.reason
    if reason == "htf_adx_filter":
        log.info(
            "[%s] HTF ADX filter: adx=%.1f < min=%.1f; skipping",
            symbol,
            details.get("adx", 0),
            details.get("required_adx", HTF_ADX_MIN),
        )
    elif reason == "htf_momentum_candle_filter":
        log.info(
            "[%s] HTF momentum candle filter: open=%.8f close=%.8f not aligned for %s; skipping",
            symbol,
            details.get("open", 0),
            details.get("close", 0),
            details.get("trend", "?"),
        )
    elif reason == "htf_di_filter":
        log.info(
            "[%s] HTF DI filter: +DI=%.1f -DI=%.1f not aligned for %s; skipping",
            symbol,
            details.get("di_plus", 0),
            details.get("di_minus", 0),
            details.get("trend", "?"),
        )
    elif reason == "htf_ma_stack_filter":
        log.info(
            "[%s] HTF MA stack filter: MA7=%.6g MA14=%.6g MA28=%.6g; skipping",
            symbol,
            details.get("ma7", 0),
            details.get("ma14", 0),
            details.get("ma28", 0),
        )
    else:
        log.info("[%s] HTF %s trend unclear (%s); skipping", symbol, HTF_TIMEFRAME, reason)


def log_signal_rejection(symbol: str, decision: SignalDecision) -> None:
    details = decision.details
    reason = decision.reason
    signal = details.get("signal", "?")

    if reason == "adx_filter":
        log.info(
            "[%s] ADX filter: adx=%.1f < min=%.1f; skipping",
            symbol,
            details.get("adx", 0),
            details.get("required_adx", ADX_MIN),
        )
    elif reason == "volume_ma_filter":
        log.info(
            "[%s] Volume MA filter: fast/slow=%.2fx < min=%.2fx; skipping %s",
            symbol,
            details.get("volume_ratio", 0),
            details.get("required_volume_ratio", 0),
            signal,
        )
    elif reason == "volume_spike_filter":
        log.info(
            "[%s] Volume spike filter: volume=%.4f slowMA=%.4f ratio=%.2fx < min=%.2fx; skipping %s",
            symbol,
            details.get("volume", 0),
            details.get("mavol_slow", 0),
            details.get("volume_spike_ratio", 0),
            details.get("required_spike_ratio", VOLUME_SPIKE_MULTIPLIER),
            signal,
        )
    elif reason == "rsi_divergence_filter":
        divergence_type = "bearish" if signal == "LONG" else "bullish"
        log.info(
            "[%s] RSI divergence filter: clear %s divergence over %s candles; skipping %s",
            symbol,
            divergence_type,
            details.get("lookback", RSI_DIVERGENCE_LOOKBACK),
            signal,
        )
    elif reason == "fresh_cross_filter":
        log.info(
            "[%s] Fresh MA cross filter: no MA7/MA14 %s cross in last %s candles; skipping %s",
            symbol,
            "bullish" if signal == "LONG" else "bearish",
            details.get("lookback", CROSS_LOOKBACK),
            signal,
        )
    elif reason == "ma_stack_filter":
        log.info(
            "[%s] No signal | MA stack filter: MA7=%.6g MA14=%.6g MA28=%.6g | ADX=%.1f",
            symbol,
            details.get("ma7", 0),
            details.get("ma14", 0),
            details.get("ma28", 0),
            details.get("adx", 0),
        )
    elif reason in {"direction_filter", "price_ma_filter"}:
        log.info(
            "[%s] %s: close=%.8f MA7=%.8f +DI=%.1f -DI=%.1f; skipping %s",
            symbol,
            reason,
            details.get("close", 0),
            details.get("ma7", 0),
            details.get("di_plus", 0),
            details.get("di_minus", 0),
            signal,
        )
    else:
        log.info("[%s] No signal (%s)", symbol, reason)


def scan_symbol(symbol: str) -> TradeCandidate | None:
    position = get_open_position(symbol)
    if position:
        monitor_open_position(symbol, position)
        return None

    clear_position_state(symbol)

    cancel_stale_reduce_only_orders(symbol)

    if check_cooldown(symbol) or consecutive_loss_lock(symbol):
        return None

    df_htf = calculate_indicators(fetch_klines(symbol, HTF_TIMEFRAME))
    htf_decision = get_htf_trend(df_htf, return_decision=True)
    assert isinstance(htf_decision, TrendDecision)
    if htf_decision.trend is None:
        log_htf_rejection(symbol, htf_decision)
        return None

    df = calculate_indicators(fetch_klines(symbol, TIMEFRAME))
    signal_decision = get_signal(df, return_decision=True)
    assert isinstance(signal_decision, SignalDecision)
    row = df.iloc[-2]
    last_close = float(row["close"])

    if not signal_decision.signal:
        log_signal_rejection(symbol, signal_decision)
        return None

    signal = signal_decision.signal
    if signal != htf_decision.trend:
        log.info("[%s] Signal %s conflicts with HTF trend %s", symbol, signal, htf_decision.trend)
        return None

    snapshot = get_market_snapshot(symbol)
    if not validate_market_quality(symbol, row, last_close, snapshot):
        return None

    funding_decision = check_funding_rate(symbol, signal, snapshot)
    if not funding_decision.allowed:
        return None

    info = get_instrument_info(symbol)
    exit_plan = build_exit_plan(symbol, signal, snapshot.mid, row, info)
    if not exit_plan:
        return None

    score_details = calculate_signal_score_details(
        df,
        signal,
        exit_plan.reward_distance,
        exit_plan.risk_distance,
    )

    log.info(
        "[%s] Candidate %s | close %.8f | mid %.8f | spread %.2fbps | score %.1f",
        symbol,
        signal,
        last_close,
        snapshot.mid,
        snapshot.spread_bps,
        score_details.score,
    )
    return TradeCandidate(
        symbol=symbol,
        signal=signal,
        row=row,
        last_close=last_close,
        score_details=score_details,
        funding_rate=funding_decision.rate,
        funding_reason=funding_decision.reason,
    )


def log_signal_ranking(candidates: list[TradeCandidate], selected: list[TradeCandidate]) -> None:
    if len(candidates) <= 1:
        return

    log.info("Signal ranking this cycle:")
    for candidate in candidates:
        details = candidate.score_details
        log.info(
            "[%s] Score: %.1f | ADX: %.1f | VolRatio: %.2fx | MA sep: %.2f%% | RR: %.2f",
            candidate.symbol,
            details.score,
            details.adx,
            details.volume_ratio,
            details.ma_separation_pct,
            details.rr_ratio,
        )

    selected_symbols = ", ".join(candidate.symbol for candidate in selected) or "none"
    if SYMBOL_RANKING_ENABLED:
        log.info("Trading %s only (max %s signal(s) per cycle)", selected_symbols, MAX_SIGNALS_PER_CYCLE)
    else:
        log.info(
            "Symbol ranking disabled; trading %s by symbol order (max %s signal(s) per cycle)",
            selected_symbols,
            MAX_SIGNALS_PER_CYCLE,
        )


def select_trade_candidates(candidates: list[TradeCandidate]) -> list[TradeCandidate]:
    if MAX_SIGNALS_PER_CYCLE <= 0:
        if candidates:
            log.info("MAX_SIGNALS_PER_CYCLE=%s; no new trades will be opened", MAX_SIGNALS_PER_CYCLE)
        return []

    ordered = sorted(candidates, key=lambda item: item.score, reverse=True) if SYMBOL_RANKING_ENABLED else candidates
    selected = ordered[:MAX_SIGNALS_PER_CYCLE]
    log_signal_ranking(ordered, selected)
    return selected


def execute_trade_candidate(candidate: TradeCandidate) -> None:
    symbol = candidate.symbol
    position = get_open_position(symbol)
    if position:
        log.info("[%s] Position opened before execution; monitoring instead of entering", symbol)
        monitor_open_position(symbol, position)
        return

    if check_cooldown(symbol) or consecutive_loss_lock(symbol):
        return

    cancel_stale_reduce_only_orders(symbol)

    open_positions = get_open_positions()
    if len(open_positions) >= MAX_OPEN_POSITIONS:
        log.info("[%s] Max open positions reached (%s); skipping", symbol, MAX_OPEN_POSITIONS)
        return

    wallet = get_wallet()
    log.info("[%s] Equity %.2f | available %.2f USDT", symbol, wallet.equity, wallet.available)

    snapshot = get_market_snapshot(symbol)
    if not validate_market_quality(symbol, candidate.row, candidate.last_close, snapshot):
        return

    if FUNDING_RATE_FILTER and candidate.funding_reason == "ok":
        if not check_funding_rate(symbol, candidate.signal, snapshot, known_rate=candidate.funding_rate).allowed:
            return
    elif FUNDING_RATE_FILTER and candidate.funding_reason == "unavailable":
        log.warning("[%s] Funding rate was unavailable during ranking; not refetching at execution", symbol)

    log.info(
        "[%s] Executing ranked %s signal | score %.1f | entry mid %.8f",
        symbol,
        candidate.signal,
        candidate.score,
        snapshot.mid,
    )
    place_entry_with_exits(symbol, candidate.signal, snapshot.mid, candidate.row, wallet)


def startup_checks() -> bool:
    log.info("=" * 68)
    log.info("Bybit USDT perp bot starting")
    log.info("Mode: %s | Dry run: %s", "TESTNET" if TESTNET else "LIVE", DRY_RUN)
    log.info("Symbols: %s", ", ".join(SYMBOLS))
    log.info(
        "Risk: %.2f%% equity/trade | max notional %.2f | max daily loss %.2f%%",
        RISK_PER_TRADE_PCT * 100,
        MAX_TRADE_USDT,
        MAX_DAILY_LOSS_PCT * 100,
    )
    log.info("Exits: stop=%s | take-profit=%s | min RR %.2f", STOP_MODE, TP_MODE, MIN_RISK_REWARD)
    log_feature_summary()
    log.info("=" * 68)

    if not DRY_RUN and (not API_KEY or not API_SECRET):
        log.error("API_KEY and API_SECRET are required when DRY_RUN=false")
        return False
    if not TESTNET and not DRY_RUN:
        log.warning("LIVE TRADING ENABLED. Verify size, symbols, and risk settings.")
    return True


def run() -> None:
    if not startup_checks():
        return

    for symbol in SYMBOLS:
        set_leverage(symbol)

    while True:
        try:
            log.info("-" * 68)
            if check_daily_loss_limit():
                log.info("Sleeping %ss until next scan", LOOP_INTERVAL)
                time.sleep(LOOP_INTERVAL)
                continue

            candidates: list[TradeCandidate] = []
            for symbol in SYMBOLS:
                try:
                    candidate = scan_symbol(symbol)
                    if candidate:
                        candidates.append(candidate)
                except Exception as exc:
                    log.error("[%s] Unexpected scan error: %s", symbol, exc, exc_info=True)

            for candidate in select_trade_candidates(candidates):
                try:
                    execute_trade_candidate(candidate)
                except Exception as exc:
                    log.error("[%s] Unexpected execution error: %s", candidate.symbol, exc, exc_info=True)

            log.info("Sleeping %ss until next scan", LOOP_INTERVAL)
            time.sleep(LOOP_INTERVAL)
        except KeyboardInterrupt:
            log.info("Shutdown requested by user")
            return


if __name__ == "__main__":
    run()
