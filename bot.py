"""Bybit USDT perpetual futures trading bot.

The bot trades only when lower-timeframe signals agree with the higher-timeframe
trend, then sizes positions from stop distance instead of blindly allocating a
large chunk of balance. It still cannot guarantee profits; the goal is to make
bad trades easier to skip and failed execution easier to survive.
"""

from __future__ import annotations

import logging
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
    CANDLE_LIMIT,
    COOLDOWN_CANDLES,
    DEFAULT_TP_DISTANCE,
    DRY_RUN,
    EMERGENCY_CLOSE_ON_PROTECTION_FAILURE,
    HTF_TIMEFRAME,
    LEVERAGE,
    LOOP_INTERVAL,
    MAX_ATR_PCT,
    MAX_CONSECUTIVE_LOSSES,
    MAX_DAILY_LOSS_PCT,
    MAX_ENTRY_DEVIATION_PCT,
    MAX_OPEN_POSITIONS,
    MAX_SPREAD_BPS,
    MAX_STOP_DISTANCE_PCT,
    MAX_TRADE_USDT,
    MIN_ATR_PCT,
    MIN_BALANCE_USDT,
    MIN_RISK_REWARD,
    MIN_STOP_DISTANCE_PCT,
    ORDER_LINK_PREFIX,
    PAPER_BALANCE_USDT,
    POSITION_CONFIRM_DELAY,
    POSITION_CONFIRM_RETRIES,
    POST_TRADE_COOLDOWN,
    RISK_PER_TRADE_PCT,
    STOP_MODE,
    SYMBOLS,
    TESTNET,
    TIMEFRAME,
    TP_DISTANCE,
    TP_MODE,
)
from indicators import calculate_indicators, get_htf_trend, get_signal


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


cooldown_until: dict[str, float] = {}
daily_state: dict[str, Any] = {
    "date": None,
    "start_equity": None,
    "halted": False,
}


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
    lot = info["lotSizeFilter"]
    price = info["priceFilter"]
    return InstrumentInfo(
        qty_step=float(lot["qtyStep"]),
        min_qty=float(lot["minOrderQty"]),
        tick_size=float(price["tickSize"]),
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
    last = float(ticker.get("lastPrice") or 0)
    bid = float(ticker.get("bid1Price") or last)
    ask = float(ticker.get("ask1Price") or last)
    if last <= 0 or bid <= 0 or ask <= 0:
        raise RuntimeError(f"[{symbol}] invalid ticker values: {ticker}")

    mid = (bid + ask) / 2
    spread_bps = ((ask - bid) / mid) * 10_000 if mid > 0 else float("inf")
    return MarketSnapshot(last=last, bid=bid, ask=ask, spread_bps=spread_bps)


def get_wallet() -> Wallet:
    if DRY_RUN:
        return Wallet(equity=PAPER_BALANCE_USDT, available=PAPER_BALANCE_USDT)

    response = api_request(
        "wallet balance",
        session.get_wallet_balance,
        accountType="UNIFIED",
        coin="USDT",
    )
    account = response.get("result", {}).get("list", [{}])[0]
    coins = account.get("coin", [])
    usdt = next((coin for coin in coins if coin.get("coin") == "USDT"), None)
    if not usdt:
        raise RuntimeError("USDT wallet balance not found")

    equity = float(usdt.get("equity") or account.get("totalEquity") or 0)
    available_raw = (
        usdt.get("availableToWithdraw")
        or usdt.get("availableBalance")
        or account.get("totalAvailableBalance")
        or equity
    )
    available = float(available_raw)
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
        if float(position.get("size") or 0) > 0:
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
            if position.get("symbol") in SYMBOLS and float(position.get("size") or 0) > 0
        ]
    except Exception as exc:
        log.warning("Could not fetch all open positions, falling back per symbol: %s", exc)
        positions = []
        for symbol in SYMBOLS:
            position = get_open_position(symbol)
            if position:
                positions.append(position)
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
        if order_id:
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
        log.debug("[%s] Leverage already set or could not be changed: %s", symbol, exc)


def place_order(label: str, **kwargs: Any) -> dict[str, Any]:
    if DRY_RUN:
        printable = {key: value for key, value in kwargs.items() if key != "category"}
        log.info("DRY_RUN %s order: %s", label, printable)
        return {"result": {"orderId": "DRY-RUN"}}

    return api_request(label, session.place_order, retries=1, **kwargs)


def cancel_order(symbol: str, order: dict[str, Any]) -> None:
    if DRY_RUN:
        return

    kwargs: dict[str, Any] = {
        "category": "linear",
        "symbol": symbol,
        "orderId": order["orderId"],
    }
    if order.get("orderFilter"):
        kwargs["orderFilter"] = order["orderFilter"]
    api_request(f"{symbol} cancel stale order", session.cancel_order, retries=1, **kwargs)


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


def audit_position_protection(symbol: str, position: dict[str, Any]) -> None:
    orders = get_open_orders(symbol)
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

    cooldown_seconds = COOLDOWN_CANDLES * int(TIMEFRAME) * 60
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
        closed_pnl = float(last.get("closedPnl") or 0)
        updated_time = int(last.get("updatedTime") or 0)
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
            if float(record.get("closedPnl") or 0) < 0:
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
    atr = float(signal_row.get("atr") or 0)
    ma28 = float(signal_row["ma28"])

    if STOP_MODE == "atr":
        if atr <= 0:
            log.info("[%s] ATR stop requested but ATR is unavailable", symbol)
            return None
        raw_stop = (
            entry_price - (atr * ATR_SL_MULTIPLIER)
            if signal == "LONG"
            else entry_price + (atr * ATR_SL_MULTIPLIER)
        )
    else:
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

    raw_take_profit = (
        entry_price + reward_distance
        if signal == "LONG"
        else entry_price - reward_distance
    )
    take_profit = decimal_step(
        raw_take_profit,
        info.tick_size,
        ROUND_DOWN if signal == "LONG" else ROUND_UP,
    )

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
    if notional < 5.0:
        log.warning("[%s] Notional %.2f below Bybit 5 USDT minimum", symbol, notional)
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


def place_protective_orders(
    symbol: str,
    signal: str,
    qty: float,
    exit_plan: ExitPlan,
) -> None:
    close_side = "Sell" if signal == "LONG" else "Buy"
    trigger_direction = 2 if signal == "LONG" else 1

    sl_response = place_order(
        f"{symbol} stop loss",
        category="linear",
        symbol=symbol,
        side=close_side,
        orderType="Market",
        qty=to_exchange_str(qty),
        triggerPrice=to_exchange_str(exit_plan.stop_loss),
        triggerBy="LastPrice",
        triggerDirection=trigger_direction,
        orderFilter="StopOrder",
        reduceOnly=True,
        closeOnTrigger=True,
        orderLinkId=order_link_id(symbol, "sl"),
    )
    log.info(
        "[%s] Stop placed at %.8f | order %s",
        symbol,
        exit_plan.stop_loss,
        sl_response.get("result", {}).get("orderId", "N/A"),
    )

    tp_response = place_order(
        f"{symbol} take profit",
        category="linear",
        symbol=symbol,
        side=close_side,
        orderType="Limit",
        qty=to_exchange_str(qty),
        price=to_exchange_str(exit_plan.take_profit),
        timeInForce="GTC",
        reduceOnly=True,
        orderLinkId=order_link_id(symbol, "tp"),
    )
    log.info(
        "[%s] Take-profit placed at %.8f | RR %.2f | order %s",
        symbol,
        exit_plan.take_profit,
        exit_plan.risk_reward,
        tp_response.get("result", {}).get("orderId", "N/A"),
    )


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
        "[%s] Trade plan %s entry %.8f | SL %.8f | TP %.8f | RR %.2f",
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
        place_protective_orders(symbol, signal, size.qty, exit_plan)
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
        log.info(
            "[%s] %s entry sent | qty %.8g | expected %.8f | order %s",
            symbol,
            signal,
            size.qty,
            entry_price,
            entry_response.get("result", {}).get("orderId", "N/A"),
        )
        cooldown_until[symbol] = time.time() + POST_TRADE_COOLDOWN

        position = wait_for_position(symbol)
        if not position:
            log.error("[%s] Entry sent but position did not register. Check Bybit manually.", symbol)
            return

        actual_entry = float(position.get("avgPrice") or entry_price)
        actual_qty = decimal_step(float(position.get("size") or size.qty), info.qty_step, ROUND_DOWN)
        actual_plan = build_exit_plan(symbol, signal, actual_entry, signal_row, info)
        if not actual_plan:
            emergency_close_position(symbol, "could not build protective exits from actual fill")
            return

        try:
            place_protective_orders(symbol, signal, actual_qty, actual_plan)
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


def scan_symbol(symbol: str) -> None:
    position = get_open_position(symbol)
    if position:
        audit_position_protection(symbol, position)
        return

    cancel_stale_reduce_only_orders(symbol)

    if check_cooldown(symbol) or consecutive_loss_lock(symbol):
        return

    if len(get_open_positions()) >= MAX_OPEN_POSITIONS:
        log.info("[%s] Max open positions reached (%s); skipping", symbol, MAX_OPEN_POSITIONS)
        return

    wallet = get_wallet()
    log.info("[%s] Equity %.2f | available %.2f USDT", symbol, wallet.equity, wallet.available)

    df_htf = calculate_indicators(fetch_klines(symbol, HTF_TIMEFRAME))
    htf_trend = get_htf_trend(df_htf)
    if htf_trend is None:
        log.info("[%s] HTF %s trend unclear; skipping", symbol, HTF_TIMEFRAME)
        return

    df = calculate_indicators(fetch_klines(symbol, TIMEFRAME))
    signal = get_signal(df)
    row = df.iloc[-2]
    last_close = float(row["close"])

    if not signal:
        log.info(
            "[%s] No signal | MA7=%.6g MA14=%.6g MA28=%.6g | volRatio=%.2f | ADX=%.1f%s",
            symbol,
            row.get("ma7", 0),
            row.get("ma14", 0),
            row.get("ma28", 0),
            row.get("volume_ratio", 0),
            row.get("adx", 0),
            " ranging" if float(row.get("adx") or 0) < ADX_MIN else "",
        )
        return

    if signal != htf_trend:
        log.info("[%s] Signal %s conflicts with HTF trend %s", symbol, signal, htf_trend)
        return

    snapshot = get_market_snapshot(symbol)
    if not validate_market_quality(symbol, row, last_close, snapshot):
        return

    log.info(
        "[%s] Signal %s confirmed | close %.8f | mid %.8f | spread %.2fbps",
        symbol,
        signal,
        last_close,
        snapshot.mid,
        snapshot.spread_bps,
    )
    place_entry_with_exits(symbol, signal, snapshot.mid, row, wallet)


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

            for symbol in SYMBOLS:
                try:
                    scan_symbol(symbol)
                except Exception as exc:
                    log.error("[%s] Unexpected scan error: %s", symbol, exc, exc_info=True)

            log.info("Sleeping %ss until next scan", LOOP_INTERVAL)
            time.sleep(LOOP_INTERVAL)
        except KeyboardInterrupt:
            log.info("Shutdown requested by user")
            return


if __name__ == "__main__":
    run()
