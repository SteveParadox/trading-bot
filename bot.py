# ============================================================
#  bot.py — Bybit USDT Perp Futures Trading Bot
#
#  Strategy:
#    • Trend  : MA7/MA14/MA28 alignment
#    • Volume : Dual MAVOL(9,18) confirmation
#    • Entry  : Market order when both conditions align
#    • Exit   : Limit TP at entry ± TP_DISTANCE + Stop-market SL at MA28
#               (fills the moment price *touches* the level,
#               even if the candle hasn't closed — "second price")
#    • 20x leverage | % of balance sizing
# ============================================================
from __future__ import annotations
import time
import logging
from decimal import Decimal, ROUND_DOWN

import pandas as pd
from pybit.unified_trading import HTTP

from config import (
    API_KEY, API_SECRET, TESTNET,
    SYMBOLS, TIMEFRAME, CANDLE_LIMIT,
    LEVERAGE, BALANCE_PCT,
    TP_DISTANCE, DEFAULT_TP_DISTANCE,
    LOOP_INTERVAL,
    MIN_BALANCE_USDT, 
    MAX_TRADE_USDT,
    HTF_TIMEFRAME, 
    COOLDOWN_CANDLES,      
    MAX_DAILY_LOSS_PCT,   
)
from indicators import calculate_indicators, get_signal, get_htf_trend

# ── Logging ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("bybit_bot")

# ── Bybit session ────────────────────────────────────────────
session = HTTP(
    testnet=TESTNET,
    api_key=API_KEY,
    api_secret=API_SECRET,
    timeout=30,
)

# ── State tracking ───────────────────────────────────────────
# Cooldown: maps symbol → timestamp of last detected loss
cooldown_until: dict[str, float] = {}

# Daily loss: tracks starting balance and current trading day
daily_state: dict = {
    "date"          : None,   # today's date string e.g. "2026-05-07"
    "start_balance" : None,   # balance at start of trading day
    "halted"        : False,  # True = daily loss limit hit, no more trades today
}

# ════════════════════════════════════════════════════════════
#  Market data helpers
# ════════════════════════════════════════════════════════════

def fetch_klines(symbol: str, interval: str) -> pd.DataFrame:
    """Fetch OHLCV candles and return as a clean DataFrame."""
    resp = session.get_kline(
        category="linear",
        symbol=symbol,
        interval=interval,  
        limit=CANDLE_LIMIT,
    )
    raw = resp["result"]["list"]
    df = pd.DataFrame(
        raw,
        columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
    )
    df[["open", "high", "low", "close", "volume"]] = df[
        ["open", "high", "low", "close", "volume"]
    ].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")

    # Bybit returns newest-first; reverse to chronological order
    df = df.iloc[::-1].reset_index(drop=True)
    return df


def get_instrument_info(symbol: str) -> dict:
    """
    Returns lot/price filter details needed for proper order sizing.
    Cached per call — call once per trade, not in a hot loop.
    """
    resp = session.get_instruments_info(category="linear", symbol=symbol)
    info = resp["result"]["list"][0]
    lot  = info["lotSizeFilter"]
    prc  = info["priceFilter"]
    return {
        "qty_step" : float(lot["qtyStep"]),
        "min_qty"  : float(lot["minOrderQty"]),
        "tick_size": float(prc["tickSize"]),
    }


def round_to_step(value: float, step: float) -> float:
    """Floor-round a value to the nearest valid step size."""
    d_val  = Decimal(str(value))
    d_step = Decimal(str(step))
    return float((d_val / d_step).to_integral_value(rounding=ROUND_DOWN) * d_step)


# ════════════════════════════════════════════════════════════
#  Account / position helpers
# ════════════════════════════════════════════════════════════
def get_available_balance() -> float:
    resp = session.get_wallet_balance(
        accountType="UNIFIED",
        coin="USDT"
    )

    coins = resp["result"]["list"][0]["coin"]

    usdt = next(c for c in coins if c["coin"] == "USDT")
    # Use walletBalance (total balance); availableToWithdraw can be empty
    balance_str = usdt.get("walletBalance", "0")
    if not balance_str or balance_str == "":
        balance_str = "0"
    return float(balance_str)

def get_open_position(symbol: str) -> dict | None:
    """Return the current open position for a symbol, or None."""
    resp = session.get_positions(category="linear", symbol=symbol)
    for pos in resp["result"]["list"]:
        if float(pos["size"]) > 0:
            return pos
    return None


def set_leverage(symbol: str) -> None:
    """Set leverage for a symbol (silently ignore if already set)."""
    try:
        session.set_leverage(
            category="linear",
            symbol=symbol,
            buyLeverage=str(LEVERAGE),
            sellLeverage=str(LEVERAGE),
        )
        log.info(f"[{symbol}] Leverage set to {LEVERAGE}x")
    except Exception as exc:
        # Bybit returns an error if leverage is already at the target — safe to ignore
        log.debug(f"[{symbol}] Leverage already set or minor error: {exc}")



def check_cooldown(symbol: str) -> bool:
    now = time.time()

    # If cooldown is already active and hasn't expired
    if symbol in cooldown_until:
        remaining = cooldown_until[symbol] - now
        if remaining > 0:
            log.info(
                f"[{symbol}] ⏳ Cooldown active — "
                f"{int(remaining)}s remaining ({COOLDOWN_CANDLES} candles after loss)"
            )
            return True
        else:
            del cooldown_until[symbol]

    # Only check recent trades — within the last N candles
    cooldown_seconds = COOLDOWN_CANDLES * int(TIMEFRAME) * 60
    cutoff_ms = int((now - cooldown_seconds) * 1000)   # convert to milliseconds

    try:
        resp = session.get_closed_pnl(
            category = "linear",
            symbol   = symbol,
            limit    = 1,
        )
        records = resp["result"]["list"]
        if records:
            last_pnl       = float(records[0]["closedPnl"])
            last_trade_time = int(records[0]["updatedTime"])  # milliseconds

            # Ignore losses older than the cooldown window
            if last_trade_time < cutoff_ms:
                return False

            if last_pnl < 0:
                cooldown_until[symbol] = now + cooldown_seconds
                log.info(
                    f"[{symbol}] 🔴 Recent loss ({last_pnl:.4f} USDT) — "
                    f"cooldown for {COOLDOWN_CANDLES} candles ({cooldown_seconds}s)"
                )
                return True

    except Exception as exc:
        log.warning(f"[{symbol}] Could not fetch closed PnL: {exc}")

    return False


def check_daily_loss_limit() -> bool:
    """
    Returns True if the daily loss limit has been hit and trading should halt.
    Resets automatically at the start of a new calendar day.
    """
    from datetime import date
    today = str(date.today())

    # New day — reset state
    if daily_state["date"] != today:
        balance = get_available_balance()
        daily_state["date"]          = today
        daily_state["start_balance"] = balance
        daily_state["halted"]        = False
        log.info(f"📅 New trading day | Starting balance: {balance:.2f} USDT")
        return False

    # Already halted today
    if daily_state["halted"]:
        log.warning(
            f"🚫 Daily loss limit already hit today — "
            f"trading halted until tomorrow"
        )
        return True

    # Check current loss against starting balance
    try:
        current_balance = get_available_balance()
        start           = daily_state["start_balance"]
        loss_pct        = (start - current_balance) / start

        if loss_pct >= MAX_DAILY_LOSS_PCT:
            daily_state["halted"] = True
            log.warning(
                f"🚫 DAILY LOSS LIMIT HIT — "
                f"Lost {loss_pct*100:.2f}% today "
                f"(${start - current_balance:.2f} USDT) — "
                f"no more trades until tomorrow"
            )
            return True
    except Exception as exc:
        log.warning(f"Could not check daily loss: {exc}")

    return False


# ════════════════════════════════════════════════════════════
#  Trade execution
# ════════════════════════════════════════════════════════════

def calculate_qty(entry_price: float, info: dict, balance: float) -> float | None:
    """
    Compute order quantity from BALANCE_PCT of available balance.
    Returns None if the resulting qty is below the symbol minimum.
    """

    # Guard: skip if balance is too low to safely trade
    if balance < MIN_BALANCE_USDT:
        log.warning(f"Balance {balance:.2f} USDT is below minimum {MIN_BALANCE_USDT} USDT — skipping")
        return None

    usdt_size = balance * BALANCE_PCT
    usdt_size = min(usdt_size, MAX_TRADE_USDT)

    raw_qty   = (usdt_size * LEVERAGE) / entry_price
    qty       = round_to_step(raw_qty, info["qty_step"])

    if qty < info["min_qty"]:
        log.warning(
            f"Calculated qty {qty} < min qty {info['min_qty']}. "
            f"Increase BALANCE_PCT or account balance."
        )
        return None
    return qty


def place_entry_with_tp(symbol: str, signal: str, entry_price: float, ma28: float, balance: float) -> None:
    """
    Step 1 — Market entry order.
    Step 2 — Confirm position registered + get actual fill price.
    Step 3 — Reduce-only limit TP at actual_entry ± TP_DISTANCE.
    Step 4 — Reduce-only stop-market SL at MA28 (dynamic, structure-based).

    LONG  SL sits below entry at MA28 — invalidated if price breaks trend anchor.
    SHORT SL sits above entry at MA28 — invalidated if price reclaims trend anchor.
    """
    info = get_instrument_info(symbol)
    qty  = calculate_qty(entry_price, info, balance)
    if qty is None:
        return

    tp_dist = TP_DISTANCE.get(symbol, DEFAULT_TP_DISTANCE)
    side    = "Buy"  if signal == "LONG" else "Sell"
    tp_side = "Sell" if signal == "LONG" else "Buy"
    sl_price = round_to_step(ma28, info["tick_size"])

    # Safety check — SL must be on the correct side of entry before we even enter
    if signal == "LONG" and sl_price >= entry_price:
        log.warning(
            f"[{symbol}] MA28 SL ({sl_price}) is above entry ({entry_price}) "
            f"for LONG — skipping trade"
        )
        return
    if signal == "SHORT" and sl_price <= entry_price:
        log.warning(
            f"[{symbol}] MA28 SL ({sl_price}) is below entry ({entry_price}) "
            f"for SHORT — skipping trade"
        )
        return

    try:
        # ── Step 1: Market entry ─────────────────────────────
        entry_resp = session.place_order(
            category    = "linear",
            symbol      = symbol,
            side        = side,
            orderType   = "Market",
            qty         = str(qty),
            timeInForce = "GTC",
            reduceOnly  = False,
        )
        order_id = entry_resp["result"].get("orderId", "N/A")
        log.info(
            f"[{symbol}] ✅ {signal} ENTRY  | "
            f"Qty: {qty} | Expected ≈ {entry_price:.4f} | OrderID: {order_id}"
        )

        # ── Step 2: Confirm position + get actual fill price ─
        time.sleep(2)
        pos = get_open_position(symbol)
        if not pos:
            log.warning(
                f"[{symbol}] Position didn't register after entry — "
                f"TP/SL not placed. Check exchange manually."
            )
            return

        actual_entry = float(pos.get("avgPrice", entry_price))
        log.info(f"[{symbol}] 📍 Actual fill price: {actual_entry:.4f}")

        # Recalculate TP from actual fill price (not expected price)
        if signal == "LONG":
            tp_price = round_to_step(actual_entry + tp_dist, info["tick_size"])
        else:
            tp_price = round_to_step(actual_entry - tp_dist, info["tick_size"])

        # ── Step 3: Limit TP (reduce-only) ───────────────────
        tp_resp = session.place_order(
            category    = "linear",
            symbol      = symbol,
            side        = tp_side,
            orderType   = "Limit",
            qty         = str(qty),
            price       = str(tp_price),
            timeInForce = "GTC",
            reduceOnly  = True,
        )
        tp_id = tp_resp["result"].get("orderId", "N/A")
        log.info(
            f"[{symbol}] 🎯 TP ORDER  | Price: {tp_price:.4f} | OrderID: {tp_id}"
        )

        # ── Step 4: Stop-market SL at MA28 (reduce-only) ─────
        trigger_direction = 2 if signal == "LONG" else 1

        sl_resp = session.place_order(
            category         = "linear",
            symbol           = symbol,
            side             = tp_side,
            orderType        = "Market",
            qty              = str(qty),
            triggerPrice     = str(sl_price),
            triggerBy        = "LastPrice",
            triggerDirection = trigger_direction,   # ← add this
            orderFilter      = "StopOrder",
            timeInForce      = "GTC",
            reduceOnly       = True,
        )
        sl_id = sl_resp["result"].get("orderId", "N/A")
        log.info(
            f"[{symbol}] 🛡  SL ORDER  | MA28: {sl_price:.4f} | OrderID: {sl_id}"
        )
    except Exception as exc:
        log.error(f"[{symbol}] Order placement failed: {exc}")

# ════════════════════════════════════════════════════════════
#  Main loop
# ════════════════════════════════════════════════════════════

def scan_symbol(symbol: str) -> None:
    """Run one full analysis + trade cycle for a single symbol."""
    # ── Daily loss limit ─────────────────────────────────────
    if check_daily_loss_limit():
        return

    # ── Open position check ──────────────────────────────────
    pos = get_open_position(symbol)
    if pos:
        side  = pos.get("side", "?")
        size  = pos.get("size", "?")
        entry = pos.get("avgPrice", "?")
        log.info(f"[{symbol}] Position open ({side} {size} @ {entry}) — skipping")
        return

    # ── Cooldown check ───────────────────────────────────────
    if check_cooldown(symbol):
        return

    # Fetch balance once and reuse throughout this cycle
    balance = get_available_balance()
    log.info(f"[{symbol}] 💰 Balance: {balance:.2f} USDT")

    # ── Higher timeframe trend filter ────────────────────────
    df_htf    = fetch_klines(symbol, interval=HTF_TIMEFRAME)
    df_htf    = calculate_indicators(df_htf)
    htf_trend = get_htf_trend(df_htf)

    if htf_trend is None:
        log.info(f"[{symbol}] ⏭  HTF ({HTF_TIMEFRAME}min) trend unclear — skipping")
        return

    log.info(f"[{symbol}] 📊 HTF trend: {htf_trend}")

    # ── 5min signal ──────────────────────────────────────────
    df     = fetch_klines(symbol, interval=TIMEFRAME)
    df     = calculate_indicators(df)
    signal = get_signal(df)

    last_close = df.iloc[-2]["close"]

    if signal:
        # Only enter if 5min signal matches HTF trend direction
        if signal != htf_trend:
            log.info(
                f"[{symbol}] ⛔ Signal {signal} blocked — "
                f"conflicts with HTF trend ({htf_trend})"
            )
            return

        ma28_val = df.iloc[-2]["ma28"]
        log.info(f"[{symbol}] 🔔 Signal: {signal}  |  Last close: {last_close:.4f}  |  MA28: {ma28_val:.4f}  |  HTF confirmed ✅")
        place_entry_with_tp(symbol, signal, last_close, ma28_val, balance)
    else:
        ma7  = df.iloc[-2]["ma7"]
        ma14 = df.iloc[-2]["ma14"]
        ma28 = df.iloc[-2]["ma28"]
        mf   = df.iloc[-2]["mavol_fast"]
        ms   = df.iloc[-2]["mavol_slow"]
        log.info(
            f"[{symbol}] No signal  |  "
            f"MA7={ma7:.2f} MA14={ma14:.2f} MA28={ma28:.2f}  |  "
            f"MAVOL_F={mf:.1f} MAVOL_S={ms:.1f}"
        )

def run() -> None:
    """Entry point — initialise leverage then run the scan loop."""
    log.info("=" * 60)
    log.info("  Bybit USDT Perp Bot starting up")
    log.info(f"  Mode    : {'TESTNET ⚠️' if TESTNET else 'LIVE 🔴'}")
    log.info(f"  Symbols : {', '.join(SYMBOLS)}")
    log.info(f"  Leverage: {LEVERAGE}x  |  Bal%: {BALANCE_PCT*100:.1f}%")
    log.info("=" * 60)

    # Set leverage for every symbol once at startup
    for symbol in SYMBOLS:
        set_leverage(symbol)

    # ── Main scan loop ───────────────────────────────────────
    while True:
        log.info("─" * 60)
        for symbol in SYMBOLS:
            try:
                scan_symbol(symbol)
            except Exception as exc:
                log.error(f"[{symbol}] Unexpected error: {exc}", exc_info=True)

        log.info(f"Sleeping {LOOP_INTERVAL}s until next scan…")
        time.sleep(LOOP_INTERVAL)


if __name__ == "__main__":
    run()