"""Configuration for the Bybit USDT perpetual futures bot.

Most operational toggles can be overridden from .env so live trading is never
enabled accidentally by editing code alone.
"""

from __future__ import annotations

import os
from typing import Iterable

from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _get_csv(name: str, default: Iterable[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return list(default)
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


# API credentials are read from .env. Keep .env out of source control.
API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")

# Start safe by default. Set TESTNET=false and DRY_RUN=false in .env only after
# you have watched the bot behave correctly with small size.
TESTNET = _get_bool("TESTNET", True)
DRY_RUN = _get_bool("DRY_RUN", True)

# Symbols and candles
SYMBOLS = _get_csv("SYMBOLS", ["SAGAUSDT", "BUSDT"])
TIMEFRAME = os.getenv("TIMEFRAME", "5")
HTF_TIMEFRAME = os.getenv("HTF_TIMEFRAME", "60")
CANDLE_LIMIT = _get_int("CANDLE_LIMIT", 150)
LOOP_INTERVAL = _get_int("LOOP_INTERVAL", 60)

# Leverage and capital protection
LEVERAGE = _get_int("LEVERAGE", 2)
MIN_BALANCE_USDT = _get_float("MIN_BALANCE_USDT", 10.0)
PAPER_BALANCE_USDT = _get_float("PAPER_BALANCE_USDT", 100.0)
RISK_PER_TRADE_PCT = _get_float("RISK_PER_TRADE_PCT", 0.005)  # 0.5% equity
MAX_TRADE_USDT = _get_float("MAX_TRADE_USDT", 50.0)  # max position notional
MAX_OPEN_POSITIONS = _get_int("MAX_OPEN_POSITIONS", 1)
MAX_DAILY_LOSS_PCT = _get_float("MAX_DAILY_LOSS_PCT", 0.02)  # 2% equity drawdown
MAX_CONSECUTIVE_LOSSES = _get_int("MAX_CONSECUTIVE_LOSSES", 3)
POST_TRADE_COOLDOWN = _get_int("POST_TRADE_COOLDOWN", 600)
COOLDOWN_CANDLES = _get_int("COOLDOWN_CANDLES", 3)

# Market-quality filters. These reduce low-quality entries; they do not make a
# strategy profitable by themselves.
MAX_SPREAD_BPS = _get_float("MAX_SPREAD_BPS", 12.0)
MAX_ENTRY_DEVIATION_PCT = _get_float("MAX_ENTRY_DEVIATION_PCT", 0.003)
MIN_RISK_REWARD = _get_float("MIN_RISK_REWARD", 1.5)
MIN_STOP_DISTANCE_PCT = _get_float("MIN_STOP_DISTANCE_PCT", 0.0015)
MAX_STOP_DISTANCE_PCT = _get_float("MAX_STOP_DISTANCE_PCT", 0.03)
MIN_ATR_PCT = _get_float("MIN_ATR_PCT", 0.001)
MAX_ATR_PCT = _get_float("MAX_ATR_PCT", 0.08)

# Feature toggles. Defaults are deliberately tradeable; the two most restrictive
# confirmations start disabled until the operator has observed signal frequency.
PARTIAL_TP_ENABLED = _get_bool("PARTIAL_TP_ENABLED", True)
TRAILING_STOP_ENABLED = _get_bool("TRAILING_STOP_ENABLED", True)
RSI_DIVERGENCE_FILTER = _get_bool("RSI_DIVERGENCE_FILTER", True)
HTF_REQUIRE_MOMENTUM_CANDLE = _get_bool("HTF_REQUIRE_MOMENTUM_CANDLE", False)
VOLUME_SPIKE_FILTER = _get_bool("VOLUME_SPIKE_FILTER", True)
FUNDING_RATE_FILTER = _get_bool("FUNDING_RATE_FILTER", True)
SYMBOL_RANKING_ENABLED = _get_bool("SYMBOL_RANKING_ENABLED", True)
FRESH_CROSS_FILTER = _get_bool("FRESH_CROSS_FILTER", False)

# Partial take-profit, breakeven, and trailing-stop management. This state is
# tracked in memory by bot.py; after a restart, check open orders manually.
TP1_QTY_PCT = _get_float("TP1_QTY_PCT", 0.50)
TP2_MULTIPLIER = _get_float("TP2_MULTIPLIER", 2.0)
BREAKEVEN_BUFFER = _get_float("BREAKEVEN_BUFFER", 0.0001)
TRAIL_ATR_MULTIPLIER = _get_float("TRAIL_ATR_MULTIPLIER", 1.5)

# Signal filters and confirmations.
RSI_PERIOD = _get_int("RSI_PERIOD", 14)
RSI_DIVERGENCE_LOOKBACK = _get_int("RSI_DIVERGENCE_LOOKBACK", 5)
HTF_ADX_MIN = _get_float("HTF_ADX_MIN", 25.0)
# 1.2 keeps the spike filter from starving lower-liquidity symbols by default;
# operators who want stricter conviction can raise it toward 1.5.
VOLUME_SPIKE_MULTIPLIER = _get_float("VOLUME_SPIKE_MULTIPLIER", 1.2)
CROSS_LOOKBACK = _get_int("CROSS_LOOKBACK", 5)

# Signal ranking. CROSS_LOOKBACK=5 means a fresh MA cross must have happened in
# the last 25 minutes on a 5-minute chart; increase to 8-10 if it is too strict.
MAX_SIGNALS_PER_CYCLE = _get_int("MAX_SIGNALS_PER_CYCLE", 2)
SIGNAL_SCORE_ADX_FLOOR = _get_float("SIGNAL_SCORE_ADX_FLOOR", 20.0)
SIGNAL_SCORE_ADX_CEILING = _get_float("SIGNAL_SCORE_ADX_CEILING", 60.0)
SIGNAL_SCORE_VOLUME_FLOOR = _get_float("SIGNAL_SCORE_VOLUME_FLOOR", 1.0)
SIGNAL_SCORE_VOLUME_CEILING = _get_float("SIGNAL_SCORE_VOLUME_CEILING", 3.0)
SIGNAL_SCORE_MA_SEP_FULL_PCT = _get_float("SIGNAL_SCORE_MA_SEP_FULL_PCT", 0.02)
SIGNAL_SCORE_RR_FLOOR = _get_float("SIGNAL_SCORE_RR_FLOOR", 1.0)
SIGNAL_SCORE_RR_CEILING = _get_float("SIGNAL_SCORE_RR_CEILING", 3.0)

# Funding extremes can mark crowded positioning. If the API value is
# unavailable, bot.py logs a warning and lets the trade continue.
FUNDING_RATE_MAX_LONG = _get_float("FUNDING_RATE_MAX_LONG", 0.0005)
FUNDING_RATE_MIN_SHORT = _get_float("FUNDING_RATE_MIN_SHORT", -0.0005)

# Exit model.
# - STOP_MODE="ma": stop at MA28, skip setups where that stop is too tight/far.
# - STOP_MODE="atr": stop at ATR_SL_MULTIPLIER x ATR from entry.
# - TP_MODE="rr": take-profit is MIN_RISK_REWARD x risk distance.
# - TP_MODE="fixed": use TP_DISTANCE below.
STOP_MODE = os.getenv("STOP_MODE", "ma").strip().lower()
TP_MODE = os.getenv("TP_MODE", "rr").strip().lower()
ATR_SL_MULTIPLIER = _get_float("ATR_SL_MULTIPLIER", 1.5)

# Fixed take-profit distances are used only when TP_MODE="fixed".
TP_DISTANCE = {
    "SAGAUSDT": 0.00035,
    "BUSDT": 0.00035,
}
DEFAULT_TP_DISTANCE = _get_float("DEFAULT_TP_DISTANCE", 1.0)

# Indicator periods
MA_PERIODS = [7, 14, 28]
MAVOL_FAST = _get_int("MAVOL_FAST", 9)
MAVOL_SLOW = _get_int("MAVOL_SLOW", 18)
VOLUME_RATIO_MIN = _get_float("VOLUME_RATIO_MIN", 1.05)
ADX_PERIOD = _get_int("ADX_PERIOD", 14)
ADX_MIN = _get_float("ADX_MIN", 22.0)
ATR_PERIOD = _get_int("ATR_PERIOD", 14)

# Execution hardening
API_RETRIES = _get_int("API_RETRIES", 3)
API_RETRY_DELAY = _get_float("API_RETRY_DELAY", 1.0)
POSITION_CONFIRM_RETRIES = _get_int("POSITION_CONFIRM_RETRIES", 5)
POSITION_CONFIRM_DELAY = _get_float("POSITION_CONFIRM_DELAY", 1.0)
TRAIL_UPDATE_MIN_SECONDS = _get_int("TRAIL_UPDATE_MIN_SECONDS", 30)
TRAIL_MIN_MOVE_TICKS = _get_int("TRAIL_MIN_MOVE_TICKS", 2)
ORDER_LINK_PREFIX = os.getenv("ORDER_LINK_PREFIX", "riskbot")
EMERGENCY_CLOSE_ON_PROTECTION_FAILURE = _get_bool(
    "EMERGENCY_CLOSE_ON_PROTECTION_FAILURE",
    True,
)
