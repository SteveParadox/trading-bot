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
ORDER_LINK_PREFIX = os.getenv("ORDER_LINK_PREFIX", "riskbot")
EMERGENCY_CLOSE_ON_PROTECTION_FAILURE = _get_bool(
    "EMERGENCY_CLOSE_ON_PROTECTION_FAILURE",
    True,
)
