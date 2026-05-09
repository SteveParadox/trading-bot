# ============================================================
#  Bybit Futures Bot — Configuration
#  API credentials are loaded from .env file (never commit .env)
# ============================================================

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Credentials (from .env file) ---
API_KEY    = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")
TESTNET    = False          # ← Set False only when ready for live trading

# --- Symbols to trade ---
SYMBOLS = [
    "TONUSDT",
    "DYMUSDT",
]

MIN_BALANCE_USDT = 3.0   # Bot won't trade if available balance drops below this

HTF_TIMEFRAME = "60"    # Higher timeframe to confirm trend (60min)

# --- Timeframe ---
TIMEFRAME = "5"            # 5-minute candles (Bybit interval string)
CANDLE_LIMIT = 60          # How many candles to fetch per cycle (must be > 28)

# --- Leverage (applied to all symbols) ---
LEVERAGE = 2              

# --- Position Sizing ---
BALANCE_PCT = 0.5         # Use 5% of available USDT balance per trade

# --- Take-Profit distance per symbol (in USDT price units) ---
# The bot places a limit TP order this many dollars above (long) or
# below (short) the entry price.  Price touches it → profit taken.
TP_DISTANCE = {
#    "NOTUSDT": 0.0000052,       # e.g. entry 60000 → TP 60050 (long)
    "TONUSDT": 0.0035,
    "DYMUSDT": 0.00035,
#    "LABUSDT": 0.0035,
}
DEFAULT_TP_DISTANCE = 1.0  # Fallback for any symbol not listed above

# --- Indicator Periods ---
MA_PERIODS   = [7, 14, 28] # Simple Moving Averages used for trend
MAVOL_FAST   = 9           # Fast Volume-MA period  (bullish when fast > slow)
MAVOL_SLOW   = 18          # Slow Volume-MA period

# --- Bot Loop ---
LOOP_INTERVAL = 60         # Seconds between each scan cycle

COOLDOWN_CANDLES     = 3      # Candles to wait after a loss (3 × 5min = 15min)
MAX_DAILY_LOSS_PCT   = 0.004  # Stop trading today if account drops 4%

MAX_TRADE_USDT = 500.0 