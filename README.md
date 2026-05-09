# Bybit USDT Perp Trading Bot

A Python scalping bot for Bybit Futures that trades based on MA trend alignment
and dual Volume-MA confirmation, with intra-candle limit Take-Profit orders.

---

## Strategy Summary

| Component       | Detail                                                             |
|-----------------|--------------------------------------------------------------------|
| Timeframe       | 5-minute candles                                                   |
| Trend filter    | MA7 / MA14 / MA28 — all three must stack in order                 |
| Volume confirm  | MAVOL(9) vs MAVOL(18) — fast must be above/below slow             |
| Entry           | Market order on last *closed* candle's signal                      |
| Take-Profit     | Limit order at entry ± TP_DISTANCE (fills on price touch, no close needed) |
| Stop-Loss       | None                                                               |
| Leverage        | 10x                                                                |
| Position size   | % of available USDT balance (configurable)                         |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure the bot

Edit `config.py`:

- Add your **API key** and **API secret** from Bybit
- Set `TESTNET = True` while testing, `False` for live
- Adjust `SYMBOLS`, `BALANCE_PCT`, and `TP_DISTANCE` per symbol

```python
API_KEY    = "abc123..."
API_SECRET = "xyz789..."
TESTNET    = True   # ← Start here

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

BALANCE_PCT = 0.05   # 5% of balance per trade

TP_DISTANCE = {
    "BTCUSDT": 50.0,   # Take profit $50 above/below entry
    "ETHUSDT": 3.0,
    "SOLUSDT": 0.30,
}
```

### 3. Run the bot

```bash
python bot.py
```

---

## File Structure

```
bybit_bot/
├── bot.py            # Main loop, order execution
├── indicators.py     # MA, MAVOL calculation + signal logic
├── config.py         # All settings — edit this
├── requirements.txt
└── README.md
```

---

## How the "Second Price" TP works

When the bot enters a trade, it immediately attaches a **limit Take-Profit order**
at `entry ± TP_DISTANCE`.  Bybit's engine monitors this order in real-time.
The moment price **touches** that level (even mid-candle), the TP limit order
fills and the position closes at profit.  The candle does not need to close
at or beyond the TP level.

---

## Risk Warning

> This bot has **no Stop-Loss**.  A strong adverse move will remain open until
> the TP is hit on a reversal.  Use small `BALANCE_PCT` values and always test
> on Testnet first.  Never trade with money you cannot afford to lose.
