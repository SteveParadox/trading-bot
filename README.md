# Bybit USDT Perp Trading Bot

Python bot for Bybit linear USDT perpetuals. It scans for MA/volume/ADX trend
signals, confirms them on a higher timeframe, and places market entries with
reduce-only stop-loss and take-profit orders.

This code cannot guarantee profit. The changes here are meant to reduce
avoidable losses: smaller risk-based sizing, bad-market filters, protective
orders, and safer defaults.

## Main Features

| Area | Behavior |
| --- | --- |
| Signal | MA7/MA14/MA28 trend stack, volume ratio, ADX/DI confirmation |
| Higher timeframe | Blocks trades that disagree with `HTF_TIMEFRAME` trend |
| Position sizing | Uses `RISK_PER_TRADE_PCT` of equity based on stop distance |
| Take-profit | Defaults to risk/reward mode: `MIN_RISK_REWARD` x stop distance |
| Stop-loss | MA28 stop by default, or ATR stop with `STOP_MODE=atr` |
| Filters | Spread, ATR range, price-chase guard, cooldown after trades/losses |
| Safety | Dry-run default, testnet default, max daily loss, max open positions |
| Execution | API retries, exchange tick/qty rounding, stale reduce-only cleanup |
| Failure handling | If protection cannot be placed, the bot can emergency-close |

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env`, then fill in your values. New installs default
to `TESTNET=true` and `DRY_RUN=true`.

```env
API_KEY=your_api_key_here
API_SECRET=your_api_secret_here
TESTNET=true
DRY_RUN=true
SYMBOLS=SAGAUSDT,BUSDT
```

Run the bot:

```bash
python bot.py
```

## Safer Live Checklist

Before setting `DRY_RUN=false`, watch several dry-run sessions and confirm the
logged entry, stop, take-profit, quantity, and risk numbers make sense.

Suggested live-start settings:

```env
TESTNET=true
DRY_RUN=false
RISK_PER_TRADE_PCT=0.0025
MAX_TRADE_USDT=10
MAX_OPEN_POSITIONS=1
MAX_DAILY_LOSS_PCT=0.01
```

Move to `TESTNET=false` only after testnet behavior is boring and predictable.

## Important Settings

| Setting | Default | Notes |
| --- | ---: | --- |
| `DRY_RUN` | `true` | Logs orders without placing them |
| `TESTNET` | `true` | Uses Bybit testnet endpoint |
| `RISK_PER_TRADE_PCT` | `0.005` | 0.5% equity risk before caps |
| `MAX_TRADE_USDT` | `50` | Max notional size per position |
| `MAX_DAILY_LOSS_PCT` | `0.02` | Halt after 2% daily equity drawdown |
| `MAX_SPREAD_BPS` | `12` | Skip wide-spread markets |
| `MIN_RISK_REWARD` | `1.5` | Minimum reward/risk ratio |
| `STOP_MODE` | `ma` | Use `ma` or `atr` |
| `TP_MODE` | `rr` | Use `rr` or `fixed` |

## Files

```text
bot.py          Main scan loop, account checks, orders, risk controls
config.py       Environment-driven settings and defaults
indicators.py   MA, MAVOL, ADX, ATR, signal logic
backtester/     Historical data, execution, risk, analytics, optimization
configs/        Backtest configuration examples
examples/       Backtest and optimization entry points
tests/          Unit tests for the backtester
requirements.txt
```

## Backtesting

The repository now includes a modular Bybit USDT perpetual futures backtester
that reuses `indicators.py` while simulating delayed market entries, fees,
spread/slippage, partial and missed fills, reduce-only exits, leverage, sizing,
portfolio risk, and performance analytics.

Generate deterministic smoke-test data:

```bash
python examples/generate_sample_data.py
```

Run a backtest:

```bash
python -m backtester.cli --config configs/backtest_config.json
```

Read the full guide in `docs/BACKTESTING.md`.

## Risk Warning

Trading perpetual futures can lose money quickly, especially with leverage.
Start in dry-run and testnet, keep risk small, and review exchange orders
manually until you trust the automation.
