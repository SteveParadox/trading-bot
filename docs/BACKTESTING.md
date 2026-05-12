# Backtesting Framework

This framework is designed to evaluate the Bybit USDT perpetual futures bot
before live deployment. It prioritizes realistic simulation over inflated
performance.

## Project Structure

```text
backtester/
  analytics.py        Performance, risk, trade, and execution reports
  cli.py              `python -m backtester.cli --config ...`
  config.py           Dataclass configuration models
  data.py             CSV/Parquet loader, validation, gaps, resampling
  engine.py           Candle-by-candle orchestration loop
  execution.py        Bybit-like futures execution simulator
  models.py           Orders, fills, positions, trades, snapshots
  optimization.py     Grid/random/walk-forward/Monte Carlo utilities
  risk.py             Sizing, exposure, drawdown, heat, liquidation estimates
  strategy.py         Strategy interface and live `indicators.py` adapter
  visualization.py    Exportable charts
configs/
  backtest_config.json
examples/
  generate_sample_data.py
  run_backtest.py
  optimize.py
tests/
  test_*.py
```

## Implementation Plan

1. Load clean OHLCV data for every symbol and timeframe.
2. Normalize timestamps to UTC, validate OHLC relationships, detect gaps, and
   optionally resample lower timeframes into higher confirmations.
3. Enrich candles with the existing `indicators.py` logic.
4. Process historical candles in chronological order.
5. Execute pending orders through the current candle before generating new
   signals.
6. Generate strategy signals only from candles that are closed by the decision
   timestamp.
7. Size trades with risk budgets, leverage caps, exposure caps, portfolio heat,
   drawdown protection, and daily loss protection.
8. Submit market entries with configurable candle latency.
9. After an entry fill, place reduce-only stop-loss and take-profit orders using
   the actual simulated fill price.
10. Export equity, trades, fills, metrics, and optional charts.

## Data Format

CSV and Parquet inputs must include:

```text
timestamp, open, high, low, close, volume
```

Optional:

```text
turnover
```

The default config expects:

```text
data/SAGAUSDT_5m.csv
data/BUSDT_5m.csv
```

It derives `1h` and `4h` candles from `5m` data.

## Example Run

```bash
python examples/generate_sample_data.py
python -m backtester.cli --config configs/backtest_config.json
```

Outputs:

```text
reports/latest/report.json
reports/latest/equity_curve.csv
reports/latest/trades.csv
reports/latest/fills.csv
reports/latest/monthly_returns.csv
```

## Realism Controls

The execution simulator supports:

* market, limit, stop-market, and stop-limit orders
* reduce-only behavior
* long and short positions
* one-way and hedge position modes
* configurable taker/maker fees
* spread and slippage models
* order latency
* partial fills
* missed fills
* conservative intrabar stop/target priority

Market entries are delayed to the next candle by default. If a stop-loss and
take-profit are both touched in the same candle, stop orders are processed first
when `conservative_intrabar_priority` is enabled.

## Optimization Workflow

```bash
python examples/optimize.py
```

Supported workflows:

* grid search
* random search
* walk-forward optimization
* out-of-sample segment testing
* Monte Carlo trade-path bootstrapping
* parameter sensitivity analysis

Use train/test or walk-forward reports as the primary decision artifact. A
single in-sample best result is not enough evidence for live deployment.

## Future Improvements

* Native Bybit historical downloader with rate-limit aware paging.
* Funding-rate and borrow-cost simulation.
* Order-book replay for spread and queue-position modeling.
* Websocket replay for tick-by-tick forward testing.
* Database-backed candles, orders, and experiment tracking.
* Distributed optimization workers.
* Paper-trading adapter that reuses the same strategy/risk interfaces.
* ML feature store integration with strict timestamp joins.
