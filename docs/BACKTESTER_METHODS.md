# Backtester Methods Documentation

A detailed guide to using the backtester's core methods, classes, and workflows.

## Table of Contents

1. [Overview](#overview)
2. [Core Components](#core-components)
3. [DataPortal - Data Management](#dataportal---data-management)
4. [BacktestEngine - Main Orchestrator](#backtestengine---main-orchestrator)
5. [SimulatedExchange - Execution Simulator](#simulatedexchange---execution-simulator)
6. [RiskManager - Trade Risk Controls](#riskmanager---trade-risk-controls)
7. [PerformanceAnalyzer - Metrics & Reporting](#performanceanalyzer---metrics--reporting)
8. [Strategy - Signal Generation](#strategy---signal-generation)
9. [OptimizationRunner - Parameter Optimization](#optimizationrunner---parameter-optimization)
10. [BacktestResult - Results Export](#backtestresult---results-export)
11. [Typical Workflows](#typical-workflows)
12. [Configuration](#configuration)

---

## Overview

The backtester simulates trading strategies against historical OHLCV data with realistic order execution, portfolio accounting, and risk management. Key principles:

- **Realistic simulation**: Market orders execute on future candles, slippage is applied
- **No forward-leakage**: Signals use only closed candles
- **Order execution**: Supports market, limit, stop-market, and stop-limit orders
- **Risk controls**: Position sizing, leverage caps, drawdown protection, portfolio heat limits
- **Multi-symbol**: Simultaneous tracking of multiple trading pairs

---

## Core Components

The backtester consists of these main components:

| Component | Purpose |
|-----------|---------|
| `DataPortal` | Load, validate, and manage OHLCV data |
| `BacktestEngine` | Orchestrate candle-by-candle simulation |
| `SimulatedExchange` | Realistic order execution and portfolio accounting |
| `RiskManager` | Trade admission, sizing, and liquidation estimates |
| `IndicatorSignalStrategy` | Signal generation from indicators.py |
| `PerformanceAnalyzer` | Calculate metrics, profitability, risk statistics |
| `OptimizationRunner` | Grid search, random search, walk-forward analysis |
| `BacktestResult` | Aggregated results and export functionality |

---

## DataPortal - Data Management

`DataPortal` loads and manages multi-symbol, multi-timeframe OHLCV data.

### Creating a DataPortal

```python
from backtester import BacktestConfig, DataPortal

# Load from config
config = BacktestConfig.from_json("configs/backtest_config.json")
data = DataPortal.from_config(config.data)

# Or manually
from backtester.config import DataConfig
data_config = DataConfig(
    symbols=["SAGAUSDT", "NEARUSDT"],
    timeframes=["5m", "1h", "4h"],
    base_timeframe="5m",
    data_dir="data/",
    resample_from="5m"  # Derive 1h/4h from 5m
)
data = DataPortal.from_config(data_config)
```

### Key Methods

#### `load()`
Loads and validates OHLCV data, optionally resampling lower timeframes into higher ones.

```python
data.load()
```

#### `get_frame(symbol: str, timeframe: str) -> pd.DataFrame`
Retrieve a single symbol's OHLCV data at a specific timeframe.

```python
df = data.get_frame("SAGAUSDT", "1h")
# Returns DataFrame with columns: open, high, low, close, volume
```

#### `set_frame(symbol: str, timeframe: str, frame: pd.DataFrame)`
Set or update a dataframe for a symbol/timeframe pair.

```python
data.set_frame("SAGAUSDT", "1h", df_hourly)
```

#### `history(symbol: str, timeframe: str, up_to: pd.Timestamp, n: int = None) -> pd.DataFrame`
Get historical data up to a specific timestamp, optionally limited to the last `n` candles.

```python
# Get last 50 candles before decision time
hist = data.history("SAGAUSDT", "1h", up_to=decision_time, n=50)

# Get all historical data up to timestamp
hist = data.history("SAGAUSDT", "1h", up_to=decision_time)
```

#### `symbols() -> list[str]`
Get all loaded symbols.

```python
symbols = data.symbols()
# Returns: ["SAGAUSDT", "NEARUSDT"]
```

#### `iter_symbol_candles(timeframe: str)`
Iterate through candles chronologically for all symbols at a given timeframe. Returns `(bar_open_time, {symbol: candle_row})`.

```python
for bar_open, candles in data.iter_symbol_candles("5m"):
    for symbol, candle in candles.items():
        open_price = candle["open"]
        close_price = candle["close"]
```

#### `apply_to_frames(func: Callable, timeframes: list[str])`
Apply a function to all frames at specified timeframes (used for indicator calculation).

```python
from indicators import calculate_indicators

data.apply_to_frames(calculate_indicators, ["5m", "1h"])
```

#### `validate() -> list[str]`
Validate all loaded data and return a list of validation messages.

```python
messages = data.validate()
for msg in messages:
    print(msg)
```

#### `detect_gaps(symbol: str, timeframe: str, ignore_market_hours: bool = True) -> GapReport`
Detect missing candles in a data series.

```python
report = data.detect_gaps("SAGAUSDT", "5m")
if report.has_gaps:
    print(f"Missing {report.missing_candles} candles")
    print(f"Largest gap: {report.largest_gap_candles} candles")
```

---

## BacktestEngine - Main Orchestrator

`BacktestEngine` orchestrates the candle-by-candle simulation loop.

### Creating an Engine

```python
from backtester import BacktestEngine, BacktestConfig, DataPortal

config = BacktestConfig.from_json("configs/backtest_config.json")
data = DataPortal.from_config(config.data)

engine = BacktestEngine(
    config=config,
    data=data,
    strategy=None,  # Uses IndicatorSignalStrategy by default
    exchange=None,  # Uses SimulatedExchange by default
    risk_manager=None  # Uses RiskManager by default
)
```

### Key Methods

#### `run() -> BacktestResult`
Execute the full backtest simulation.

**Flow:**
1. Calls `prepare()` to initialize data
2. Iterates through candles chronologically
3. For each candle timestamp:
   - Executes pending orders through the current candle
   - Marks portfolio to market
   - Generates new strategy signals
   - Risk manager sizes and admits trades
4. Calculates final metrics
5. Optionally exports reports

```python
result = engine.run()
print(f"Net Profit: {result.metrics['profitability']['net_profit']:.2f}")
print(f"Max Drawdown: {result.metrics['risk']['max_drawdown']:.2%}")
```

Returns a `BacktestResult` object containing:
- `config`: The BacktestConfig used
- `snapshots`: List of `PortfolioSnapshot` (portfolio state at each bar)
- `fills`: List of all trade fills
- `trades`: List of completed trades with P&L
- `orders`: List of all orders
- `metrics`: Dictionary of performance metrics
- `execution_stats`: Statistics about order execution

#### `prepare()`
Initialize strategy data (indicator calculations) before running the backtest.

```python
engine.prepare()  # Calculates all indicators
result = engine.run()  # Runs with pre-computed indicators
```

This is automatically called by `run()`, but can be called separately if needed.

---

## SimulatedExchange - Execution Simulator

`SimulatedExchange` simulates realistic Bybit futures order execution.

### Overview

The simulated exchange handles:
- Market, limit, stop-market, stop-limit orders
- Long and short positions
- One-way and hedge position modes
- Reduce-only behavior
- Taker/maker fees and slippage
- Configurable order latency

### Key Methods

#### `process_candle(symbol, timestamp, candle, bar_index) -> list[Fill]`
Process a single candle for order fills (called internally by BacktestEngine).

```python
fills = exchange.process_candle(
    symbol="SAGAUSDT",
    timestamp=pd.Timestamp("2024-01-01 12:00:00"),
    candle={"open": 100, "high": 105, "low": 99, "close": 102, "volume": 1000},
    bar_index=100
)
```

Returns a list of `Fill` objects (trades that were executed).

#### `mark_to_market(timestamp)`
Update all open positions' unrealized P&L based on the latest close price.

```python
exchange.mark_to_market(pd.Timestamp("2024-01-01 12:00:00"))
equity = exchange.equity()  # Returns cash + unrealized P&L
```

#### `submit_market_entry(symbol, side, qty, timestamp, current_index, metadata)`
Submit a market entry order that will execute on the next candle (candle latency).

```python
exchange.submit_market_entry(
    symbol="SAGAUSDT",
    side=Side.LONG,  # Side.LONG or Side.SHORT
    qty=10.5,
    timestamp=pd.Timestamp("2024-01-01 12:00:00"),
    current_index=100,
    metadata={"signal_row": 99, "create_bracket": True}
)
```

#### `submit_reduce_only(symbol, side, order_type, qty, price, timestamp, current_index)`
Submit a reduce-only order (stop-loss or take-profit).

```python
exchange.submit_reduce_only(
    symbol="SAGAUSDT",
    side=Side.SHORT,  # Opposite of position side
    order_type=OrderType.STOP_MARKET,
    qty=10.5,
    price=95.0,  # Stop-loss trigger price
    timestamp=pd.Timestamp("2024-01-01 12:00:00"),
    current_index=100
)
```

#### Portfolio Status Methods

**`equity() -> float`**
Current account equity (cash + unrealized P&L).

```python
eq = exchange.equity()
```

**`cash -> float`**
Available cash (does not include unrealized P&L).

```python
cash = exchange.cash
```

**`realized_pnl -> float`**
Cumulative realized profit/loss from closed trades.

```python
rpnl = exchange.realized_pnl
```

**`positions: dict[str, Position]`**
Current open positions keyed by symbol.

```python
if "SAGAUSDT" in exchange.positions:
    pos = exchange.positions["SAGAUSDT"]
    print(f"Position size: {pos.size}, Entry price: {pos.entry_price}")
```

**`open_position_count() -> int`**
Number of currently open positions.

```python
count = exchange.open_position_count()
```

**`has_position(symbol, side=None) -> bool`**
Check if a position exists (optionally for a specific side).

```python
if exchange.has_position("SAGAUSDT", side=Side.LONG):
    print("Has long position")
```

**`gross_exposure() -> float`**
Total notional exposure across all positions.

```python
exp = exchange.gross_exposure()
print(f"Gross exposure: {exp:.2f} USDT")
```

**`symbol_exposure(symbol) -> float`**
Notional exposure for a specific symbol.

```python
exp = exchange.symbol_exposure("SAGAUSDT")
```

**`margin_used() -> float`**
Margin currently in use (contracts * entry_price / leverage).

```python
margin = exchange.margin_used()
```

**`portfolio_heat() -> float`**
Sum of risk amounts from all open positions.

```python
heat = exchange.portfolio_heat()
```

#### `orders: dict[str, Order]`
All orders (both filled and unfilled) keyed by order ID.

#### `fills: list[Fill]`
All fills throughout the backtest.

#### `trades: list[TradeRecord]`
All closed trades with entry/exit details and P&L.

#### `snapshots: list[PortfolioSnapshot]`
Portfolio state at each bar (equity, positions, margin, etc.).

---

## RiskManager - Trade Risk Controls

`RiskManager` evaluates trade intents and applies sizing, leverage, and portfolio constraints.

### Creating a RiskManager

```python
from backtester import BacktestConfig, RiskManager

config = BacktestConfig.from_json("configs/backtest_config.json")
risk_manager = RiskManager(config)
```

### Key Methods

#### `evaluate_intent(intent: SignalIntent, exchange, timestamp) -> RiskDecision`
Evaluate a strategy signal and return a sized order decision or rejection reason.

**Returns:** `RiskDecision` with fields:
- `allowed: bool` - Whether the trade was admitted
- `reason: str` - Rejection reason or "OK"
- `qty: float` - Sized quantity
- `risk_amount: float` - Dollar risk for the trade
- `exit_plan: ExitPlan` - Calculated stop-loss and take-profit levels
- `metadata: dict` - Additional context

```python
decision = risk_manager.evaluate_intent(intent, exchange, decision_time)
if not decision.allowed:
    print(f"Trade rejected: {decision.reason}")
else:
    print(f"Trade sized to {decision.qty} contracts")
    print(f"Stop-loss: {decision.exit_plan.stop_price}, TP: {decision.exit_plan.target_prices}")
```

**Rejection reasons include:**
- `max_open_positions` - Too many open trades
- `symbol_position_already_open` - Already have a position in this symbol
- `insufficient_margin` - Not enough margin available
- `max_leverage_exceeded` - Would exceed max leverage
- `insufficient_risk_reward` - Risk/reward ratio too low
- `drawdown_exceeded` - Max drawdown limit reached
- `heat_exceeded` - Portfolio heat (total risk) exceeded
- `daily_loss_limit` - Daily loss limit reached

#### `build_exit_plan(intent: SignalIntent, instrument) -> ExitPlan`
Calculate stop-loss and take-profit levels for a trade.

```python
exit_plan = risk_manager.build_exit_plan(intent, instrument)
print(f"Stop price: {exit_plan.stop_price}")
print(f"Target prices: {exit_plan.target_prices}")
print(f"Risk/reward: {exit_plan.reward_to_risk_ratio}")
```

#### `validate_market_quality(intent) -> MarketQualityDecision`
Validate that market conditions are suitable for a trade (volatility, liquidity checks).

```python
quality = risk_manager.validate_market_quality(intent)
if not quality.allowed:
    print(f"Market quality issue: {quality.reason}")
```

#### Position & Portfolio Status

**`peak_equity: float`**
Highest equity reached during backtest (used for drawdown calculations).

```python
peak = risk_manager.peak_equity
```

---

## PerformanceAnalyzer - Metrics & Reporting

`PerformanceAnalyzer` calculates trading performance metrics and exports reports.

### Creating an Analyzer

```python
from backtester.analytics import PerformanceAnalyzer

analyzer = PerformanceAnalyzer(config)
```

### Key Methods

#### `calculate(snapshots, trades, fills, execution_stats) -> dict`
Calculate all performance metrics.

```python
metrics = analyzer.calculate(
    snapshots=result.snapshots,
    trades=result.trades,
    fills=result.fills,
    execution_stats=result.execution_stats
)
```

**Returns a dictionary with these metric groups:**

**`profitability`**
```python
prof = metrics["profitability"]
print(f"Net profit: {prof['net_profit']:.2f} USDT")
print(f"Net return: {prof['net_return_pct']:.2%}")
print(f"Trade count: {prof['trade_count']}")
print(f"Win rate: {prof['win_rate']:.2%}")
print(f"Avg trade: {prof['avg_trade_pnl']:.2f}")
print(f"Largest win: {prof['largest_win']:.2f}")
print(f"Largest loss: {prof['largest_loss']:.2f}")
print(f"Profit factor: {prof['profit_factor']}")
```

**`risk`**
```python
risk = metrics["risk"]
print(f"Max drawdown: {risk['max_drawdown']:.2%}")
print(f"Drawdown duration: {risk['drawdown_duration_bars']} bars")
print(f"Sharpe ratio: {risk['sharpe_ratio']:.2f}")
print(f"Sortino ratio: {risk['sortino_ratio']:.2f}")
print(f"Volatility: {risk['volatility']:.2%}")
print(f"VAR (95%): {risk['var_95']:.2%}")
```

**`trades`**
```python
trade_stats = metrics["trades"]
print(f"Avg trade duration: {trade_stats['avg_trade_duration_bars']} bars")
print(f"Long trades: {trade_stats['long_trades']}")
print(f"Short trades: {trade_stats['short_trades']}")
print(f"Long win rate: {trade_stats['long_win_rate']:.2%}")
print(f"Short win rate: {trade_stats['short_win_rate']:.2%}")
```

**`execution`**
```python
exec_stats = metrics["execution"]
print(f"Slippage (market orders): {exec_stats['slippage_bps']:.1f} bps")
print(f"Fill rate: {exec_stats['fill_rate']:.2%}")
```

#### `export(output_dir, snapshots, trades, fills, metrics, execution_stats)`
Export reports to CSV and JSON files.

```python
analyzer.export(
    output_dir="reports/latest",
    snapshots=result.snapshots,
    trades=result.trades,
    fills=result.fills,
    metrics=result.metrics,
    execution_stats=result.execution_stats
)
```

**Creates:**
- `report.json` - All metrics in JSON format
- `equity_curve.csv` - Portfolio equity over time
- `trades.csv` - All trades with entry/exit details
- `fills.csv` - All individual fills
- `monthly_returns.csv` - Returns by month

---

## Strategy - Signal Generation

`IndicatorSignalStrategy` generates trading signals using your `indicators.py` logic.

### Overview

The strategy adapter bridges your existing live trading indicators to the backtester, preventing forward-leakage by simulating a synthetic forming candle.

### Creating a Strategy

```python
from backtester.strategy import IndicatorSignalStrategy

strategy = IndicatorSignalStrategy(config.strategy)
```

### Key Methods

#### `prepare_data(data: DataPortal)`
Calculate indicators on all data for the required timeframes.

```python
strategy.prepare_data(data)
# This calls indicators.calculate_indicators on your data
```

#### `generate_intents(context: StrategyContext) -> list[SignalIntent]`
Generate trading signals for the current bar.

**Input:** `StrategyContext` with:
- `timestamp` - Decision timestamp (after current candle close)
- `bar_index` - Current bar index
- `data` - DataPortal with OHLCV and indicator data
- `exchange` - SimulatedExchange with portfolio state

**Output:** List of `SignalIntent` objects with:
- `symbol` - Trading pair
- `side` - Entry direction (LONG or SHORT)
- `score` - Signal strength (0-100)
- `entry_price_hint` - Optional entry price
- `stop_price` - Suggested stop-loss
- `target_prices` - List of take-profit targets

```python
context = StrategyContext(
    timestamp=decision_time,
    bar_index=100,
    data=data,
    exchange=exchange
)
intents = strategy.generate_intents(context)
for intent in intents:
    print(f"Signal: {intent.symbol} {intent.side} (score: {intent.score})")
```

---

## OptimizationRunner - Parameter Optimization

`OptimizationRunner` performs systematic optimization of strategy and risk parameters.

### Creating a Runner

```python
from backtester.optimization import OptimizationRunner

config = BacktestConfig.from_json("configs/backtest_config.json")
runner = OptimizationRunner(config)
```

### Key Methods

#### `grid_search(param_grid: dict, keep_results: bool = False) -> list[OptimizationResult]`
Run all combinations of parameters using brute-force grid search.

```python
results = runner.grid_search({
    "strategy.min_risk_reward": [1.2, 1.45, 1.8],
    "risk.risk_per_trade_pct": [0.0025, 0.005, 0.0075],
    "execution.market_slippage_bps": [2.0, 4.0, 6.0]
})

# Results are sorted by objective value (best first)
for result in results[:10]:
    print(f"Sharpe: {result.objective_value:.2f}")
    print(f"Params: {result.params}")
    print(f"Net profit: {result.metrics['profitability']['net_profit']:.2f}")
```

**Returns:** Sorted list of `OptimizationResult` with:
- `params` - Parameter values for this run
- `objective_value` - Sharpe ratio (or custom objective)
- `metrics` - Full performance metrics
- `result` - Full `BacktestResult` (if `keep_results=True`)

#### `random_search(param_grid: dict, iterations: int, keep_results: bool = False)`
Sample random parameter combinations.

```python
results = runner.random_search(
    param_grid={
        "strategy.min_risk_reward": [1.2, 1.5, 1.8, 2.0],
        "risk.risk_per_trade_pct": [0.002, 0.005, 0.01],
    },
    iterations=100
)
```

#### `walk_forward(param_grid: dict, keep_results: bool = False)`
Walk-forward analysis: train on one period, test on next period.

Prevents overfitting by ensuring optimization parameters are never tested on training data.

```python
segments = runner.walk_forward({
    "strategy.min_risk_reward": [1.2, 1.45, 1.8],
    "strategy.stop_mode": ["ma", "atr"]
})

for segment in segments:
    print(f"Train: {segment.train_start} to {segment.train_end}")
    print(f"Test: {segment.test_start} to {segment.test_end}")
    print(f"Best params: {segment.best_result.params}")
```

#### `monte_carlo(n_samples: int = 1000)`
Monte Carlo analysis: reshuffle returns to assess luck vs. skill.

```python
mc_results = runner.monte_carlo(n_samples=1000)
print(f"Empirical Sharpe: {mc_results.empirical_sharpe:.2f}")
print(f"Percentile rank: {mc_results.percentile_rank:.1%}")
```

---

## BacktestResult - Results Export

`BacktestResult` aggregates all backtest outputs and provides export functionality.

### Fields

```python
result.config           # BacktestConfig used
result.snapshots        # list[PortfolioSnapshot] - Portfolio state each bar
result.fills            # list[Fill] - All trade executions
result.trades           # list[TradeRecord] - Closed trades with P&L
result.orders           # list[Order] - All orders (filled/unfilled)
result.metrics          # dict - Performance metrics
result.execution_stats  # dict - Execution statistics
```

### Key Methods

#### `export(output_dir: str | Path)`
Export results to CSV and JSON files.

```python
result.export("reports/latest")
```

**Creates files:**
- `report.json` - Comprehensive metrics and config
- `equity_curve.csv` - Equity over time
- `trades.csv` - Trade-by-trade details
- `fills.csv` - All fills
- `monthly_returns.csv` - Monthly P&L

---

## Typical Workflows

### Workflow 1: Basic Backtest

```python
from backtester import BacktestConfig, DataPortal, BacktestEngine

# Load config and data
config = BacktestConfig.from_json("configs/backtest_config.json")
data = DataPortal.from_config(config.data)

# Run backtest
engine = BacktestEngine(config=config, data=data)
result = engine.run()

# Display results
print(f"Net Profit: {result.metrics['profitability']['net_profit']:.2f} USDT")
print(f"Return: {result.metrics['profitability']['net_return_pct']:.2%}")
print(f"Max Drawdown: {result.metrics['risk']['max_drawdown']:.2%}")
print(f"Sharpe Ratio: {result.metrics['risk']['sharpe_ratio']:.2f}")
print(f"Trade Count: {result.metrics['profitability']['trade_count']}")

# Export reports
result.export("reports/latest")
```

### Workflow 1B: Real Bybit Data Before Backtesting

Use `backtester.bybit_cli` to maintain real local candles, then run the normal
backtest. The backtester still reads local CSV files through `DataPortal`; the
Bybit layer is responsible for keeping those files fresh and clean.

```bash
# One-time historical backfill
python -m backtester.bybit_cli --config configs/bybit_backtest_config.json backfill --start 2024-01-01

# Daily incremental update
python -m backtester.bybit_cli --config configs/bybit_backtest_config.json update --fallback-start 2024-01-01

# Detect and repair missing/corrupt ranges
python -m backtester.bybit_cli --config configs/bybit_backtest_config.json check-gaps
python -m backtester.bybit_cli --config configs/bybit_backtest_config.json repair

# Run backtest from real local data
python -m backtester.cli --config configs/bybit_backtest_config.json
```

You can also combine maintenance with the backtest command:

```bash
python -m backtester.cli --config configs/bybit_backtest_config.json --bybit-update --bybit-repair --bybit-start 2024-01-01
```

The websocket recorder stores only confirmed candles by default:

```bash
python -m backtester.bybit_cli --config configs/bybit_backtest_config.json websocket --timeframe 5m
```

### Workflow 2: Grid Search Optimization

```python
from backtester import BacktestConfig
from backtester.optimization import OptimizationRunner

config = BacktestConfig.from_json("configs/backtest_config.json")
runner = OptimizationRunner(config)

# Define parameter grid
param_grid = {
    "strategy.min_risk_reward": [1.2, 1.45, 1.8],
    "risk.risk_per_trade_pct": [0.0025, 0.005, 0.0075],
    "execution.market_slippage_bps": [2.0, 4.0, 6.0]
}

# Run grid search
results = runner.grid_search(param_grid)

# Analyze top 10 results
print("Top 10 Results:")
for i, result in enumerate(results[:10], 1):
    print(f"{i}. Sharpe: {result.objective_value:.2f}")
    print(f"   Params: {result.params}")
    print(f"   Net Profit: {result.metrics['profitability']['net_profit']:.2f}")
    print()

# Export best result
best = results[0]
print(f"Best parameters: {best.params}")
```

### Workflow 3: Walk-Forward Analysis

```python
from backtester import BacktestConfig
from backtester.optimization import OptimizationRunner

config = BacktestConfig.from_json("configs/backtest_config.json")
runner = OptimizationRunner(config)

# Run walk-forward with parameter ranges
segments = runner.walk_forward({
    "strategy.min_risk_reward": [1.2, 1.45, 1.8],
    "strategy.stop_mode": ["ma", "atr"]
})

print(f"Total segments: {len(segments)}")
for segment in segments:
    print(f"\nSegment {segment.index}")
    print(f"Train: {segment.train_start.date()} to {segment.train_end.date()}")
    print(f"Test: {segment.test_start.date()} to {segment.test_end.date()}")
    if segment.best_result:
        print(f"Best params: {segment.best_result.params}")
        print(f"Test metrics: {segment.test_result.metrics['profitability']}")
```

### Workflow 4: Custom Strategy

```python
from backtester import BacktestConfig, DataPortal, BacktestEngine, Strategy, StrategyContext
from backtester.models import SignalIntent, Side

class CustomStrategy(Strategy):
    def prepare_data(self, data):
        # Calculate custom indicators
        pass
    
    def generate_intents(self, context: StrategyContext):
        intents = []
        for symbol in context.data.symbols():
            # Generate signals based on your logic
            intent = SignalIntent(
                symbol=symbol,
                side=Side.LONG,
                score=75,
                entry_price_hint=None,
                stop_price=95.0,
                target_prices=[110.0, 115.0]
            )
            intents.append(intent)
        return intents

# Run backtest with custom strategy
config = BacktestConfig.from_json("configs/backtest_config.json")
data = DataPortal.from_config(config.data)
strategy = CustomStrategy()

engine = BacktestEngine(config=config, data=data, strategy=strategy)
result = engine.run()
```

### Workflow 5: Data Validation & Gap Detection

```python
from backtester import BacktestConfig, DataPortal

config = BacktestConfig.from_json("configs/backtest_config.json")
data = DataPortal.from_config(config.data)

# Validate all data
messages = data.validate()
print("Validation messages:")
for msg in messages:
    print(f"  {msg}")

# Check for gaps
for symbol in data.symbols():
    report = data.detect_gaps(symbol, "5m")
    if report.has_gaps:
        print(f"{symbol} has {report.missing_candles} missing 5m candles")
        for gap_start, gap_end, count in report.gaps[:5]:
            print(f"  Gap: {gap_start} to {gap_end} ({count} candles)")
```

---

## Configuration

The backtester is controlled via `BacktestConfig`, typically loaded from JSON:

```python
config = BacktestConfig.from_json("configs/backtest_config.json")
```

### Key Config Sections

**Data Config** (`config.data`)
```python
config.data.symbols           # ["SAGAUSDT", "NEARUSDT"]
config.data.timeframes        # ["5m", "1h", "4h"]
config.data.base_timeframe    # "5m"
config.data.data_dir          # "data/"
config.data.resample_from     # "5m" (derive higher TFs from lower)
```

**Risk Config** (`config.risk`)
```python
config.risk.initial_equity           # 10000.0 (starting capital)
config.risk.max_leverage             # 5.0
config.risk.risk_per_trade_pct       # 0.01 (1% of equity per trade)
config.risk.max_open_positions       # 3
config.risk.max_drawdown_pct         # 0.20 (20% max drawdown)
config.risk.portfolio_heat_max_pct   # 0.10 (max % of equity at risk)
config.risk.daily_loss_limit_pct     # 0.05 (daily loss limit)
```

**Strategy Config** (`config.strategy`)
```python
config.strategy.entry_timeframe      # "5m"
config.strategy.htf_timeframe        # "1h"
config.strategy.min_risk_reward      # 1.5 (minimum R:R ratio)
config.strategy.max_signals_per_bar  # 3 (max trades per candle)
config.strategy.use_signal_ranking   # True (rank signals by strength)
config.strategy.stop_mode            # "ma" or "atr"
```

**Execution Config** (`config.execution`)
```python
config.execution.taker_fee_pct              # 0.0002 (0.02%)
config.execution.maker_fee_pct              # 0.0001 (0.01%)
config.execution.market_slippage_bps        # 2.0 (2 basis points)
config.execution.limit_fill_probability     # 0.8 (80% chance to fill)
config.execution.market_order_latency_bars  # 1 (execute next candle)
config.execution.stop_first_if_ambiguous    # True
```

**Analytics Config** (`config.analytics`)
```python
config.analytics.export_json    # True
config.analytics.export_csv     # True
config.analytics.output_dir     # "reports/latest"
```

**Optimization Config** (`config.optimization`)
```python
config.optimization.objective           # "sharpe_ratio"
config.optimization.walk_forward_months # 3
config.optimization.train_months        # 6
```

---

## Summary

| Task | Class/Method |
|------|--------------|
| Load data | `DataPortal.from_config()` → `.load()` |
| Run single backtest | `BacktestEngine().run()` |
| Check market data | `DataPortal.history()`, `.detect_gaps()` |
| Process order fills | `SimulatedExchange.process_candle()` |
| Size trades | `RiskManager.evaluate_intent()` |
| Calculate metrics | `PerformanceAnalyzer.calculate()` |
| Export reports | `BacktestResult.export()` |
| Grid search | `OptimizationRunner.grid_search()` |
| Walk-forward test | `OptimizationRunner.walk_forward()` |
| Custom signals | Implement `Strategy` protocol |

For detailed examples, see the `examples/` directory.
