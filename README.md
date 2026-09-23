# FX Forward-Test Platform

MetaTrader 5 FX forward-testing service with a FastAPI backend, SQLite trade
journal, persisted risk halts, and a React dashboard. The original Bybit bot
remains in this repo for reference and backtesting continuity, but the FX path
lives under `fxbot/` and is the runnable product for MT5 forward testing.

## Experimental sniper qualification

The optional FX entry layer defaults to `FX_SNIPER_MODE=off`. Start with shadow
observations using the [configuration and research guide](docs/FX_SNIPER_RESEARCH.md).
The [baseline audit](docs/FX_SNIPER_AUDIT.md) records the rationale and evidence
limits. No profitability improvement has been established. Install offline
FX test dependencies with `pip install -r requirements-fx-research.txt`.

## FX Safety Defaults

This build defaults to `MT5_DEMO_ONLY=true` and refuses to run when the
connected MT5 account reports a real trade mode. Change that setting only after
you have deliberately reviewed the order sizing, broker symbol mapping, and
risk controls. MT5 execution requires a locally installed MetaTrader 5 terminal
that can connect to your broker account.

## FX Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env`, then fill in the MT5 values:

```env
MT5_LOGIN=12345678
MT5_PASSWORD=your_mt5_password
MT5_SERVER=YourBroker-Demo
MT5_TERMINAL_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
MT5_DEMO_ONLY=true
MT5_DEVIATION_POINTS=20
MT5_MAGIC_NUMBER=260828
# Signed correction for measured broker-server timestamps; UTC+3 requires -10800.
FX_MT5_TIME_OFFSET_SECONDS=0
FX_API_KEY=change-this-demo-control-key
FX_INSTRUMENTS=EUR_USD,GBP_USD,USD_JPY,AUD_USD
MT5_SYMBOL_MAP=EUR_USD=EURUSD,GBP_USD=GBPUSD,USD_JPY=USDJPY,AUD_USD=AUDUSD
```

If your broker uses suffixed symbols such as `EURUSD.a`, set
`MT5_SYMBOL_MAP=EUR_USD=EURUSD.a,...`. The strategy keeps underscore pair
names internally and sends the mapped symbol to MT5.

MT5 timestamps should normally be UTC, so leave `FX_MT5_TIME_OFFSET_SECONDS=0`.
If a direct host-versus-quote measurement proves that every broker timestamp
has the same whole-hour offset, configure the inverse correction. For example,
quotes consistently three hours ahead of synchronized UTC require
`FX_MT5_TIME_OFFSET_SECONDS=-10800`. The correction applies to quotes, candles,
position open times, and closed-deal times while preserving freshness checks.

Optional high-impact news blackout events can be loaded from a JSON file:

```env
FX_NEWS_EVENTS_FILE=data/news_events.example.json
```

### Free Forex Factory calendar (demo/forward testing only)

Merge [configs/fx_forexfactory_demo.env.example](configs/fx_forexfactory_demo.env.example)
into your existing `.env` without replacing MT5 credentials. The sniper shadow
profile includes these news settings too. Configuration is opt-in; no global
provider default or running account is changed by installing this code.

```env
MT5_DEMO_ONLY=true
FX_LIVE_TRADING_ENABLED=false
FX_NEWS_API_ENDPOINT=
FX_NEWS_API_KEY=
FX_NEWS_EVENTS_FILE=
FX_NEWS_EVENTS_JSON=
FX_USE_FOREX_FACTORY=true
FX_REQUIRE_NEWS_DATA=true
FX_NEWS_SYNC_INTERVAL_SECONDS=300
FX_NEWS_DATA_MAX_AGE_SECONDS=3600
FX_NEWS_MANUAL_OVERRIDE=none
FX_NEWS_EVENT_OVERRIDES=
```

The provider reads the public
[Forex Factory weekly JSON export](https://nfs.faireconomy.media/ff_calendar_thisweek.json).
It accepts the flat event array (currency in `country`, offset-aware ISO `date`),
as well as the legacy nested format. Timestamps are normalized to UTC, missing
actuals remain unknown, and the existing scorer and pair-aware blackout gate
remain authoritative. The example retains the 30-minute pre/post high-impact
blackouts; AI cannot authorize an entry rejected by the news gate.

Validate without starting MT5 or the trading worker:

```bash
python -m fxbot.news_check --env-file configs/fx_forexfactory_demo.env.example
```

This command uses a temporary cache, prints source/state/event count/upcoming
high-impact events, and exits nonzero for unavailable news. The profile is
applied only to the check process; merge it into `.env` and restart the backend
to enable it in the worker. After restart, check `/api/news` for the
`forexfactory` source, freshness, and parsed events before starting a demo run.

Provider priority remains HTTP endpoint, then manual file, then Forex Factory;
clear the higher-priority inputs to select the free feed. Unsupported JSON,
malformed events (including partial failures), empty weekly exports, and expired
weekly coverage are failures, not evidence of "no news". Generic providers may
still return an authoritative empty calendar. A static snapshot cannot mask an
initial live-provider failure. Ordinary fetch failures retain the last good cache
only until its 3600-second age limit; they do not reset its freshness. Entries
are blocked if there is no usable cache. HTTP 429 honors `Retry-After` with a
minimum five-minute cooldown; other free-feed failures also back off five minutes.

This free export has no guaranteed uptime, complete breaking-news coverage, or
release-time latency. Cache age measures our successful retrieval, not proof
that the publisher updated every event. The requested-week check detects old
weekly exports but cannot detect every upstream omission or delay. It is a
scheduled-event safety filter, not a low-latency news-trading signal. Do not rely
on it alone for funded/live trading: configuration rejects this provider when
demo protection is disabled or live trading is enabled. Use a separately
validated/licensed provider before any live rollout. Do not bypass freshness
or blackout blocks to force a trade.

News safety behavior is deterministic and runs outside the strategy and the
research LLM. The live entry gate evaluates market hours, calendar freshness,
currency/instrument relevance, high/medium impact windows, event overrides,
manual block/allow mode, and the emergency news kill switch before risk sizing
and MT5 execution. For a real account, `FX_REQUIRE_NEWS_DATA` defaults to
`true`; keep a licensed or otherwise verified calendar provider configured and
confirm its timestamps are UTC-normalized. A stale calendar blocks new entries
but does not cancel existing positions automatically.

Useful controls include `FX_MEDIUM_IMPACT_NEWS_ENABLED`,
`FX_NEWS_RESTRICTED_CURRENCIES`, `FX_NEWS_RESTRICTED_INSTRUMENTS`,
`FX_NEWS_EVENT_OVERRIDES`, `FX_NEWS_MANUAL_OVERRIDE`, and
`FX_NEWS_EMERGENCY_KILL_SWITCH`. The dashboard `/api/news` response exposes
the same structured decision used by the worker, including the matching event,
impact, timing, and restriction reason.

## Run The Backend

Start the FastAPI service. The worker runs as a background task, but it will not
scan or trade until the persisted bot state is set to `running`.

```bash
uvicorn fxbot.api:app --host 127.0.0.1 --port 8000
```

Control endpoints require `X-API-Key: $FX_API_KEY`:

```bash
curl -X POST http://127.0.0.1:8000/api/control/start ^
  -H "Content-Type: application/json" ^
  -H "X-API-Key: change-this-demo-control-key" ^
  -d "{\"reason\":\"manual forward test\"}"
```

Useful endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/status` | Bot run state, halt reason, worker state |
| `GET /api/positions` | Current open MT5 positions |
| `GET /api/trades` | Trade history with `instrument`, `state`, `outcome`, date filters |
| `GET /api/equity` | Equity snapshots for the chart |
| `GET /api/performance` | Win rate, P&L, profit factor, drawdown, Sharpe/Sortino |
| `GET /api/config` | Sanitized strategy/risk/runtime config |
| `WS /ws/live` | Live dashboard snapshots |

## Run The Dashboard

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. The dashboard displays a persistent
`MT5 DEMO / FORWARD TEST` badge when `MT5_DEMO_ONLY=true`, live bot state,
equity curve, statistics, open
positions, filtered trade history, and start/pause/stop controls.

Set a different backend URL with:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## Unattended Forward Testing

MT5's official Python package talks to a local desktop terminal, so native
Windows execution is the straightforward deployment path. The MT5 dependency is
installed only on Windows from `requirements.txt`. The Docker compose file
still runs the API/dashboard container for non-execution use, but live MT5
trading from a Linux container requires a separately managed terminal bridge.

```bash
docker compose up -d --build
```

`docker-compose.yml` uses `restart: unless-stopped`, persists `data/`, and runs
the FastAPI service with the worker in the background. A systemd unit is also
provided at `deploy/systemd/fx-forward-test.service` for operators who prefer a
host-level service.

## FX Retune Summary

The FX strategy keeps MA7/MA14/MA28 stacking, ADX/DI confirmation, HTF
agreement, ATR stops, partial take-profit, daily loss halts, and balance-aware
sizing. It changes the crypto assumptions: 15-minute entries by default, London
and New York sessions only, Friday/Sunday market-close guards, rollover guards,
pip-based ATR/stop filters, MT5 lot-backed unit sizing, quote-to-account
pip-value conversion, broker contract/volume-step handling, account-leverage
margin sizing, swap logging from MT5 positions/deals, and portfolio exposure
caps by pair/gross/currency. Full rationale is in
`docs/FX_STRATEGY_RATIONALE.md`.

The default FX entry profile is intentionally selective: it requires a
minimum quality score, directional DI separation, non-decreasing ADX, MA28
slope in the trade direction, an aligned higher-timeframe momentum candle, and
a bounded distance from MA7. These are configurable filters, not profitability
guarantees. Risk and target calculations use executable prices (ask for longs,
bid for shorts) rather than the midpoint so forward-test sizing is not
optimistically understated. See `winstrat.txt` for the current tuning decision
record and validation requirements.

---

## Trade Analysis Agent

The `forex_agent/` package is a **quantitative research and diagnostic system**
that watches the bot's trade journal and explains *why* trades behaved
differently from expectation. It does not predict prices, place trades, or
modify strategy parameters. It is a research assistant, not a trading guru.

### What it does

The bot (`fxbot/`) executes trades and records them in a SQLite/JSONL journal.
The agent reads that journal and answers seven questions for every trade:

1. **What happened?** -- structured summary of entry, exit, P&L, R-multiple.
2. **How unusual was it?** -- comparison against the portfolio baseline and
   statistically similar historical trades.
3. **What factors were associated with the outcome?** -- multi-factor
   diagnosis across six dimensions (signal, regime, execution, risk, timing,
   trade management).
4. **Why did the trade likely behave that way?** -- primary diagnosis with
   contributing and protective factors, each tagged as observation, association,
   hypothesis, or conclusion.
5. **How confident are we in that explanation?** -- numeric confidence score
   backed by sample sizes and statistical tests.
6. **Could the apparent explanation simply be noise?** -- the critic/anti-bias
   agent challenges every major finding for sample size, independence,
   survivorship bias, look-ahead bias, overfitting, and multiple-testing
   inflation.
7. **What experiment should we run to test the hypothesis?** -- structured
   experiment proposals with control/treatment groups, statistical test,
   acceptance criteria, and out-of-sample validation plan.

### How it helps the bot

| Problem the bot faces | How the agent helps |
| --- | --- |
| Repeated losses in the same session or regime | Detects regime/session deterioration and generates a hypothesis with a proposed experiment to validate before changing parameters. |
| Execution quality degrades silently | Tracks spread and slippage over time; flags when execution costs exceed historical norms. |
| Strategy drift goes unnoticed | Computes rolling expectancy, distribution shifts, and MAE/MFE shifts; alerts when recent performance diverges from baseline. |
| Manual review is slow and biased | Produces structured diagnostics for every trade (winners and losers) so the operator can focus on high-confidence findings instead of reading raw logs. |
| Overfitting from data-mined "patterns" | Every finding goes through the critic which flags small samples, multiple-testing, and out-of-sample instability. The agent explicitly warns "potential data-mined relationship" when appropriate. |
| No memory of past investigations | Research memory persists findings, hypotheses, experiments, and decisions across runs so the agent does not rediscover the same patterns. |

### Architecture

```
fxbot/                  Live trading bot (MT5 execution, journal writes)
    |
    v
forex_agent/
    |
    +-- data/           Ingestion (JSONL/SQLite), validation, schemas
    +-- analysis/       Statistical engine, anomaly detection, regime,
    |                   similarity matching, evidence packages,
    |                   critic/anti-bias, LLM reasoner
    +-- agent/          TradeAnalyst orchestrator, explanations,
    |                   hypothesis engine, research memory, reporting
    +-- main.py         CLI entry point
```

### Key concepts

**Multi-factor diagnosis.** Every trade receives a `TradeDiagnostic` with a
primary diagnosis, contributing factors, protective factors, observations,
hypotheses, and unknowns. Factors are classified by dimension (signal, regime,
execution, risk, timing, management) and evidence level (observation,
association, hypothesis, conclusion). The system never forces a trade into a
single category.

**Evidence packages.** For each trade the agent builds a structured `EvidencePackage`
containing baseline metrics, similar-trade statistics, regime context, execution
quality, timing context, risk context, anomalies, and statistical tests. This
becomes the factual foundation for any explanation -- the LLM (or template)
cannot invent data that is not in the package.

**Similar-trade analysis.** The agent finds historically comparable trades
using transparent rule-based matching on symbol, direction, session, regime,
signal strength, and stop distance. It always shows the match count, the
definition of "similar", and a sample-size warning when the match pool is small.

**Critic / anti-bias agent.** Every major finding is challenged by a critic
that checks for small sample sizes, non-independent trades, survivorship bias,
look-ahead bias, overfitting, multiple-testing inflation, and out-of-sample
instability. The critic can downgrade confidence and flag findings as
"preliminary" or "potential data-mined relationship."

**LLM reasoning layer.** When an OpenAI or Ollama API key is configured, the
agent sends the evidence package to an LLM constrained to produce a structured
explanation in the required format (Trade Summary, Expected Behavior, Primary
Diagnosis, Contributing Factors, Evidence, Counterfactuals, Confidence,
Alternative Explanations, Critic Assessment, Research Recommendation). Without
an API key the system uses a deterministic template fallback -- the agent
works identically, just without natural-language synthesis.

**Research memory.** Findings, hypotheses, experiments, and decisions are
persisted to `data/research_memory.jsonl` so the agent accumulates knowledge
across runs instead of rediscovering the same patterns.

### Running the agent

```bash
# Full analysis (metrics, health, failures, anomalies, winner analysis)
python -m forex_agent.main analyze

# Multi-factor diagnostic for a specific trade
python -m forex_agent.main diagnose <trade_id>

# Evidence package (structured JSON)
python -m forex_agent.main evidence <trade_id>

# Find historically similar trades
python -m forex_agent.main similar <trade_id>

# LLM-powered explanation (requires OpenAI or Ollama key)
python -m forex_agent.main explain <trade_id>

# Critic assessment of a finding
python -m forex_agent.main critique "Strategy edge degrades in high-volatility regimes"

# Diagnose all trades (winners and losers)
python -m forex_agent.main diagnose-all

# Strategy health score
python -m forex_agent.main health

# Weekly/monthly research report
python -m forex_agent.main report
python -m forex_agent.main monthly

# Research memory summary
python -m forex_agent.main research

# Propose experiments
python -m forex_agent.main experiments

# Data validation
python -m forex_agent.main validate

# Anomaly detection
python -m forex_agent.main anomalies

# Health dashboard (JSON)
python -m forex_agent.main dashboard

# Alerts
python -m forex_agent.main alerts
```

### LLM configuration (optional)

The agent works without any LLM. When configured, OpenAI is tried first,
Ollama is the fallback, and a template is used if neither is available.

```env
# OpenAI (primary)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o

# Ollama (fallback -- local, no API key needed)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
```

### Statistical methods

The agent uses classical statistics -- no machine learning, no neural networks,
no prediction models:

| Method | Purpose |
| --- | --- |
| Bootstrap confidence intervals | Uncertainty around expectancy and other metrics |
| Permutation tests | Non-parametric comparison of trade groups |
| Welch's t-test | Parametric comparison with unequal variances |
| Cohen's d / Hedges' g | Effect size measurement |
| Benjamini-Hochberg FDR | Multiple-testing correction |
| Bonferroni correction | Conservative multiple-testing correction |
| Z-score anomaly detection | Outlier identification in R-multiples, spreads, durations |
| R-multiple distribution analysis | Performance breakdown by regime, session, pair |
| MAE/MFE analysis | Adverse and favorable excursion tracking |
| Walk-forward validation | Temporal split between in-sample and out-of-sample |

### Design principles

- **The LLM must never become the authority for numerical calculations.** All
  important numerical claims originate from the quantitative/statistical layer.
- **The agent is comfortable saying "I don't know."** When evidence is
  insufficient, the system says so rather than manufacturing an explanation.
- **Findings are reproducible.** Every conclusion is traceable to a finding ID,
  hypothesis, dataset query, statistical method, and result.
- **The agent recommends experiments, not strategy changes.** It proposes
  controlled experiments with acceptance criteria, not automatic parameter
  adjustments.

## Legacy Bybit Bot

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
bot.py              Main scan loop, account checks, orders, risk controls
config.py           Environment-driven settings and defaults
indicators.py       MA, MAVOL, ADX, ATR, signal logic
backtester/         Historical data, execution, risk, analytics, optimization
configs/            Backtest configuration examples
examples/           Backtest and optimization entry points
fxbot/              FX forward-test platform (MT5 execution, API, journal)
forex_agent/        Trade analysis agent (diagnostics, statistics, LLM reasoner)
tests/              Unit tests
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

### Real Bybit Historical Data

The bundled sample data is deterministic smoke-test data, not research-grade
market history. For real validation, backfill Bybit candles into local CSVs and
run the backtester from those files:

```bash
python -m backtester.bybit_cli --config configs/bybit_backtest_config.json backfill --start 2024-01-01
python -m backtester.bybit_cli --config configs/bybit_backtest_config.json update --fallback-start 2024-01-01
python -m backtester.bybit_cli --config configs/bybit_backtest_config.json check-gaps
python -m backtester.bybit_cli --config configs/bybit_backtest_config.json repair
python -m backtester.cli --config configs/bybit_backtest_config.json
```

For scheduled maintenance, run the `update` command daily. For live candle
capture, run:

```bash
python -m backtester.bybit_cli --config configs/bybit_backtest_config.json websocket --timeframe 5m
```

## Risk Warning

Trading perpetual futures can lose money quickly, especially with leverage.
Start in dry-run and testnet, keep risk small, and review exchange orders
manually until you trust the automation.
