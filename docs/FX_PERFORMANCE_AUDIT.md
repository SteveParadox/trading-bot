# FX performance audit and implementation — 2026-10-19

The [September audit](FX_PERFORMANCE_AUDIT_2026-09-22.md) is retained unchanged as historical evidence; its configuration descriptions refer to that earlier baseline.

## A. Executive summary

Inspected PR #14 at `c3a00653ba19a36a823b7fd4089f12234eab915b`, on
`codex/fx-h4-h1-m5-confirmation`. The branch name is retained to update the
existing review; the intended entry timeframe is now **M15**, with H4 bias,
H1 confirmation and independently configurable M5 TP. Programmatic
`StrategySettings()` retains the legacy M15/H1 baseline for research and
existing integrations; environment-loaded worker settings select M15/H1/H4.

Implemented concrete strategy/configuration, replay, broker-query, numerical
model compatibility, provenance, API scheduling and dashboard defects. Wilder
indicator processing was the measured CPU bottleneck. The recurrence, seed,
NaN behavior and completed-candle causality are preserved; no risk limits or
strategy thresholds were tuned. No live account or order was used.

**Validation status:** local implementation and fake-broker regressions pass.
Hosted CI is externally blocked by an account billing lock. Broker latency,
strategy profitability and A/B/C portfolio results remain unverified because
no Windows MT5 terminal, authenticated broker history, execution ledger,
commission schedule or forward sample is available here. This is a demo-test
candidate, not a live release or a claim that every possible defect is resolved.

## B. Confirmed defects and corrections

| ID | Severity | Root cause and correction | Principal files |
|---|---|---|---|
| D1 | P0 | Environment default and deployment profile selected M5 entry. Restore M15; expose actual four timeframe roles on dashboard. Preserve explicit M5 research override. | `fxbot/config.py`, `configs/fx_h4_h1_m5_demo.env.example`, `frontend/src/App.tsx`, `README.md` |
| D2 | P0 | Serving checked feature schema/hash but accepted artifacts from other timeframe distributions. Worker now supplies exact entry/structure/bias/TP context. Primary, entry-timing and auxiliary artifacts must all match `validated_strategy_context`; otherwise no numerical probabilities are returned and required mode blocks. | `fxbot/ai/predictor.py`, `fxbot/ai/schemas.py`, `fxbot/forward.py` |
| D3 | P0 | Replay precomputed recursive indicators over full history; live MT5 seeds them on the last configured candle count. Replay now calculates each decision from the same bounded completed-bar window. | `fxbot/historical_reconstruction.py` |
| D4 | P0 | Missing/stale/nonpositive M5 TP ATR silently fell back to the ordinary target during replay. Replay now rejects that candidate, persists the rejection, increments `tp_data_blocks` and produces no fabricated outcome. Reconstruction version and candidate IDs advance to v2, keeping v1 evidence distinct. | `fxbot/historical_reconstruction.py` |
| D5 | P0 | Infinity was not classified as unavailable OHLCV or a precomputed direction indicator; it could pass MA/ADX comparisons. Reject nonfinite raw data and nonfinite required directional indicators. | `fxbot/strategy.py` |
| D6 | P0 | MT5 `positions_get`/`orders_get` failure (`None`) was treated as an empty result during duplicate lookup. Raise a disconnection error and retain the durable reservation; never authorize a new submission from an unsuccessful query. | `fxbot/mt5.py` |
| D7 | P1 | Database/analytics work for websocket snapshots ran synchronously in the API event loop. Offload snapshot construction to a thread; clean up websocket registrations in `finally`. A blocked snapshot regression verifies that the authenticated API remains responsive. | `fxbot/api.py` |
| D8 | P1 | Five-second fallback and thirty-second core polling overlapped; slow responses could stack request groups. Share in-flight core requests and keep the thirty-second core refresh for a live socket. Periodically refresh news and filtered trade history without resetting the current page. | `frontend/src/App.tsx`, `frontend/src/polling.mjs` |
| D9 | P1 | Code provenance ignored imported shared indicator math/configuration. Include `indicators.py` and root `config.py` in automatic code fingerprints. | `fxbot/security.py` |
| D10 | P1 | Two-frame mode permitted identical entry and bias timeframes. Enforce entry duration shorter than HTF even without a structure filter. | `fxbot/config.py` |
| D11 | P1 | Browser theme test used an ambiguous API-key label that also matched the reveal button; reduced-motion preference disabled only the theme-toggle transition. Use exact label matching and honor reduced motion on buttons. Browser tests now pass on desktop/mobile. | `frontend/tests/theme.browser.mjs`, `frontend/src/styles.css` |
| D12 | P1 | CI's browser-dependency installation could re-resolve locked dashboard dependencies. Install Playwright outside the frontend dependency tree, preserving the `npm ci` application tree for smoke tests. Include polling and strategy-role browser regressions in CI. | `.github/workflows/frontend-validation.yml`, browser tests |
| B1 | P0 performance | Six Wilder smoothing loops plus ADX recursion repeatedly dispatched pandas scalar reads/writes. Use owned output arrays with identical arithmetic order and pandas seeding; no input mutation or market-data cache. | `indicators.py` |

New regression evidence lives in `tests/test_fx_performance_audit.py`.
Existing M5 multi-timeframe tests remain; their shared fixture assertions also
support the intended M15 regression subclass. Frontend request coalescing has
dedicated success/rejection/retry tests in `frontend/tests/polling.test.mjs`.

## C. Measured engineering performance

Same Linux host, Python 3.12.14, pandas 2.2.3, NumPy 2.3.5.
Same deterministic synthetic 180-bar input, explicit M15/H1/H4 settings,
3 warmups and 60 timed calls per workload. Traced allocation peaks are
measured separately across 3 calls so tracing overhead does not contaminate
latency measurements. CPU is process CPU time across 60 calls, not a CPU
utilization percentage. These are engineering workloads, **not backtest data**.

| Workload | Median before ms | Median after ms | p95 before ms | p95 after ms | CPU before / after s | Peak allocations before / after bytes |
|---|---:|---:|---:|---:|---:|---:|
| `indicators_180_bars` | 27.035 | 6.631 | 28.270 | 8.735 | 1.628 / 0.412 | 192,388 / 193,628 |
| `three_frame_raw_decision` | 87.081 | 20.675 | 95.121 | 24.559 | 5.246 / 1.259 | 375,843 / 376,846 |
| `journal_read_100_of_10000` | 1.189 | 1.274 | 1.424 | 2.423 | 0.074 / 0.087 | 183,585 / 183,817 |
| `journal_signal_write` | 0.211 | 0.218 | 0.292 | 0.401 | 0.015 / 0.016 | 18,882 / 19,074 |
| `authenticated_config_api` | 2.281 | 2.057 | 3.912 | 2.560 | 0.139 / 0.131 | 84,961 / 104,211 |

The repeatable gain is in indicator and three-frame decision computation.
No material memory reduction is established. Database read/write variation
and API variation do not establish a production improvement; those paths
were instrumented, with no speculative index/cache changes. The database
read workload returns 100 rows from 10,000 synthetic signal rows. The API
workload includes authentication and configuration serialization, not broker
I/O or a loaded production deployment.

Initial cProfile captured 20 indicator calls: 2.081 seconds total under
profiling; `_wilders_smooth` consumed 1.574 seconds cumulatively (six calls
per frame), with scalar pandas dispatch dominating. Do not compare those
profiled times directly with the unprofiled table.

Machine-readable results: `reports/fx_audit_baseline.json` and
`reports/fx_audit_after.json`. Both record source fingerprints; the after
benchmark was run on uncommitted implementation code, so its git revision is
the base revision and its source SHA-256 distinguishes the measured tree.

Reproduce against clean source trees:

```bash
python scripts/benchmark_fx_performance.py --repo /path/to/pr14-base --output /tmp/before.json
python scripts/benchmark_fx_performance.py --repo /path/to/audited-tree --output /tmp/after.json
```

## D. Strategy comparison and data limitations

| Variant | Entry | Intermediate confirmation | Primary bias | Genuine performance results |
|---|---|---|---|---|
| A: original baseline | M15 | H1 is the primary confirmation | H1 | Unavailable: broker history/ledger not provided |
| B: intended strategy | M15 | H1 | H4 | Unavailable: broker history/ledger not provided |
| C: research only | M5 | H1 | H4 | Unavailable: broker history/ledger not provided |

For **each** variant, signal/trade counts, win/loss rates, profit factor,
net average wins/losses, expectancy, maximum drawdown, risk-adjusted return,
holding time, time-to-TP, MFE/MAE, false-breakout frequency, signal-to-execution
latency, 5/15-minute profitability, spread/slippage costs and instrument /
session / volatility / direction breakdowns are **not measured on genuine
broker data**. Synthetic fixtures cannot fill these cells. Crypto CSVs in
`data/` are unsuitable substitutes for executable FX broker quotes.

Replay reconstructs *conditional candidates*, not a portfolio backtest. It
explicitly does not replay account exposure, fills, AI, sniper execution gates,
partial exits or the complete position-management lifecycle. Matching candle
and TP semantics removes two real discrepancies; it does not establish full
historical/forward execution parity. Do not label hypothetical outcome rates
as realized win rates or compute portfolio drawdown from overlapping candidates.

On the Windows demo terminal, use separate databases/reports for every variant:

```powershell
$env:FX_ENTRY_TIMEFRAME='15m'
$env:FX_HTF_TIMEFRAME='1h'
$env:FX_STRUCTURE_TIMEFRAME=''
$env:FX_HTF_REQUIRE_MOMENTUM_CANDLE='true'
$env:FX_TP_TIMEFRAME=''
python -m fxbot.historical_reconstruction --start 2023-01-01 --end 2025-01-01 --database-url sqlite:///./data/research_A.db --report-output reports/research_A.json --dataset-output data/training/research_A.csv --news-authoritative
```

For B use `15m` / `1h` / `4h`, and explicitly set the H4 momentum option to
`false`; for C change only entry to `5m`. Keep TP blank for an original-target
comparison across all three, then evaluate B + M5 TP as a separate experiment.
Retain the same symbols, fees, sessions and news policy. `--news-authoritative`
is valid only with a genuinely complete historical news archive in
`FX_NEWS_EVENTS_JSON` / `FX_NEWS_EVENTS_FILE`; do not use it to bypass missing
news. No reconstruction command submits an order.

Use 2023–2024 train, 2025 validation, an untouched early-2026 test period and a
separate 2026 forward sample. Purge overlapping outcome horizons across split
boundaries. Choose thresholds on train/validation only and retain rejected
candidates to measure TAKE/WAIT/SKIP selection effects. Define false breakout
labels before comparing results. No parameter experiments or strategy
improvements were promoted from synthetic evidence.

## E. Safety and pipeline validation

| Pipeline stage | Inspection and evidence |
|---|---|
| Market data / clocks | MT5 completed-bar retrieval, executable bid/ask, signed time offsets and quote age remain in charge. Existing clock/freshness/weekend/session regressions pass. No quote cache or order-send retry was added. |
| Indicators / confirmation | M15 entry logic is retained; H1 disagreement/neutral/missing states block; H4 closed bars set bias. M15 and M5 test matrices cover incomplete and stale bars, explicit decision time, matching evidence IDs and expired order legs. |
| News / spread | Hard news/session/spread gates retain precedence. No protective filter was loosened. |
| ML / LLM | Feature/hash/version checks remain. Added exact timeframe compatibility for every contributing numerical artifact. Required unavailable predictions still reject; shadow remains observational. LLM timeout/policy tests remain intact. |
| Risk / execution | Existing lot/pip/currency/volume, RR, daily loss, drawdown, exposure and maximum-position tests pass. Fresh risk/price/news/confirmation revalidation remains before submission. Failed duplicate queries now fail closed. |
| Lifecycle / exits | Existing split TP, breakeven, trailing, partial fills, unknown-order and reconciliation regressions pass. M5 TP is metadata for target calculation; M15 ATR still sizes stop and position. No automatic unwind policy or risk-limit change was added. |
| Journaling / analytics | Durable candidate, reservation, outcome and closed-trade regressions pass. TP data rejections remain auditable. Existing summary/history caps are a limitation for large journals, not all-history quantitative proof. |
| API / frontend | Auth, API/config synchronization, theme initialization, polling error recovery, actual timeframe roles and responsive layouts tested. Synchronous snapshot work is moved off the event loop. |

Recommended `.env` for the requested demo profile:

```env
FX_ENTRY_TIMEFRAME=15m
FX_HTF_TIMEFRAME=4h
FX_STRUCTURE_TIMEFRAME=1h
FX_TP_TIMEFRAME=5m
MT5_DEMO_ONLY=true
FX_LIVE_TRADING_ENABLED=false
FX_ML_PREDICTION_MODE=off
FX_AI_DELIBERATION=off
```

Keep all existing position-risk, loss, spread and news values. M5 TP remains
optional: clear `FX_TP_TIMEFRAME` to keep the existing risk/reward target.
The minimum RR floor remains authoritative; M5 ATR does not guarantee a closer
TP or a five-minute exit. The existing profile filename contains `m5` for
compatibility but its contents now explicitly select M15 entry.

## F. Validation and CI

- Initial PR baseline: `python -m pytest -q tests/test_fx*.py` → **787 passed**.
- `python -m compileall -q fxbot indicators.py` → passed.
- Final `python -m pytest -q --ignore=tests/test_bybit_data.py` → **953 passed**, 2 subtests passed, 20 warnings.
- New audit module alone → **85 passed** (including M15 inherited cases).
- Full unfiltered `python -m pytest -q` → collection blocked by missing `backtester.bybit_data`. The same collection error was independently reproduced on the untouched PR base. It is unrelated to these FX fixes; the unfiltered failure remains explicitly reported.
- Locked `npm ci`, production TypeScript/Vite build, 4 theme tests and 2 polling tests → passed.
- Playwright theme smoke: desktop 1280 and mobile 375; authenticated strategy roles: desktop 1440 and mobile 390. API responses are explicit fixtures, not a running broker. No runtime errors or horizontal overflow observed.
- Existing warnings: deprecated naive UTC in legacy agent validation; FastAPI/TestClient dependency deprecation; sklearn probability-sum warnings in entry-model fixture tests. Vite warns about the pre-existing chart vendor chunk over 500 kB. No bundle-size or deployed load-time improvement is claimed.

GitHub Actions evidence:

- PR run `39210461816`, job `117922450762`: `runner_id=0`, `steps=[]`.
- Main run `39112051421`, job `117630081628`: same no-runner condition.
- PR job annotation: **“The job was not started because your account is locked due to a billing issue.”**
- Required owner action: resolve the lock under GitHub account **Settings → Billing and licensing**, correct the outstanding payment/account issue, then rerun FX and frontend validation on this PR. Workflow edits cannot resolve that account restriction.
- Workflow changes here are limited to regression coverage and isolating browser test dependencies. Keep PR draft and do not merge merely because local tests pass.

## G. Monitoring and remaining limitations

Authenticated `GET /api/monitoring` now includes a `performance` object:

- Fixed operation labels for market data, indicators, strategy, ML, LLM, risk,
  order submission/modification, selected journal reads/writes, dashboard
  snapshots, scans and execution revalidation.
- Counts, exception counts, cumulative milliseconds and rolling p50/p95/last
  milliseconds; 256 samples per label, at most 64 labels. No per-candidate or
  symbol labels and no market-data cache. The registry is process-local.
- `signal_to_order` measures candidate decision timestamp to submission attempt,
  not broker acknowledgement or fill latency.
- Cumulative process CPU seconds; peak RSS bytes where the OS exposes them.
  Windows returns null for RSS when `resource` is unavailable. These are not
  current-memory or utilization percentages.

Existing monitoring still supplies quote freshness, clock health, unknown
orders, database-lock retries and rejection/event evidence. These durations
measure selected operations, not every SQL statement or every broker call.
Exceptions are counted; an ordinary returned rejection is recorded through the
existing decision journal rather than treated as an exception.

Still requires external validation: broker retrieval/submission/modification
latency, terminal disconnections, clock correction, fill quality, restart
recovery against the real terminal, model calibration and value after costs,
long-session memory/CPU utilization, and full portfolio strategy comparison.
Local numerical inference is synchronous; no claim is made of cancellation
of a hung native estimator. The API's regular summary uses bounded 10,000-row
history and is not a substitute for complete exported-ledger research. The
chart vendor chunk remains large. No speculative historical tuning, data
provider replacement or architectural rewrite was performed.

## H. Demo rollout and rollback

1. Keep the PR unmerged until hosted CI can run. Back up `.env` and the journal.
   Install the documented Python dependencies and use the locked frontend
   build (`npm ci`, `npm run build`) on the deployed revision.
2. Stop the worker, update the six timeframe/safety variables above, retain
   existing news/risk parameters, restart, and verify `GET /api/config` plus
   the dashboard display M15 / H1 / H4 / M5. Existing open trade TP orders
   are not retroactively resized by this configuration change.
3. Verify demo account identity, broker symbol mapping, fresh executable
   quotes, completed H1/H4 bars and the measured UTC correction. Do not
   infer a time offset from wall-clock appearance alone.
4. Begin with AI and numerical influence off. Check new operation timings,
   rejections, reservations and broker/journal state through a controlled demo
   session. Inject disconnect/restart cases in demo and reconcile every order.
5. A compatible model artifact needs `validated_strategy_context` equal to the
   active configuration on **every** contributing model, for example:

```json
{"validated_strategy_context":{"entry_timeframe":"15m","structure_timeframe":"1h","htf_timeframe":"4h","tp_timeframe":"5m"}}
```

   This declaration is a compatibility contract, not evidence of predictive
   skill. Add it only after independent out-of-sample validation, keeping the
   validation report with the model. Old artifacts lacking it are unavailable
   to the worker; required mode fails closed. Direct legacy research callers
   that omit strategy context retain their previous API behavior.
6. Roll back by pausing entries and restoring the previous deployment commit
   and saved environment. For the original strategy explicitly set M15 entry,
   H1 bias, blank structure and TP, and restore the original H1 momentum gate.
   Do not roll back to M5 entry just because it was PR #14's accidental default.
   Preserve all journals, reservations and v1/v2 research databases. Never
   clear unresolved-order state merely to resume trading.

Changed-file inventory is available through the PR diff. Machine-readable
benchmark evidence is committed beside this report; the final PR description
records implementation commits and validation status.
