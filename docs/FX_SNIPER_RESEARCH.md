# FX sniper research implementation

Read [the baseline audit](FX_SNIPER_AUDIT.md) first. This is an experimental
qualification layer, not evidence of a trading edge. No broker was contacted
and no MT5 trading process was started during implementation.

## Modes and rollout

`FX_SNIPER_MODE=off` is the default. `shadow` records supplemental qualification
without altering baseline entries, stops or exits. `enforce` rejects baseline
setups that fail a critical check. It never creates a new signal or bypasses
news, market-hours, account, risk, spread or AI policy. Enforced signals receive
an independent wall-clock candle-age check and safety revalidation after AI.
The final MT5 request also refuses an adverse quote move beyond approved entry
or a spread above the absolute/ATR ceiling. Fills can still slip or gap.

Start with [the demo shadow environment](../configs/fx_sniper_shadow.env.example).
It requires an authoritative calendar; missing calendar coverage intentionally
blocks entries. AI shadow requires the existing endpoint configuration and can
be set to off if no endpoint is available. Unknown cost assumptions are recorded
as unknown, and block qualification when enforcement/cost filtering is enabled.
Set commission and slippage from the actual broker/account/pair before research.

### Deliberate correctness changes in every mode

- Market closes use bid for a long and ask for a short, with a complete request
  passed to MT5's filling-mode check.
- Successful stop updates immediately update the local snapshot, so trailing
  cannot undo a just-applied break-even improvement. New fills preserve initial
  stop context across reconciliation/restarts. Old trades retain the legacy
  activation fallback when their initial stop is unavailable.
- At most one newly submitted setup per scan consumes the portfolio snapshot;
  subsequent scans reload account state. Unresolved orders block new allocation.
  Recovery recognises a confirmed position as a fill. Partial exits fall back
  to one order if fewer than two position slots remain. Closed signal IDs cannot
  reenter on the same candle/trigger after their original trade closes.
- TP, as well as SL, must meet the instrument's minimum stop distance.

These fixes intentionally apply in off/shadow mode too; baseline-equivalent
strategy selection does not mean retaining execution defects.

## Mathematical definitions

All signal features use completed bars. Baseline HTF/ADX/DI/slope filters provide
direction and setup; the supplemental regime check measures current ATR divided
by median prior ATR. It labels compression, trend, trend expansion, or unstable
expansion. This is a simple volatility classification, not an exhaustive regime
model or a statistical classification of exhaustion.

For side sign `s` (+1 long, -1 short), extension is
`max(0, s*(executable_entry-MA7)/ATR)`. Recent movement is
`s*(executable_entry-close[trigger_lookback])/ATR`; candle expansion is
`(high-low)/ATR`. A trigger requires a directional candle and close-location
strength. Breakout means a close beyond the prior local extreme. Pullback means
the preceding candle touched MA7 and the current candle closes beyond that
preceding candle's directional extreme. These are alternatives, not a
stateful multi-bar breakout/retest system. No lower timeframe is invented.

Structural obstacles use prior confirmed one-neighbour swing extrema plus the
prior range extreme. No candidate-bar highs are used to invent a prior obstacle.
No observed obstacle is recorded explicitly; it does not prove unlimited room.
The optional stop is the recent adverse extreme plus an ATR buffer. It remains
subject to the existing ATR/pip bounds, broker minimum distance and minimum RR.

Let `r` be executable entry-to-target reward and `S` the spread. Underlying gross
movement is `G=r+S`; estimated round-trip cost is
`C=S+pip_size*(2*slippage_per_side+round_trip_commission)`.
Qualification requires `C/G <= max_cost_ratio` and positive net target.
Net target is `r - nonspread_cost`; spread is not subtracted a second time.
Enforced sizing includes estimated nonspread costs in loss per unit. This is a
cost budget, not a guarantee that a stop caps losses during gaps.

The baseline score remains unchanged (maximum 80 with volume disabled). It is
not a probability. The supplemental layer records stage booleans and measurements
instead of manufacturing a second calibrated-looking score. No EV/win-probability
model is fitted from inadequate evidence.

## Configuration

All names below have prefix `FX_SNIPER_`. Settings are immutable and validated.
Numerical defaults are research hypotheses; no historical optimum was selected.

| Suffix | Default | Meaning |
|---|---|---|
| MODE | off | off / shadow / enforce |
| REGIME_FILTER | true | Prior-median ATR ratio gate |
| LOCATION_FILTER | true | Executable extension and structural room gates |
| TRIGGER_REQUIRED | true | Require an entry event |
| ANTI_CHASE | true | Candle/recent-movement gates |
| COST_FILTER | true | Require known, acceptable round-trip costs |
| STRUCTURE_STOP | false | Use structural stop in enforce only |
| FAILURE_EXIT | false | Close after completed-bar loss of trigger level |
| TIME_EXIT | false | Close stagnant enforce-owned trades |
| LOOKBACK | 20 | Prior bars for volatility and structure |
| TRIGGER_LOOKBACK | 3 | Prior breakout/recent-movement bars; stop extrema window |
| TRIGGER_MODE | either | either / breakout / pullback |
| MIN_ATR_RATIO | 0.7 | Minimum current/prior-median ATR |
| MAX_ATR_RATIO | 2.0 | Maximum current/prior-median ATR |
| MAX_CANDLE_ATR | 1.8 | Maximum trigger range/ATR |
| MAX_MOVE_ATR | 2.0 | Maximum directional recent move/ATR |
| MAX_EXTENSION_ATR | 0.6 | Maximum executable distance from MA7/ATR |
| MIN_CLOSE_STRENGTH | 0.65 | Directional close fraction within trigger candle |
| STRUCTURE_BUFFER_ATR | 0.1 | Buffer beyond structural invalidation |
| MAX_COST_RATIO | 0.20 | Maximum round-trip cost / gross move |
| SLIPPAGE_PIPS_PER_SIDE | unset | Unknown until measured/configured; explicit zero permitted |
| COMMISSION_PIPS_ROUND_TRIP | unset | Both sides combined, converted to pair pip equivalent |
| TIME_EXIT_BARS | 4 | Fully completed bars starting after entry |
| MIN_PROGRESS_ATR | 0.25 | Minimum current favorable progress for time exit |
| FAILURE_BUFFER_ATR | 0.1 | Completed-bar buffer beyond trigger invalidation |

Failure/time exits use entry ATR, not a moving risk denominator. They apply only
to trades whose persisted entry snapshot was enforce mode and while current mode
is enforce. Failure exit is disabled for entries without a trigger. A pending or
uncertain close is persisted before/after submission and is not automatically
retried; reconcile it before operator action. The broker SL remains in place.
These are scan/closed-bar exits, not instantaneous tick-level invalidation exits.
No new fixed-dollar profit target, martingale or averaging-down rule is added.

## Instrumentation and offline use

The existing SQLite event journal gains `sniper_candidate` snapshots with baseline
features, HTF context, side, executable bid/ask, observation/signal timestamps,
regime, normalized location, trigger/structure, costs and stage decisions.
Risk-rejected intents also carry available snapshots in their existing records.
Candidates blocked before baseline setup qualification retain existing skip logs.
Open trades persist sampled executable MAE/MFE and MFE observation time; sampling
misses excursions between scans. Reconciliation merges context instead of losing it.

Enable `FX_JSONL_JOURNAL` for the offline export. Supply actual broker quote CSV:

```csv
timestamp,instrument,bid,ask
2026-09-21T08:00:01Z,EUR_USD,1.10000,1.10010
```

The row above illustrates schema, not real evidence. Timestamps must be timezone
aware, strictly increasing per symbol, and compatible with journal observation
time. Missing quote coverage is excluded and counted, not converted into a result.

```bash
python -m fxbot.sniper_research --journal data/fx_sniper_shadow.jsonl --quotes data/fx_quotes.csv --train-end 2026-11-01T00:00:00Z --validation-end 2026-12-01T00:00:00Z --output reports/fx_sniper_candidate_study.json
```

Choose boundaries before looking at outcomes. The dates illustrate command
syntax, not an instruction to optimize on a particular period. The tool deduplicates
rescans, preserves the earliest observation, and purges candidate horizons crossing
split boundaries. It evaluates target R values 0.25/0.5/0.75/1/1.5 and cost multipliers
1/1.25/1.5/2, with first observed executable stop/target/horizon exits. Gap losses
are retained and favorable target gaps are capped. It reports accepted/rejected
groups, individual-gate ablations, extension caps 0.4/0.6/0.8, instrument/session/
regime groups, expectancy, payoff, win/loss rates, PF, holding times and excursions.

Entry gate membership is frozen at observed settings across target/cost scenarios;
these are outcome diagnostics, not deployable alternate configurations. The tool
does not replay live partial exits, trailing, news updates, portfolio allocation,
failure/time exits or revised entry timing. It cannot reconstruct setups that the
baseline never generated. Overlapping candidates are not independent trades.
Therefore portfolio net profit, drawdown, recovery factor, Sharpe and Sortino are
explicitly unavailable. Do not sum candidate R into a claimed trading equity curve.

## Evidence status and remaining work

No broker FX quote history or forward-test database is checked in. Crypto CSVs
and `reports/latest` / `reports/bybit_latest` were not used for parameter tuning.
No empirical before/after profitability, selected threshold, confidence interval,
pair/session preference or smallest viable target is claimed.

Before enabling enforcement beyond controlled demo experiments, collect paired
FX candles/quotes, real commission/slippage, broker symbol specs, candidate logs,
and point-in-time news coverage. Then replay the complete portfolio/execution
policy on rolling chronological folds, with untouched final holdout periods.
Compare immediate/confirmation/retest timing, fixed/ATR/structure targets,
failure/time exits and BE/trailing as separate ablations. Select broad stable
regions on training/validation only; assess untouched folds once.

Full walk-forward parameter selection, Monte Carlo on sufficient independent
trade history, missed-trade/delay tests and portfolio before/after results remain
unperformed. Outcome-level candidate diagnostics do not replace those studies.
The retained broker comment marker truncates client IDs and deserves a dedicated
collision/migration audit before live release. MT5 hedging/netting differences,
manual positions, stop/freeze constraints, sparse quotes, clock disagreement and
broker fill slippage also require terminal validation. Do not run competing workers
against one account. No live release gate or account setting was enabled here.
