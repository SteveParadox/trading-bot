# FX entry/exit audit and research design

Baseline: `b2d8e222b2d5a19edf994803cc919c585ffbbca8`, inspected 2026-09-21.
Scope: `fxbot`, its indicator dependency, FX tests, and the read-only research
agent. Legacy Bybit execution and its reports are excluded from FX inference.

## Existing decision pipeline

`ForwardTestWorker.scan_once` checks run/demo/release state, loads quotes and
instruments, checks quote freshness, reconciles orders, snapshots account risk,
syncs trades, manages stops/news, and persists daily-loss/drawdown halts. Each
instrument passes position, market-hours, rollover, session, news and absolute
spread restrictions before `_scan_instrument` runs.

`prepare_indicators` uses explicit MA7/14/28, ADX14/DI, ATR14 and optional tick
volume. `evaluate_signal_frame` selects closed entry/HTF bars, requires MA stack,
directional DI and price above/below MA7, HTF agreement/momentum, DI separation,
non-decreasing ADX, ATR bounds, optional candle close strength, MA28 slope,
MA7 extension, and minimum score. The score allocates 45 points to ADX, 35 to
DI, and optionally 20 to volume: without volume its maximum is 80, not 100.

The worker checks spread/ATR and executable-price deviation, builds an intent
at ask for buys/bid for sells, then `FxRiskManager` sizes an ATR or MA stop and
fixed-R target against equity, portfolio heat, pair/currency/gross exposure,
margin, lot steps and broker limits. An optional target/spread gate follows.
AI runs only after deterministic approval; off/shadow/advisory are implemented.
Orders use reservations and deterministic IDs, optionally split into TP1 and a
runner. MT5 runs order checks and order submission. Open trades get break-even
and ATR trailing updates, news protection, and broker SL/TP exits.

## Findings before implementation

| Priority | Finding | Consequence / proposed response |
|---|---|---|
| P0 | `Mt5Client.close_position` chooses ask for closing a long and bid for closing a short; it checks a partial request for filling mode | Use executable liquidation quote and check the complete close request |
| P0 | Break-even and trailing use the same original trade snapshot | A later trailing update can undo a just-tightened stop; update the local stop only after broker success |
| P1 | All symbols in a scan share the pre-order portfolio | Multiple approvals can consume the same budget; refresh account/positions and account for unresolved reservations before later entries |
| P1 | Feed decision time is derived from the last candle; it is not a candle freshness check | Stale/future bars require an independent check in enforcement; do not infer clock alignment from a candle itself |
| P1 | AI latency separates original quote/news approval from execution | Revalidate safety after an AI call when executing the new mode |
| P1 | MA stack is a setup state, not an entry event | Add an explicit closed-bar breakout or pullback/reclaim trigger |
| P1 | No local obstacle/structural invalidation model | Add historical extrema, room checks and optional structure stops without relaxing existing bounds |
| P1 | Spread-to-target check defaults to disabled; commission/slippage absent | Require explicit cost assumptions for enforcement; record gross opportunity, cost and ratio |
| P2 | Directional score families are correlated and location is outside the score | Preserve score for compatibility; critical stage failures override it; do not present score as win probability |
| P2 | No thesis-failure/time exit; TP1 defaults to 1.5R and runner may have no target | Add separately flagged closed-bar exits; compare small targets offline without reducing live minimum RR |
| P2 | BE/trailing activation uses the current stop, not immutable original risk | Persist initial stop/risk for new candidate trades and use it for exit decisions |
| P2 | Rejection payloads vary and no counterfactual execution evaluation exists | Persist staged snapshots; add strictly offline outcome analysis, including rejected candidates |
| P2 | Rationale document describes old pip bands and no live news feed | Current code/configuration is authoritative; document current defaults separately |
| Evidence | Checked-in price CSVs and performance reports concern crypto, not FX | No empirical parameter selection or claimed before/after FX return is possible from these files |

Existing news `allow` overrides are operator configuration; the new layer must
not set them or change deterministic policy. Tick volume is broker activity,
not consolidated FX volume. The news research module and research agent are
not a full replay of the FX worker, and their output is not execution parity.

## Proposed architecture (presented before strategy edits)

1. Existing data, session, news, portfolio and baseline setup gates remain.
2. **Regime / bias:** use the existing ADX, DI, MA slope and HTF requirements;
   distinguish expansion/compression against historical ATR, avoid unstable
   volatility. No extra oscillators or inferred order flow.
3. **Location / setup:** inspect only prior completed bars, local extrema,
   normalized MA distance and recent movement. Preserve the baseline setup.
4. **Trigger:** breakout of prior local high/low with directional close, or
   previous-bar pullback followed by directional reclaim. No retrospective pivots.
5. **Execution:** recheck current executable location, spread, slippage and
   round-trip commission assumptions, and sufficient distance to an obstacle.
6. **Risk / entry:** existing risk manager stays authoritative; optional
   structural stop is accepted only inside existing stop/RR bounds. AI cannot
   promote rejected candidates. Order reservations remain authoritative.
7. **Management / exit:** immutable entry context, sampled executable MAE/MFE,
   separately flagged thesis failure and stagnation exits; broker stops remain.

Mode `off` preserves strategy selection; `shadow` computes/persists supplemental
decisions without changing entry or exit behavior; `enforce` can only suppress
baseline signals, and permits explicitly configured exit/stop changes. New
thresholds are research hypotheses, not optimized values. Each gate is separately
switchable for ablations. No new production win-probability/EV estimate is fitted
without FX evidence.

## Evidence and safety limits

Software tests use synthetic fixtures solely for correctness. Real before/after,
parameter surfaces, session/pair conclusions, entry-delay selection, conditional
time-stop expectancy and Monte Carlo confidence require broker FX bid/ask paths,
costs, instrument specifications and time-aligned historical news coverage.
These must be evaluated chronologically with disjoint tuning and held-out windows.
Do not change sessions or pair thresholds based on the checked-in crypto data.

Small targets increase sensitivity to fees, slippage, gaps and occasional full
losses. Parameter combinations, overlapping trade outcomes and repeated selection
all increase overfitting risk. Demo fills can understate live costs. Sparse quote
sampling understates true intratrade extrema; news revisions can leak information.
No unrestricted live deployment or profitability claim follows from this work.

Primary execution references: [MT5 bars](https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesfrompos_py),
[order checks](https://www.mql5.com/en/docs/python_metatrader5/mt5ordercheck_py),
[broker stop/freeze properties](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants).
