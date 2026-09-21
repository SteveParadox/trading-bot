# Implementation validation — 2026-09-21

Baseline commit: `b2d8e222b2d5a19edf994803cc919c585ffbbca8`.

## Software evidence

- Untouched baseline, FX suites plus indicator suite: **449 passed**.
- Modified branch, FX suites, indicator suite, implementation and research-module
  suites: **547 passed**, with 17 pre-existing `datetime.utcnow()` deprecation
  warnings from research-agent validation. This broader command includes 39
  new sniper/research test cases; counts alone are not strategy-performance evidence.
- `git diff --check` and module compilation passed.
- Offline CLI smoke test on a labelled synthetic fixture produced 60 experiment
  combinations with both source hashes. The fixture/results are not FX evidence.

Reproduce the changed-branch verification:

```bash
python -m pip install -r requirements-fx-research.txt
python -m pytest tests/test_fx_*.py tests/test_indicators.py tests/test_imp_implementation.py tests/test_new_modules.py -q
```

Tests cover long/short qualification, malformed data, each critical rejection
family, trigger versus persistent trend, structural obstacles, spread/fee/slippage
accounting without double-counting spread, stop/target constraints, cost-inclusive
risk sizing, partial-order position slots, off/shadow order equivalence, enforcement
suppression before AI in all AI modes, failure/stagnation exits, exit persistence
and restart deduplication, stale/news revalidation, executable close-side pricing,
stop tightening, order recovery, closed-trigger deduplication, quote-path gap
losses, target capping, cost stress, split purging, missing coverage and CLI export.

## Before/after behavior

| Area | Baseline | Changed branch |
|---|---|---|
| Setup | MA/ADX/DI/HTF plus slope/extension/score | Retained |
| Entry event | No distinct event gate | Optional breakout or pullback/reclaim |
| Location | MA extension | Optional executable extension, prior structure and room |
| Cost gate | Optional spread-only target ratio | Supplemental known round-trip cost / gross move |
| Stop sizing | ATR or MA, price-distance risk | Optional structure stop; enforced estimated costs included in sizing |
| Thesis/time exits | No corresponding implementation | Separate flags, enforce-owned trades only |
| Quote for market close | Wrong quote side | Bid for closing long, ask for closing short |
| BE then trailing | Same stale stop snapshot | Successful updates ratchet local snapshot |
| Multiple new setups | Reused scan portfolio | One newly submitted setup per fresh account scan |
| Closed signal | Could submit again on same candle | Closed ID remains terminal for entry deduplication |
| Research | No supplemental counterfactual journal study | Candidate diagnostics with conservative quote coverage |

## Performance evidence

| Requested result | Status |
|---|---|
| Actual FX before/after trade count, win rate, PF, expectancy, profit | Unavailable: no matching FX history/database supplied |
| Actual MAE/MFE, holding-time, entry-delay comparison | Unavailable; instrumentation and quote diagnostics added |
| Actual parameter stability surface | Unavailable; diagnostic extension/target/cost scenarios available |
| Portfolio drawdown, Sharpe/Sortino, recovery | Unavailable; candidate outcomes are not portfolio trades |
| Pair/session optimal thresholds | Not selected; no evidence justifies changing them |
| Full rolling walk-forward and Monte Carlo | Not performed; requires complete FX execution replay and sufficient trades |
| Live MT5 terminal/broker behavior | Not tested in this environment |

This is a tested first research implementation, not completion of empirical
tuning. No new parameter is described as statistically optimal. Existing crypto
data was excluded. The safe recommendation is demo **shadow observation** with
verified calendar and broker-specific costs, followed by the validation work in
[the research guide](FX_SNIPER_RESEARCH.md). Do not promote this draft to live
trading on the strength of synthetic unit tests.
