# MT5 performance audit — 2026-09-22

Baseline: main e0621169cf5c58f8f57b56910ef201babe4a5bd6.
Scope: FX strategy, risk sizing, worker lifecycle, order splitting, exits,
news and dashboard configuration. No live orders or strategy optimization.
No production journal, broker specification export or recent trade history
was available for attribution of actual returns. Legacy Bybit performance
is excluded.

## Corrected defects

1. `strategy.py`: a standalone minimum or maximum ATR-in-pips setting was
   ignored unless both were supplied. Each bound now applies independently.
2. `risk.py`: the same defect affected stop-distance limits. A maximum-only
   stop constraint could silently allow excessive distances; fixed for both sides.
3. `forward.py`: splitting a valid order could create positive-sized legs
   smaller than the broker minimum when volume step is below volume minimum.
   Such orders now use one full-size leg with the planned take-profit.
4. `config.py`: the generic CSV reader uppercased additional CORS origins.
   Browser origin matching is exact; these values now retain their case.
5. Transient MT5 failures now retry with exponential backoff capped at 300
   seconds. Connection status no longer overwrites operator pause/halt state.
   Missing SDK or a refused real account still pauses entry permission.
6. Paused and halted workers reconcile trades and manage software exits using
   fresh quotes. STOPPED still stops all worker activity. A stale quote for
   another configured symbol blocks new risk but does not disable management
   of positions with fresh quotes. Broker SL/TP remain essential during outages.
7. Both live entry timeframes must have recent, closed UTC bars; future,
   duplicated, unordered, empty and timezone-naive history is rejected.
   Trailing stops also require current history. Strategy evaluation uses actual
   UTC, while order identity uses signal-candle close time to prevent re-entry
   on the same trigger. Future ticks are no longer treated as fresh.
8. Clock health now compares host time with an actual tick timestamp, rather
   than comparing host time with itself. Quote lag can also mean market inactivity.
9. Split exits require confirmed MT5 hedging mode. Netting or unknown account
   mode uses a single order with the planned take-profit.
10. `FX_EXECUTION_COST_PIPS_ROUND_TRIP` adds commission and slippage allowance
    to risk sizing and breakeven protection. Targets fully consumed by the
    allowance are rejected. Breakeven/news stop modifications must respect the
    broker's minimum distance and never loosen an existing stop.
11. Minimum-volume rejections now report the binding cap, all unit limits,
    broker minimum units and minimum-lot risk/notional in account currency.
    Orders are never rounded upward past risk limits. Programmatic exposure
    defaults now match the existing environment defaults (0.35/1.20/0.70).
12. Submission rechecks RUNNING before each exit leg so a pause during signal
    evaluation cannot bypass entry permission. An already in-flight broker
    request cannot be recalled by pausing the worker.

## Performance constraints and remaining risks

- Small accounts: default pair exposure is 0.35 times equity, measured in
  notional value, not margin. At EUR/USD 1.10 and a broker minimum of 1,000
  units, minimum notional is USD 1,100; this cap alone requires about USD
  3,143 equity. A USD 100 account would be limited to about 31 units before
  lot rounding and would receive `units_below_minimum`. Raising leverage
  does not remove this cap. Do not round orders upward past risk limits.
- Default strategy is 15-minute trend continuation with one-hour confirmation,
  a 1.8 ATR stop and 1.5R first target. It is not a fixed-dollar scalper.
  Half the order may remain as a runner without a fixed TP when splitting
  is possible. Frequent USD 1–2 exits are not an implemented objective.
- With volume confirmation disabled, quality score has a maximum of 80
  points (45 ADX + 35 DI), yet the default threshold is 60. This strongly
  selects mature trends. Changing the scale/threshold needs out-of-sample
  evidence; it has not been silently relaxed.
- The 1.5-pip entry-deviation limit includes the difference between executable
  ask/bid and the signal close. It can reject trades even below the separate
  3-pip spread ceiling. Review `entry_deviation_filter` frequencies before tuning.
- MT5 documents UTC bar/tick timestamps:
  https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesfrom_py.
  If the new freshness check blocks the deployment, correct host/terminal
  timestamps and history synchronization rather than shifting decision time.
- Account-mode behavior follows MT5's netting/hedging distinction:
  https://www.metatrader5.com/en/terminal/help/trading/general_concept.
  Broker acceptance, fill quality, freeze levels and account-specific contract
  specifications still require a demo smoke test on the actual deployment.
- Free calendar failure/staleness intentionally blocks entries with required
  news enabled. Earlier DNS/403 reports are known deployment blockers, not
  proof of an entry-strategy defect. Preserve fail-closed behavior.
- Small-profit performance must be measured after commission, spread,
  slippage and swap. The configurable cost allowance defaults to zero for
  compatibility; this is not a claim that execution is free. Stops can slip,
  so an estimated breakeven level does not guarantee a net-zero exit.

## Deployment and cost configuration

After merging, pull main on Lightsail and restart the Python worker/API.
No frontend deployment or ngrok restart is needed for these worker changes.
An existing operator/error pause remains paused; after verifying MT5 health,
use Start to re-enable entries. Do not run two workers against one account.

Keep demo-only, live-release restrictions and required news enabled. Set in
the backend environment, using measured round-trip commission plus adverse
slippage converted to pips:

```dotenv
FX_EXECUTION_COST_PIPS_ROUND_TRIP=0.0
FX_BREAKEVEN_BUFFER_PIPS=0.2
```

Replace 0.0 with an evidence-based allowance before evaluating net performance.
Do not add the quoted spread again: entries use executable ask/bid and exits
use the liquidation side. Swap is separate. This global allowance is a coarse
budget across symbols; mixed instruments and account currencies need careful
calibration. Sniper enforce uses the larger of this allowance and its own
commission/slippage allowance for sizing, not their sum.

Entry thresholds, leverage, daily loss/drawdown limits and target multiples
were not relaxed to manufacture more trades. A broker/account whose minimum
lot exceeds the risk budget remains incompatible at that equity. Tune only
after cost-inclusive out-of-sample and demo comparisons establish improvement.

## Next evidence needed

Validation after the reliability fixes: 600 FX/config/indicator tests passed,
including 23 new performance-safety cases, plus 2 subtests. There are 17
pre-existing datetime deprecation warnings. Full legacy test collection still
requires the missing `backtester.bybit_data` module. No Windows MT5 terminal,
live fills or Lightsail deployment were exercised by these offline tests.

Export recent signal rejection counts, order rejections, closed trades and
execution costs from the deployment, plus account currency/equity and broker
minimum lot, lot step and account mode. Do not include credentials.
Measure rejection proportions, fills, net expectancy, drawdown, MAE/MFE and
holding time by symbol/session before changing thresholds. Compare candidate
changes against the unchanged baseline on unseen periods and demo forward runs.
