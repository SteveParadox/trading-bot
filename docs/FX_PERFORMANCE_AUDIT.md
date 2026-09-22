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
- Candle freshness: `feed_decision_time` anchors decisions to the last candle
  instead of host UTC. Baseline entries have no equivalent of sniper enforce's
  candle-age guard. Fresh ticks do not prove fresh history. The helper's
  simulated-clock assumption is not the documented general MT5 contract:
  https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesfrom_py documents UTC.
  Capture host/tick/bar timestamps before changing this workaround.
- Clock monitoring compares host time with itself and cannot detect broker
  skew. A healthy clock indicator is currently not independent evidence.
- Any stale quoted instrument aborts the entire scan before trade management.
  Paused/stopped/halted workers also skip software management; broker-side
  SL/TP remain, but software breakeven/trailing/time exits do not run.
- A broker exception pauses the worker. Its paused loop does not attempt
  automatic reconnection. Check `broker_disconnected` and `missing_mt5_connection`.
- Independent TP/runner orders need validation on the actual account mode:
  the MT5 adapter has no explicit netting-versus-hedging gate. Do not assume
  two orders produce two independent positions on every broker account.
- Free calendar failure/staleness intentionally blocks entries with required
  news enabled. Earlier DNS/403 reports are known deployment blockers, not
  proof of an entry-strategy defect. Preserve fail-closed behavior.
- Small-profit performance must be measured after commission, spread,
  slippage and swap. Baseline breakeven's 0.2-pip buffer does not establish
  net breakeven for every broker. Cost assumptions require actual fills.

## Next evidence needed

Export recent signal rejection counts, order rejections, closed trades and
execution costs from the deployment, plus account currency/equity and broker
minimum lot, lot step and account mode. Do not include credentials.
Measure rejection proportions, fills, net expectancy, drawdown, MAE/MFE and
holding time by symbol/session before changing thresholds. Compare candidate
changes against the unchanged baseline on unseen periods and demo forward runs.
