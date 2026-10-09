# Exit intelligence and challenger retraining: first safe implementation slice

**Status (2026-10-09): Research-only, review required. NOT a production AI exit release.**

This implementation deliberately makes the smallest broker-safe change possible.
It **does not** claim completion of the broader AI-assisted exit / adaptive MLOps
initiative. In particular, it cannot issue AI-initiated MT5 closing or stop
modifications. That requires independently validated counterfactual exit labels,
an approved challenger, a locked execution coordinator, fresh broker-position
revalidation, and forward demo evidence.

## Audited trading path

The existing forward worker still follows:

MT5 prices / closed candles -> strategy candidate -> candidate journal ->
ML entry scores -> bounded LLM TAKE/WAIT/SKIP -> deterministic risk recheck ->
MT5 entry -> `_sync_open_trades` -> sniper failure/time exits -> breakeven and
ATR trailing stop -> broker close/history reconciliation -> trade journal and
candidate outcome tracker.

This change adds an observation **after** the existing deterministic protection
functions in `_sync_open_trades`. That observation runs in a try/catch
boundary and has no reference to the MT5 order adapter. Unavailable models and
storage errors cannot authorize an exit or suppress the existing position
manager. `FX_EXIT_AI_MODE=off` returns to original behavior. Existing entry
AI / ML and live trading modes are untouched.

## Delivered components

- `fxbot/ai/exit_intelligence.py`: strict five-action taxonomy, causal
  position snapshots, stable feature manifest, SHA-256 and feature-manifest
  validated shadow predictor, structured version-tagged predictions. A missing
  model is `unavailable`, never a fabricated HOLD.
- `fxbot/ai/exit_policy.py`: **pure**, non-executing eligibility checks for
  TAKE_PROFIT_NOW, HOLD, TRAIL_STOP, REDUCE_POSITION and EXIT. It blocks
  unverified net profit, stop widening, missing broker lot sizes, and unapproved
  defensive exits. An actual advisory execution coordinator is **not wired**.
- `fxbot/forward.py`: journal `exit_ai_observation` (or a failure event)
  through the existing SQL + JSONL event infrastructure, at most once per
  configured evaluation interval per position per worker process.
- `fxbot/ai/model_registry.py`: append-only audit history in an atomically
  replaced JSON manifest, immutable model identifiers, SHA-256 checks, explicit
  reviewer approval, candidate versus active metadata pointers and rollback.
  Registry active pointers do **not** silently reconfigure running model
  instances. Regulated live execution stays separately blocked.
- `fxbot/retrain.py`: separate-process entry-quality challenger training
  from a previously exported v2 CSV. Uses existing chronological splitter,
  XGBoost baseline, per-job unique artifact directories, an exclusive
  training lock, a minimum-new-record gate, candidate registration and
  validation/test reports. **Never auto-promotes**.
- `GET /api/exit-ai`: API-key-authenticated read-only model/registry state and
  recent exit observations. It reports execution as disabled.
- Focused test suites: `test_fx_exit_intelligence.py`,
  `test_fx_exit_policy.py`, and `test_fx_model_registry.py`.

No new database table was necessary for this slice: existing `event_log` JSON
payloads preserve every observed position decision. A dedicated indexed exit
decision/outcome table remains future work for high-volume studies.

## Snapshot semantics and limitations

At observation time the worker uses a fresh executable quote (bid to
liquidate a BUY, ask to liquidate a SELL), the broker position ticket,
opening time/price, direction, signed broker units, SL/TP, holding time,
unrealized broker P&L, and past sampled MFE/MAE where available. These
excursions are **sampling lower bounds**, not full intratrade tick paths.
Unsupported news/ATR/RSI/momentum/partial-close fields remain null and cannot
be silently filled from future market observations.

The snapshot deliberately does **not** equate broker unrealized gross P&L
with realized net P&L. Unknown exit commission, slippage, financing, conversion,
and account-currency accounting mean `estimated_net_pl=null`. The pure
TAKE_PROFIT_NOW policy therefore rejects that action rather than inventing
profitable fills.

A trained exit artifact would need: `target=EXIT_ACTION`,
`feature_builder_version=exit-v1`, the precise
`EXIT_FEATURE_COLUMNS` in source order, the complete five-class
`EXIT_ACTIONS` order, `model_version`, and `model_sha256`. No trustworthy
exit training artifact is bundled and no strategy backtest performance claims
are made. Never deserialize untrusted external model files.

## Configuration

```dotenv
# Demo accounts: shadow by default. Non-demo accounts: off by default.
FX_EXIT_AI_MODE=shadow
FX_EXIT_EVALUATION_INTERVAL_SECONDS=60
FX_EXIT_MODEL_PATH=
FX_EXIT_MODEL_METADATA_PATH=
FX_EXIT_VERIFY_MODEL_HASH=true
FX_MODEL_REGISTRY_PATH=data/models/registry
```

Only `off` and `shadow` are accepted. `advisory` fails configuration
validation because safe MT5 modifications and demonstrated model value are not
yet implemented. The pure policy's `advisory_released` argument defaults
to false, and the worker never calls it to execute.

For existing behavior, set `FX_EXIT_AI_MODE=off`. MT5 demo protection,
existing stop-loss behavior, risk limits, news safeguards, and entry LLM
configuration are unchanged.

## Periodic training (entry-quality only)

Prerequisites: install `requirements-fx-research.txt` plus
`requirements-fx-ml.txt`, then export a fresh v2 causal training CSV using
the repository's `fxbot.training_dataset` / historical reconstruction path.
Keep research CSV files and model artifacts out of version control.

Example from the repository root:

```powershell
python -m fxbot.retrain --dataset data/training/fx_4pair_test_v2.csv --artifacts-root data/models/challengers --registry-root data/models/registry --target TP_BEFORE_SL --min-new-samples 100
```

Default split boundaries are train 2023-2024, validation 2025, test January
through June 2026, forward July 2026 onward, subject to the actual dataset
having usable examples in every required training/evaluation partition. The
existing split logic purges overlapping label horizons. The task is repeatable
but **does not** manufacture sufficient samples or interpolate missing future
labels. Dataset timestamps and candidate IDs are validated; new rows are
counted relative to the previous registered job watermark.

Use Windows Task Scheduler or cron to call the command in a separate low-priority
process, after your export pipeline has completed. For example a monthly cron
entry (adapt to your server environment):

```cron
10 3 1 * * cd /srv/trading-bot && /srv/trading-bot/.venv/bin/python -m fxbot.retrain --dataset data/training/exported_v2.csv >> /var/log/fx-retrain.log 2>&1
```

There is no embedded training scheduler inside the MT5 worker. A lock prevents
concurrent retraining; after an interrupted process leaves a lock, inspect
running jobs before manually removing `retraining.lock`.

## Registry approval and rollback

Registry register creates a **candidate**. An operator can inspect the
candidate through `ModelRegistry.status()`, review the reports and approve it
with `ModelRegistry.approve(model_id, approver=..., evidence=...)`.
`activate(model_id)` switches a local registry metadata pointer after hash
verification. `rollback(model_type, previous_model_id)` restores a
previously approved artifact pointer.

**Do not confuse a metadata pointer with runtime activation.** The current
entry-serving `VersionedModelLoader` still reads its configured local paths
and caches the artifact. Repointing it requires a separately controlled
deployment/restart following normal verification. Exit observation similarly
uses configured paths and is shadow-only.

Training results include validation and test metrics but no independent
champion-vs-challenger trading backtest. The report explicitly marks that
comparison `not_evaluated` and promotion `manual_review_required`.

## Tests

```powershell
python -m compileall -q fxbot
python -m pytest -q tests/test_fx_exit_intelligence.py tests/test_fx_exit_policy.py tests/test_fx_model_registry.py
python -m pytest -q tests/test_fx*.py
```

MT5 broker calls require a separate mocked or demo environment. This PR
does not authorize execution testing against a real-money account.

## Required future work before production advisory exits

1. Construct a dedicated exit-decision dataset with linked original positions,
   repeated time-stamped observations, future-only TP/SL outcomes and
   realistic, feasible alternative-action replay. Model unknown counterfactuals
   as unknown; address incomplete quote paths and entry-selection bias.
2. Train and calibrate an exit model only when class support and genuine
   out-of-sample performance justify it. Compare identical-entry baseline,
   quick-profit rules, trailing, reduction, ML and (if useful) bounded LLM
   deliberation under realistic broker costs.
3. Build independently locked, idempotent MT5 position action coordinator:
   re-fetch ticket, account, signed units, broker min volume/step, freeze
   levels, SL/TP, quotes, and broker ack; distinguish hedging/netting; preserve
   all deterministic emergency exits.
4. Add a dedicated persisted action lifecycle, pending/unknown ack recovery,
   coherent realized/partial P&L attribution, exit outcome tracker, historical
   reconstruction and independent shadow-vs-baseline reporting.
5. Produce a true reusable rolling/expanding data policy, minimum sample
   coverage per action/regime, walk-forward purging, champion comparisons,
   tested model hot-switch and full API/UI approval workflow.
6. Demonstrate measurable net expectancy and controlled drawdown improvements
   in forward demo runs before **explicitly** implementing and enabling
   `FX_EXIT_AI_MODE=advisory`.

**Acceptance boundary:** candidate prediction infrastructure exists; approved
broker-executable exit intelligence does not. This PR should remain in review
until CI and regression checks pass, and must not be represented as delivering
all 29 phases of the requested feature.
