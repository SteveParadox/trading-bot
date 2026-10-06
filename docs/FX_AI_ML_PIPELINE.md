# FX AI/ML research pipeline

## Safety boundary

The AI/ML subsystem evaluates strategy-generated candidates. It cannot create a
candidate, submit or close an MT5 order, size a position, alter a stop, or
change deterministic risk state. The execution worker owns those operations.

The live path is:

1. Read fresh MT5 quotes and closed candles.
2. Apply market-hours, data-freshness, news, spread, strategy, sniper, account,
   exposure, daily-loss, drawdown, and risk checks.
3. Persist the candidate and its point-in-time market snapshot.
4. Optionally run numerical models and the strict LLM deliberator.
5. Apply the configured AI mode.
6. Re-read execution-critical market, news, account, exposure, and risk state.
7. Submit through the MT5 adapter only if every deterministic check still
   passes.
8. Persist execution state and observe future candidate outcomes.

The invariant is that AI may reduce eligibility in advisory mode but cannot
expand it. A deterministic denial cannot become an execution because a model
returns TAKE. Revalidation immediately before submission also protects against
quotes, candles, news, exposure, account state, or risk changing while an AI
request is in flight.

## AI and ML modes

`FX_ML_PREDICTION_MODE` accepts `off`, `shadow`, or `required`. In shadow mode,
prediction failures are recorded without changing the deterministic result. In
required mode, an unavailable or invalid primary artifact rejects the
candidate. Auxiliary and entry-timing model failures are isolated and reported
separately.

`FX_AI_DELIBERATION` accepts `off`, `shadow`, or `advisory`:

- `off`: the LLM is not called.
- `shadow`: TAKE, WAIT, SKIP, timeout, provider failure, malformed output, and
  persistence failure are logged where possible and never block or authorize an
  otherwise deterministic decision.
- `advisory`: WAIT or SKIP may delay or reject a candidate according to policy.
  If confirmation is required, missing or invalid AI evidence fails closed.
  Advisory mode still cannot override a deterministic rejection.

Both modes default to `off`. SHADOW is the recommended development and data
collection mode. Configuring candidate-only model artifacts is rejected when
demo protection is disabled.

The live LLM response has exactly three fields:

```json
{
  "decision": "TAKE",
  "confidence": 0.82,
  "reason_codes": ["TREND_ALIGNED", "HIGH_TP_PROBABILITY"]
}
```

The provider schema rejects additional properties, unsupported decisions,
unsupported or duplicate reason codes, free-form prose, malformed JSON, and
confidence outside `[0, 1]`. Historic CONFIRM/FLAG/REJECT responses can be read
only by storage/evaluation compatibility code and are not accepted from a live
provider.

## Candidate snapshots and outcomes

Every serious strategy candidate has a stable unique ID and a durable row,
including rejected candidates. Candidate-time feature fields and the market
snapshot are immutable after insertion. Statuses distinguish generated,
deterministically rejected, risk accepted, eligible, delayed/advisory blocked,
executed, and execution failure states.

Snapshots use executable bid/ask prices and closed historical candles. They
record the strategy signal and score, entry/stop/target, spread and pip units,
ATR, RSI, momentum, trend strength, support/resistance distance, volatility,
session, portfolio/risk context, account currency/free margin, version data,
and structured news context. News context is informational to AI; the
deterministic news gate remains authoritative and required stale data blocks
entries according to configuration.

The outcome tracker observes prices after candidate creation. A long candidate
enters at ask and liquidates at bid; a short enters at bid and liquidates at
ask. It records future returns, entry improvements, MFE/MAE, TP/SL order and
times, time to profit/loss, and realized net P&L for executed candidates.
Restarted trackers process pending candidates oldest first. An observation that
arrives after a requested horizon cannot fabricate the earlier label; incomplete
or delayed paths are marked degraded.

## Training dataset v2

`fxbot.training_dataset` builds rows from immutable candidate-time features and
future-only outcome targets. Its manifest separates:

- identifiers and versions;
- `FEATURE_COLUMNS` used at inference;
- `TARGET_COLUMNS` learned from future observations;
- `AUXILIARY_OUTCOME_COLUMNS` used only for labeling and audit;
- `AUDIT_COLUMNS` describing execution/rejection.

Targets, execution outcomes, future returns, MFE/MAE, and AI decisions are not
features. The builder fails if feature and outcome/audit manifests overlap.

Entry-quality targets include immediate adverse movement within its configured
early window, executable expected pullback, continuation, nullable fake breakout
for genuine breakout candidates, and the fixed action order `ENTER_NOW`,
`WAIT_30S`, `WAIT_1M`, `WAIT_3M`, `SKIP`. A wait is not labeled when the
original entry already reached TP before that delay.

## Chronological splits and training

The default split policy never shuffles samples:

| Partition | Candidate timestamps |
| --- | --- |
| TRAIN | 2023-01-01 through 2024-12-31 |
| VALIDATION | 2025-01-01 through 2025-12-31 |
| TEST | 2026-01-01 through 2026-06-30 |
| FORWARD | 2026-07-01 onward |

The interval convention is `[start, end)`. A default 1,830-second purge removes
earlier-partition candidates whose 30-minute observation horizon could cross a
boundary. Recorded `label_end_timestamp` values provide an additional purge.
TRAIN alone is fitted; validation, test, and forward partitions remain
evaluation-only.

The baseline and five-class entry-timing trainers use deterministic shared
numeric/categorical preprocessing and XGBoost. Fake-breakout nulls are removed
instead of being coerced to class zero. The timing trainer requires every class
and a configurable minimum per-class TRAIN count (default 20). Artifacts are
candidate-only by default and are never automatically promoted.

Each artifact metadata file records target, model and feature-builder versions,
exact feature manifest, class order where applicable, split configuration,
dataset/label versions, and SHA-256. The loader checks all of these before
deserializing, verifies `predict_proba`, rejects corrupted or mismatched files,
and maps timing probabilities through the stored and estimator class order.

Example artifact configuration:

```env
FX_ML_PREDICTION_MODE=shadow
FX_ML_TARGET=TP_BEFORE_SL
FX_ML_MODEL_PATH=/path/to/model.joblib
FX_ML_MODEL_METADATA_PATH=/path/to/model.metadata.json
FX_ML_ENTRY_MODEL_PATH=/path/to/entry.joblib
FX_ML_ENTRY_MODEL_METADATA_PATH=/path/to/entry.metadata.json
FX_ML_VERIFY_MODEL_HASH=true

FX_AI_DELIBERATION=shadow
FX_AI_PROVIDER=openai_compatible
FX_AI_ENDPOINT=https://provider.example/v1/chat/completions
FX_AI_MODEL=provider-model-id
FX_AI_TIMEOUT_SECONDS=8
```

Keep `FX_AI_API_KEY` and broker credentials outside source control. `/api/config`
reports configured-state booleans and safe model names only; it does not expose
keys, credentials, database URLs, journal paths, or private artifact paths.

## Evaluation

`GET /api/ai-evaluation` and `fxbot.ai_evaluation` join one candidate, one
completed outcome, and one SHADOW deliberation. Reports compare candidate-level
baseline expectancy with a counterfactual AI TAKE filter and include TAKE,
WAIT, and SKIP attribution classes. Executable bid/ask returns already include
spread, so only configured non-spread costs are subtracted.

Candidate Sharpe and drawdown are explicitly proxies. Candidates may overlap,
and SHADOW never controlled execution, so these values are not realized
portfolio performance. Realized MT5 P&L is reported separately; unknown or
mixed currencies are not summed as if directly comparable. The report does not
establish profitability or future performance.

## Verification

Install both research requirement sets, then run:

```bash
python -m compileall -q fxbot
python -m pytest -q tests/test_fx_ai_evaluation.py tests/test_fx_prediction_service.py \
  tests/test_fx_ai_market_snapshot.py tests/test_fx_ai_deliberation.py \
  tests/test_fx_forward.py tests/test_fx_performance_safety.py \
  tests/test_fx_journal.py tests/test_fx_ml_baseline.py tests/test_fx_mt5.py \
  tests/test_fx_forexfactory.py tests/test_fx_news.py \
  tests/test_fx_news_policy.py tests/test_fx_strategy.py tests/test_fx_sniper.py
python -m pytest -q tests/test_fx*.py
```

The repository CI workflow runs compileall and the full FX test set on pull
requests and pushes to `main` and the AI feature branch.

## Known limitations

- Models remain research/candidate artifacts; there is no automatic promotion.
- The free Forex Factory calendar is unsuitable as the sole live safety source.
- Candidate-level evaluation is counterfactual and does not model a fully
  synchronized portfolio, overlapping signals, financing, or unknown costs.
- Sparse classes require more forward samples before timing-model research is
  meaningful.
- PostgreSQL schema evolution is additive at startup; production deployments
  should still use reviewed, backed-up migration procedures.
