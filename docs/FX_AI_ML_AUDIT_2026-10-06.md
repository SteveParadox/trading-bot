# FX AI/ML implementation audit — 2026-10-06

## Scope and pull-request dependency map

GitHub and repository history were inspected rather than relying on PR text.
The default branch and PR target are `main`. The only open PR related to market
snapshots, candidate outcomes, ML training, entry timing, LLM deliberation, or
AI evaluation was PR #10, `codex/ai-market-snapshot` into `main`. It was 147
commits ahead and zero behind the inspected `main` head, had no merge conflict,
and had no dependency on another open PR. No duplicate or superseding open PR
was found, so no other PR should be merged or closed as part of this audit.

## Material findings and corrections

### Critical

- `fxbot/forward.py`: execution-critical checks were not uniformly repeated
  after AI latency in every execution mode. A universal pre-submit revalidation
  now checks bot state, unresolved orders, demo/live release, quote and candle
  freshness, news, account/MT5 health, exposure, daily loss/drawdown, risk,
  spread/cost, TP-candle validity, and sniper rules where enforced. Exceptions
  fail closed.
- `fxbot/risk.py` and `fxbot/forward.py`: hedged and external broker positions
  could understate gross exposure or omit unknown stop risk. Fresh broker state
  now includes gross pair exposure and fails closed for unpriced or unprotected
  external positions and invalid portfolio values.

### High

- `fxbot/journal.py` and `fxbot/training_dataset.py`: candidate-time fields could
  be overwritten after creation, allowing hindsight contamination. Snapshot,
  trade, signal, score, and version inputs are immutable and duplicate stable
  candidate scans are rejected.
- `fxbot/outcome_tracker.py`: restart/late sampling could associate a late quote
  with an earlier fixed horizon. Pending outcomes are processed oldest first;
  late horizons are marked incomplete/degraded rather than fabricated.
- `fxbot/chronological_split.py`: boundary samples lacked a full label-horizon
  purge. Split v2 applies a 1,830-second purge and recorded label-end timestamps.
- `fxbot/baseline_model.py`, `fxbot/entry_timing_model.py`, and
  `fxbot/ai/preprocessing.py`: training and inference preprocessing could drift,
  and empty/all-missing columns were unsafe. Both paths now share deterministic
  preprocessing. Timing training enforces configurable minimum class support.
- `fxbot/ai/model_loader.py` and `fxbot/ai/predictor.py`: artifact compatibility,
  tampering, estimator API, and timing class order were not all enforced. The
  loader verifies hashes and manifests before loading; inference checks finite,
  normalized probabilities and stable class mapping. Optional model failures
  remain isolated.
- `fxbot/ai_deliberation.py`: duplicate JSON object keys could survive ordinary
  parsing. Strict decoding now rejects duplicates, extra fields, malformed
  output, legacy live decisions, invalid confidence, and unsupported reason
  codes.

### Medium

- `fxbot/market_snapshot.py`: choosing only the nearest news item could hide a
  nearby high-impact event behind a closer low-impact event. Structured context
  now evaluates all relevant nearby events and preserves exact release timing.
- `fxbot/ai_evaluation.py`: evaluations were not explicitly limited to SHADOW
  rows and cost/currency comparability required stronger handling. Queries are
  bounded and shadow-only; nonfinite returns and unknown costs are excluded;
  mixed/unknown P&L currencies are noncomparable; spread is not double counted.
- `fxbot/mt5.py`: split closing deals could produce premature realized outcome
  attribution. P&L is emitted only after all legs close and includes opening and
  closing commissions, fees, and swap.
- `fxbot/api.py` and `fxbot/security.py`: sanitized configuration still exposed
  private local database/journal paths. The API now reports only safe metadata
  and configured-state booleans.
- `fxbot/database.py`: additive migration helpers were SQLite-specific. Column
  inspection and additive changes now support the configured SQLAlchemy backend
  without destructive data changes.

### Low / informational

- `fxbot/forward.py`: code-version hashing omitted AI subpackage source. It now
  hashes all `fxbot/**/*.py` files.
- `.github/workflows/fx-validation.yml`: the PR had no Python validation job. A
  workflow now compiles `fxbot` and runs every `tests/test_fx*.py` test.
- `docs/FX_AI_ML_PIPELINE.md`: added source-aligned architecture, modes, dataset,
  splitting, artifact, evaluation, limitation, and verification documentation.

## Verified invariants

- AI code has no MT5 submission, closing, sizing, or risk-mutation dependency.
- A deterministic hard denial cannot be converted to execution by ML or LLM.
- SHADOW TAKE, WAIT, SKIP, timeout, unavailable provider, malformed output, and
  persistence failure do not change deterministic execution authority.
- Advisory output can only delay or reject; it cannot create/reverse a trade,
  increase units, loosen a stop, bypass news/risk/exposure, or enable live mode.
- Candidate features are point-in-time; targets and auxiliary outcome/audit
  fields are disjoint from `FEATURE_COLUMNS`.
- Long/short outcome prices are direction-aware and fixed-horizon labels are
  future-only.
- Chronological partitions are ordered, disjoint, unshuffled, and purged.
- Artifacts default to candidate-only and are never automatically promoted.
- API research/evaluation routes are authenticated, bounded, and read-only.
- Defaults remain ML off, AI off, demo-only on, and live trading disabled.

## Validation record

The completed correction tree produced these results:

| Command | Result |
| --- | --- |
| Requested 14-file focused pytest command | 305 passed, 2 subtests passed, 2 warnings |
| `python -m pytest -q tests/test_fx*.py` | 672 passed, 2 subtests passed, 20 warnings |
| `python -m compileall -q fxbot` | passed |
| `git diff --check` | passed |
| `npm ci && npm run build` in `frontend` | passed; TypeScript and Vite compiled 2,210 modules |
| Clean-environment safety-default assertion | ML off, AI off, demo-only true, live trading false, live release false |
| `python -m pytest -q` | collection stopped on pre-existing missing `backtester.bybit_data` imported by `tests/test_bybit_data.py` |

There is no configured Python formatter, linter, or type-checker command in the
repository. The frontend `build` script runs `tsc` before Vite. The optional
whole-repository failure is outside the FX implementation: `main` and the
feature branch both track the Bybit test but not the module it imports. It does
not affect the requested FX suites or their 672 passing tests.

At the time of audit, GitHub reported two infrastructure failures unrelated to
the source validation: Vercel rejected the deployment because its build-rate
limit was reached, and the GitHub Advanced Security job did not start because
the account was locked for a billing issue. The PR had no Python workflow before
this correction; `.github/workflows/fx-validation.yml` adds one. These external
statuses are not treated as substitutes for the successful local Python gate.
