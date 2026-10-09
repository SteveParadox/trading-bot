# PR #12 correction and verification report — 2026-10-09

## A. Repository findings

Repository: `SteveParadox/trading-bot`. PR: https://github.com/SteveParadox/trading-bot/pull/12.
Audited starting head: `118b9aefeb2684e8b16d06ecb6a1ab2c99ff29c2`.
Main/base: `ebaa4d4f10bcbbf3711e503495b8a036a66efb33`.
Branch: `codex/exit-ai-shadow-registry-20261009`. PR was open, draft and mergeable;
the accessible open-PR search found no superseding PR. Original PR changed 13 files,
adding 1,852 lines across 30 commits. The current correction commit is the commit
containing this report; its SHA is available in the PR commit list (avoids a
self-referential commit hash). No main-branch modifications or merge are authorized here.

The original implementation is a research foundation, not complete exit automation.
Entry path: market data → strategy candidate → causal candidate snapshot/journal →
entry ML → strict LLM deliberation → deterministic gates/risk revalidation → MT5
order/idempotent recovery → position reconciliation. Open-position management runs
sniper protection, breakeven and ATR trailing; news protection follows before new
entry risk. Broker history, trade reconciliation and candidate outcomes are existing
systems. The correction collects historical exit captures during position management,
finishes news protection, then dispatches broker-free observation to a bounded worker.
It does not replace the position manager, change entry orders, or insert training
into the scan loop.

| Feature | Implemented | Verified | Defect/action |
|---|---|---|---|
| Exit snapshots | Partial causal schema | BUY/SELL, JPY, freshness, quality, lifetime tests | Missing excursions now nullable; configured freshness; preserve origin context |
| Exit inference | Strict five-action shadow proposals | Hash/manifest, classes, malformed probabilities, cache tests | Exact class validation and model attribution; no bundled trained exit model |
| Exit policy | Pure validator only | Profit/stop/volume/gate regressions | Stronger cost, finite-number, causal ATR and action-state checks |
| Exit journaling | Existing event SQL/JSONL | Throttle/failure/API tests | Background isolation, immutable prediction events, bounded attempt state |
| Exit dataset | Sampled mark-return exporter | Horizon boundaries, missingness, identity, export hash tests | No fabricated optimal-action targets; revised v2 schema |
| Retraining | Off-engine entry challenger CLI | Real training, failures, repeats, locks | Eligible TRAIN fingerprints replace raw timestamp watermark |
| Registry | Candidate/approved/active/retired metadata | Approval, integrity, collision, rollback, lock tests | Family/target/feature checks; nested corruption and approval history checks |
| Version attribution | Entry/exit event traces | Prediction/cache/context tests and existing candidate tests | Preserve model hash on inference errors; origin strategy, timing/auxiliary metadata |
| MT5 protection | Existing deterministic owner | Existing FX suite + new sequencing tests | No AI execution adapter or real broker test |
| API monitoring | Authenticated read-only endpoint | Auth, query bounds, corruption, nested-field filtering | Filter before LIMIT; bounded allowlisted output; queue and loaded-model status |

## B. Confirmed bugs and corrections

No CRITICAL unauthorized-order path was found. Several HIGH research/integration
defects and latent MEDIUM policy defects were found. The pure policy was not wired
to MT5 before this audit; its defects did not demonstrate an actual unsafe order.
Every test named below ran successfully in the final full FX suite.

| Severity | File/function and root cause | Reproduction / fix | Regression evidence |
|---|---|---|---|
| HIGH | `exit_intelligence.build_exit_snapshot`: missing historical MFE/MAE replaced by current mark/zero | Original head reports MFE≈10, MAE=0 without telemetry. Now nullable with explicit unavailable quality; future telemetry excluded | `test_missing_excursions_are_unknown_and_future_telemetry_is_excluded` |
| HIGH | `forward._sync_open_trades`: synchronous model deserialization/inference and SQL between protective operations | A slow callback could delay the next position and news protection. Now collect first, complete protection, enqueue with 64-slot bound and one daemon worker | `test_original_position_operations_precede_optional_capture`, `test_scan_news_protection_completes_before_observation_dispatch`, `test_slow_optional_observer_is_bounded_and_submit_does_not_wait` |
| HIGH | `exit_observation_dataset.build_observed_exit_rows`: same ticket/opening time could mix different entry prices | Original head produced a 4-pip future label after changing entry price. Now separate lifetimes and verify symbol, direction, entry, pip and executable mark | `test_reused_ticket_with_changed_entry_and_stale_base_quote_do_not_mix`, `test_recovered_position_cannot_inherit_another_lifetimes_candidate` |
| HIGH | `retrain.train_challenger`: raw rows/maximum timestamp counted holdouts and incomplete targets; old completed labels missed | Now count fingerprints of fit-eligible, label-covered, purged TRAIN rows; old completed labels count, forward rows do not | `test_late_completed_label_is_new_but_forward_and_unknown_labels_are_not`, `test_real_training_registers_only_candidate_and_repeat_is_rejected` |
| MEDIUM | `exit_policy.evaluate_exit_policy`: caller-provided `estimated_net_pl` alone passed profit gate | Original head accepted unsupported value 1000. Require quote-matched cost breakdown and confirmed account-currency accounting; compute finite positive net amount | `test_gross_profit_and_unverified_partial_state_are_never_authority`, `test_verified_net_profit_exact_threshold`, existing profit test |
| MEDIUM | `exit_policy`: nonfinite broker constraints/configuration and unproven ATR/freeze state | Reject NaN/Infinity, unknown/future ATR, nonpositive stop, insufficient tightening and old stop in freeze zone | Parameterized bad-metadata/settings regressions; `test_freeze_zone_and_future_atr_are_rejected`; existing BUY/SELL tests |
| MEDIUM | `exit_policy`: arbitrary default volume rules and no pending-action evidence | Require actual broker min/step, valid grid, remaining minimum and verified hedging/netting/action state. Model supplies no closing volume | Existing partial-close test and pending-state regression |
| MEDIUM | `ExitPredictionService.predict`: integer coercion silently mapped fractional estimator labels | Compare actual classes to exact serving classes. Preserve loaded model version/hash on invalid inference, validate action/confidence distribution, reject successful predictions without attribution | `test_invalid_model_classes_and_probabilities_fail_closed_with_attribution`, `test_prediction_contract_rejects_failed_decisions_and_mismatched_confidence` |
| MEDIUM | `ModelRegistry.register/_read/status`: family/target/feature incompatibility, weak nested-schema checks and denylist output | Original head registered TP_BEFORE_SL as exit_management. Now enforce canonical manifests/targets, coherent pointers and approval history; allowlist bounded API output | `test_registry_rejects_cross_family_targets_and_corrupt_nested_records`, `test_registry_failed_activation_preserves_champion_and_hides_nested_private_fields` |
| MEDIUM | `forward._observe_exit_shadow`: failed writes retried every scan; old ticket context could be reused | Throttle attempts including failures, cap lifetime bookkeeping, validate recorded entry time/price before reading context | `test_failure_throttle_and_model_failure_cannot_call_broker`, recovered-lifetime regression |
| MEDIUM | `api.exit_ai_status` / exporter: global event LIMIT applied before event-type filter | 1,001 heartbeats previously hide exit history. SQL filters by event type first and limits result; allowlist nested observation schemas | `test_exit_api_auth_bounds_corrupt_registry_and_nested_secret_redaction` |
| MEDIUM | `FxBotSettings` / `ExitAiSettings`: direct non-demo constructor says shadow although observer disabled; nonfinite intervals accepted | Direct and environment non-demo configurations now resolve OFF. Integer intervals limited to 1..86400 | `test_non_demo_constructor_and_invalid_intervals_are_consistent` |
| LOW | `exit_observation_dataset._atomic_write`: platform newline translation can invalidate CSV hash on Windows | Write exact serialized bytes through `newline=""`, flush/fsync before replacement; duplicate resolution stable | `test_conflicting_duplicates_export_stably_and_hash_actual_bytes`; native Windows not executed |
| LOW | `retrain._atomic_json`: exception cleanup attempted unlink while Windows file remained open | Cleanup occurs after context closes; original state retained on serialization failure | `test_atomic_json_failure_cleans_temp_and_preserves_original`; native Windows not executed |
| LOW | Workflow push filter excluded exit feature branch | Push validation now covers `codex/**`; existing PR trigger retained | Workflow diff review; hosted execution separately blocked by billing |

Original-head reproductions were executed in an isolated checkout, not inferred
solely from new tests. Original output also confirmed non-demo direct settings
reported SHADOW. The unmodified original FX suite passed 702 tests before fixes;
this illustrates why the added regressions are necessary.

## C. Architectural oversights and release boundaries

Retained intentionally disabled capabilities: AI-controlled closing, trailing,
partial reduction and defensive liquidation; alternative-action counterfactual
simulation; trained/calibrated five-action exit model; independent champion trading
comparison; automatic promotion; adaptive rolling forward/live assimilation;
atomic runtime switching and independent forward validation. No speculative fills
or profitability claims have been added. Registry implements four metadata states,
not a complete REGISTERED→VALIDATED→SHADOW_TESTED→APPROVED release pipeline.

The observation worker is a bounded daemon thread, not a killable subprocess. A
stuck estimator remains in that optional thread; further captures are bounded and
excess captures drop. No hard inference cancellation, durable queue, exactly-once
restart semantics, or automatic SQL/artifact retention is claimed. Brief shared
database contention and CPU contention remain possible. Use OFF for deployments
requiring a strict proven timing SLA until process isolation is validated.
Interrupted exclusive lock files intentionally fail closed until an operator
confirms no job remains active. Failed registration can leave an inactive run
directory for diagnosis; it cannot advance sample state or change a champion.

## D. Files changed

Runtime: `fxbot/forward.py`, `fxbot/config.py`, `fxbot/api.py`, `fxbot/journal.py`.
AI/research: `fxbot/ai/exit_intelligence.py`, `exit_observer.py` (new),
`exit_observation_dataset.py`, `exit_policy.py`, `model_registry.py`, `fxbot/retrain.py`.
Tests: `test_fx_exit_intelligence.py`, `test_fx_exit_policy.py`,
`test_fx_model_registry.py`, `test_fx_exit_audit_regressions.py` (new),
`test_fx_retrain.py` (new). Documentation: this report and `FX_EXIT_AI_RESEARCH.md`.
Workflow: `.github/workflows/fx-validation.yml`.

No new runtime dependency, destructive database migration or broker-adapter edit.
Existing event_type/timestamp indexes support filtered bounded monitoring. A
dedicated decision/outcome table and retention policy remain future volume work.
Generated frontend build outputs, test databases, pycache and node_modules are
excluded from the correction commit.

## E. Checks actually executed

Environment: Linux, Python 3.12.14; dependencies installed from the two existing
FX requirements files. MT5 tests use fakes; no terminal or money account contacted.

| Command | Final result |
|---|---|
| `python -m pip install -r requirements-fx-research.txt -r requirements-fx-ml.txt` | Succeeded |
| `python -m compileall -q fxbot` | Exit 0 |
| `python -m pytest -q tests/test_fx_exit_intelligence.py tests/test_fx_exit_policy.py tests/test_fx_exit_observation_dataset.py tests/test_fx_model_registry.py` | 27 passed |
| `python -m pytest -q tests/test_fx*.py` | 770 passed, 20 warnings, 2 subtests passed |
| `npm ci && npm run build` (frontend directory) | Passed TypeScript/Vite build; large chart-chunk and dependency deprecation warnings |
| `git diff --check` | Passed |
| `git merge-base --is-ancestor origin/main HEAD` | Exit 0; main is ancestor |
| Side-by-side diff of MT5/risk/sniper/strategy/news/outcomes/dataset/split/trainers/entry predictor/loader against main | No changes in these modules |

No skipped tests or failing tests in the final FX run. Earlier development runs
had a test manifest-key typo and seven fixture/compatibility failures; corrected,
then rerun in full. Warnings: existing UTC deprecation, Starlette/httpx deprecation,
and existing entry-timing probability-sum metric warnings; no warning suppressed.
No repository-configured formatter, static analyzer or security scanner was found;
none is reported as executed. No native Windows, real broker, production scheduler,
actual model-comparison backtest or independent forward-demo validation was run.

Blocked-callback stress measurement: 1,000 queue submission attempts took 5.362 ms;
64 waiting captures accepted, 936 dropped, one already in progress. This verifies
bounded nonwaiting dispatch in this mocked environment, not a production latency SLA.

## F. Trading safety evidence

Existing `_manage_sniper_trade`, `_maybe_move_stop_to_breakeven`,
`_maybe_update_trailing_stop`, `_protect_positions_for_news` and the MT5 adapter
remain unchanged. New collection does not run inference inside these methods.
OFF performs no inference or new exit events and adds no broker operation.
Successful/uncertain sniper close attempts suppress capture; protective news
windows suppress stale pre-news capture. Historical observations do not assert a
position remained open while a model ran. Shutdown discards queued work and avoids
logging a late inference result after observer closure.

Concrete tests in the successful full run include original breakeven-not-undone-by-
trailing and ATR-ratchet tests, sniper failure/stagnation BUY/SELL tests, MT5 ticket,
lot and timestamp normalization, high-impact news restrictions, manual allow unable
to bypass emergency/stale-data gates, paused/halted protective management, stale
other-symbol protection, live-release gating, rejected-order recovery and entry AI
failures. New sequencing tests cover two-position OFF/SHADOW and close/no-close
paths plus position→news→optional dispatch ordering. DB/AI observer failures are
contained and cannot call broker methods. This is suite-level mocked evidence;
it does not independently certify every real broker/manual intervention scenario.
Existing global database failures outside the new observer can still affect the
baseline worker; this audit does not claim a wholly database-independent engine.

## G. Data/model integrity

Feature schema is now `exit-v2`; observed dataset is `exit-observed-v2`. Old events
remain readable through journal/API; incompatible v1 records/models are rejected
by the revised exporter/loader instead of inventing quality or missing history.
Executable liquidation remains BUY bid / SELL ask. Horizons 60/180/300 seconds
are nullable unless a genuinely later quote occurs at the horizon or within 20
seconds; base quote age also fits that lag. Features, targets and audit manifests
are disjoint; future labels never become features. Duplicate quotes resolve
deterministically and CSV SHA matches actual bytes.

Model SHA, ordered features/builder and exact probability classes are checked.
Trusted configured joblib files can be evaluated in SHADOW without approval, but
are explicitly labeled shadow artifacts. Hashing does not make pickle safe.
Already-loaded bytes remain pinned and attributed to the original version/hash
after disk replacement; a restarted loader rejects corrupt files. Registry pointer
changes are not model deployment. Numerical prompt versions remain null/not applicable.
Entry traces now include base, timing and auxiliary model attribution; exit traces
retain feature-builder/model metadata even for a failed inference after loading.
Origin strategy/candidate context is checked against the captured position lifetime.

## H. Retraining verification

Real XGBoost fixture training successfully produced TRAIN-only candidates with
12 train, 6 validation and 6 test rows. Repeating unchanged input is rejected;
one late completed old label triggers a new run; added FORWARD samples do not.
Missing/empty CSV, duplicate IDs, invalid target, missing validation/test or label
coverage, concurrent locks, training/registration failures and JSON-write failures
are tested. Failed jobs leave the champion manifest and sample state unchanged.
Artifacts have unique directories, hashes, metadata and metrics corresponding to
the fitted model. Fixed split purging and observed label ends remain authoritative.
Current dates deliberately exclude present forward data from fitting. This is a
repeatable separate-process CLI, not a deployed automatic rolling training service.
Challenger comparison remains `not_evaluated`; all candidates need manual review.

## I. Remaining limitations

No reliable optimal-action exit labels or trained exit artifact; no runtime net-cost
adapter; no AI partial-close/stop/exit executor or independent action coordinator;
no comparable cost-aware champion/challenger trading metrics or forward evidence;
no runtime atomic model swap; no durable observation queue/exact-once collection;
no installed scheduler or automatic storage-retention service. These are explicit
release blockers, not features enabled to make tests pass. No profit improvement
or production-readiness claim is made.

## J. PR/CI status and operator action

Retain draft PR #12 on its existing branch. The correction commit is available in
the PR commits; main has no known divergence/conflict at audit time. No merge.
GitHub hosted CI is blocked, not a demonstrated Python test failure. Original run:
https://github.com/SteveParadox/trading-bot/actions/runs/37891482174
Job/check ID `113693163010` had no job steps. Its fetched annotation says exactly:

> The job was not started because your account is locked due to a billing issue.

Operator action: resolve the GitHub account billing lock, then rerun FX Python
validation on the current correction head. A second annotation notices the future
ubuntu-latest migration; it is not the reported failure cause. Workflow push scope
now includes this feature branch. Local results are verified; hosted CI cannot be
declared successful while the account lock remains. No merge readiness until CI
executes successfully and a reviewer accepts the limited SHADOW research scope.
Advisory exit release separately requires the unimplemented safety/value gates.
