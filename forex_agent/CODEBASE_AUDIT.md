# Forex Agent — Codebase Audit

Audit scope: the `forex_agent/` package (data ingestion, agent/analyst, analysis,
storage) plus the surrounding `fxbot/`, `backtester/`, `bot.py`, `config.py`,
`indicators.py` to the extent that duplication crosses package boundaries.

Audit was performed **before** the cleanup pass and is recorded here as the
verified findings that drive this cleanup. Quantitative behaviour is preserved
(see `CLEANUP_REPORT.md` for the verification trail).

---

## 1. Duplicated computation

### 1.1 R-multiple (Risk-per-trade) — FIXED
`forex_agent/agent/analyst.py` and `forex_agent/agent/hypotheses.py` each
implemented a private `_r_multiple(trade)` helper that computed the same
R-multiple formula in isolation. Two independent copies of the same financial
formula are a drift hazard: a fix to one copy silently diverges from the other.

- `analyst.py`: `_r_multiple` + `_get_r_values` (private helpers, now removed).
- `hypotheses.py`: `_r_multiple` + `_get_r_values` (duplicate, now removed).

**Resolution:** introduced a single batched helper
`forex_agent/data/ingestion.py::compute_r_values(trades)`; both agents now wrap
it. The canonical per-trade `compute_r_multiple(trade)` already lived in
`ingestion.py`; the batched helper delegates to it and applies the same
open-trade / zero-risk handling.

> Behaviour note: `_r_multiple` returned `0.0` for an *open* trade; the new
> `compute_r_values` excludes open (and zero-risk) trades from the returned
> list. In practice this is behaviour-identical because both call sites always
> filtered to `closed` trades before building the R list. Locked with
> `TestComputeRValues` regression tests.

### 1.2 Expectancy / profit factor / drawdown
These metrics are computed centrally in `analysis/performance.py` and wrapped by
`TradeAnalyst._compute_metrics`. No meaningful duplication of the *formulas*
exists inside the agent; the duplication is conceptual: the strategy-health
score in `analyst.py` recalculates per-component scores from the same inputs.
This was not consolidated (would risk changing behaviour) but is documented as a
candidate for a single shared metrics layer.

### 1.3 Session / day-of-week extraction — cross-package (documented only)
Within `forex_agent`, session and day-of-week extraction is centralised in
`ingestion.py::extract_session` / `extract_day_of_week` and reused by every
caller. However, near-identical session-window logic exists in the **outer**
packages (`fxbot/`, `backtester/`, `indicators.py`). These are separate concerns
(live trading vs. backtesting) and were **not** merged, to avoid destabilising
the larger system. Documented here as technical debt.

---

## 2. Dead code & unused imports — FIXED

Removed:
- `forex_agent/agent/analyst.py`: dead `_row_to_trade_record` method (unused),
  unused `import statistics`, `import math`; restored needed `from typing import
  Any`.
- `forex_agent/data/schemas.py`: unused `import sys`.
- `forex_agent/analysis/anomaly_detection.py`: unused import of
  `extract_day_of_week`.
- `forex_agent/analysis/performance.py`: unused `import math`.
- `forex_agent/agent/reporting.py`: unused `AnalysisReport` import.
- `forex_agent/storage/database.py`: unused `import json` at module scope.

### 2.1 The `analysis/` package — kept (documented)
`forex_agent/analysis/` contains: `performance.py`, `risk.py`, `statistics.py`,
`regime.py`, `execution.py`, `mae_mfe.py`, `trade_forensics.py`, `alerts.py`,
`anomaly_detection.py`.

Only two modules are currently consumed by `TradeAnalyst`:
- `analysis/alerts.py` (`AlertEngine`)
- `analysis/anomaly_detection.py`

The remaining modules are **not** dead: each is exercised by its own dedicated
test file and represents the intended analytical architecture. They were **not**
deleted — removing them would delete covered, intended functionality.
They are flagged as consolidation candidates: the agent imports shared helpers
from `ingestion.py` rather than these modules, so the modules form a
parallel/partially-redundant layer. Recommend a follow-up to route the analyst
through the canonical `analysis/*` modules (or deprecate them) — out of scope
for this behaviour-preserving pass.

---

## 3. Error handling & logging — IMPROVED

- `forex_agent/agent/analyst.py::_load_trades` swallowed all exceptions from
  JSONL/DB loads silently (`except Exception: self._trades = []`). Now logs the
  failure at `warning` level and records the source path.
- `forex_agent/data/ingestion.py::load_trades_from_database` used
  `except (sqlite3.Error, Exception)` (redundant — `Exception` subsumes
  `sqlite3.Error`) and was silent. Now logs and narrows to `sqlite3.Error`.
  `Path` re-import inside the function was redundant (module-level import).
- `ingestion.py::load_trades_from_jsonl` silently skipped malformed lines; now
  logs skip events at `debug` level (kept quiet to avoid noise on large files).
- Added `forex_agent/log.py`: centralised, structured logging setup
  (`get_logger`, `_configure_root`, `set_level`) — stderr handler, ISO
  timestamps, no secrets.

---

## 4. Configuration & hard-coded values — IMPROVED

- Strategy-health classification thresholds were hard-coded inside
  `analyst.py::_classify_health_status` (20-trade minimum, expectancy < -0.5R,
  profit factor < 0.7, drawdown > 15%, "normal variance" expectancy < 0.15R).
  Moved to `AgentConfig.health_thresholds` with identical defaults, so tuning
  no longer requires code edits. Wired through `load_config()` with the same
  `AGENT_*` env-var override pattern used for `alert_thresholds`.
- The main performance thresholds were already parameterised via
  `config.alert_thresholds`; verified no other hard-coded magic numbers affect
  behaviour.

---

## 5. Security gut-check — PASSED / HARDENED

- **No credentials in code.** `forex_agent/` contains no API keys, tokens,
  passwords, or `Authorization`/`Bearer` headers. The agent is a local analysis
  tool with no outbound API integration.
- **No secret logging.** `log.py` explicitly documents and enforces "no secrets
  are ever logged"; no code path prints environment values.
- **SQL is parameterised.** `forex_agent/storage/database.py` uses `?`
  placeholders for all inserts/selects (e.g. `save_trade_failure`,
  `save_alerts`, `save_report`, `save_metrics_history`). No string-concatenated
  SQL.
- **`.env` ignored.** `.gitignore` covers `.env` and `.env.local`.
- **Gap (HARDENED):** live/private trade data in `data/` (`fx_journal.jsonl`,
  `fx_forward_test.db`) and generated `reports/` were **not** gitignored.
  Added `data/`, `reports/`, `logs/` to `.gitignore` (with an exception for
  `*.example.*` template files). Note: the repo is not currently under Git; this
  prepares it for version control.
- Configuration is read from process environment (and `AGENT_*` vars) rather
  than a `.env` loader, so no secrets are encoded into the package.

---

## 6. Quantitative correctness preservation

All changes were behaviour-preserving. Verification:
- Full suite grew from **320 → 327 passing tests** (7 new regression tests).
- New tests lock in R-multiple batching (`compute_r_values`) and the new
  `health_thresholds` config surface.
- The health-score formula and status classification were **not** changed, only
  parameterised with identical default values.
- Live `analyze` output remains consistent with baseline ratios (win rate,
  expectancy, profit factor) — exact totals cannot be compared run-to-run
  because the journal/database are still growing.
