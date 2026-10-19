"""Reproducible synthetic engineering benchmark; never a trading backtest.

Run this same script against both source trees using --repo. Workloads,
iterations and warmup are fixed; traced allocations are measured separately
because tracemalloc distorts latency. All databases are temporary.
"""
from __future__ import annotations
import argparse
import subprocess
import hashlib
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time
import tracemalloc


def measure(operation, iterations):
    for _ in range(3):
        operation()
    samples = []
    cpu = time.process_time()
    for _ in range(iterations):
        start = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - start) * 1000)
    cpu = time.process_time() - cpu
    tracemalloc.start()
    for _ in range(3):
        operation()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ordered = sorted(samples)
    return {"iterations": iterations, "median_ms": statistics.median(samples),
            "p95_ms": ordered[min(len(ordered)-1, int(len(ordered)*.95))],
            "cpu_seconds": cpu, "peak_traced_bytes_3_calls": peak}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=60)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    sys.path.insert(0, str(args.repo.resolve()))
    import numpy as np
    import pandas as pd
    from fxbot.strategy import prepare_indicators, evaluate_signal_frame
    from fxbot.config import StrategySettings
    from fxbot.instruments import FxInstrument
    from fxbot.journal import StructuredJournal
    from fxbot.database import SignalJournalRow

    def frame(freq, last, step):
        close = 1.08 + np.arange(180) * step
        return pd.DataFrame({"open": close-step*.5, "high": close+step*2, "low": close-step*2,
                             "close": close, "volume": np.linspace(100,150,180)},
                            index=pd.date_range(end=last, periods=180, freq=freq))
    entry = frame("15min", "2026-01-08T11:45Z", .00008)
    htf = frame("4h", "2026-01-08T08:00Z", .0007)
    structure = frame("1h", "2026-01-08T11:00Z", .0003)
    now = datetime(2026,1,8,12,0,tzinfo=timezone.utc)
    settings = StrategySettings(entry_timeframe="15m", htf_timeframe="4h", structure_timeframe="1h")
    source_files = sorted((args.repo / "fxbot").rglob("*.py")) + [args.repo / "indicators.py", args.repo / "config.py"]
    source_hash = hashlib.sha256(b"".join(path.read_bytes() for path in source_files)).hexdigest()
    revision = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()
    report = {"git_revision": revision, "source_sha256": source_hash, "kind": "synthetic_engineering_benchmark_not_trading_results", "python": platform.python_version(),
              "platform": platform.platform(), "pandas": pd.__version__, "numpy": np.__version__, "workloads": {}}
    report["workloads"]["indicators_180_bars"] = measure(lambda: prepare_indicators(entry), args.iterations)
    report["workloads"]["three_frame_raw_decision"] = measure(lambda: evaluate_signal_frame(entry, htf,
        instrument=FxInstrument("EUR_USD"), settings=settings, timestamp=now, structure_frame=structure), args.iterations)
    with tempfile.TemporaryDirectory() as temporary:
        journal = StructuredJournal(f"sqlite:///{temporary}/benchmark.db")
        try:
            with journal.sessions.begin() as session:
                session.add_all([SignalJournalRow(timestamp=now, instrument="EUR_USD", status="skipped", reason="benchmark") for _ in range(10_000)])
            report["workloads"]["journal_read_100_of_10000"] = measure(lambda: journal.recent_signals(100), args.iterations)
            report["workloads"]["journal_signal_write"] = measure(lambda: journal.record_signal(timestamp=now,
                instrument="EUR_USD", status="skipped", reason="benchmark"), args.iterations)
        finally:
            journal.close()
        # API requests use real middleware and serialization, no broker.
        import os
        os.environ["FX_DATABASE_URL"] = f"sqlite:///{temporary}/default.db"
        from fxbot.api import create_app
        from fxbot.config import FxBotSettings, RuntimeSettings
        from fastapi.testclient import TestClient
        api_settings = FxBotSettings(runtime=RuntimeSettings(database_url=f"sqlite:///{temporary}/api.db",
            log_jsonl_path=None, start_worker_with_api=False, api_key="benchmark-local-key", api_rate_limit_per_minute=100000))
        with TestClient(create_app(api_settings)) as client:
            def request():
                response = client.get("/api/config", headers={"X-API-Key": "benchmark-local-key"})
                response.raise_for_status()
            report["workloads"]["authenticated_config_api"] = measure(request, args.iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
