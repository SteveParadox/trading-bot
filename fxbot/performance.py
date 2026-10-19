"""Bounded, process-local operation timings; no market data or trading state.

The authenticated monitoring endpoint exposes counters and a rolling latency
window. Labels are fixed by code, never by symbols/candidate IDs/user input.
"""
from collections import deque
from functools import wraps
from threading import Lock
from time import perf_counter, process_time


class OperationTimings:
    def __init__(self, capacity: int = 256, max_operations: int = 64):
        self.capacity = capacity
        self.max_operations = max_operations
        self._lock = Lock()
        self._operations = {}

    def observe(self, operation: str, milliseconds: float, failed: bool = False):
        with self._lock:
            if operation not in self._operations:
                if len(self._operations) >= self.max_operations:
                    return
                self._operations[operation] = {"count": 0, "failures": 0, "total_ms": 0.0,
                                               "samples": deque(maxlen=self.capacity)}
            metric = self._operations[operation]
            metric["count"] += 1
            metric["failures"] += int(failed)
            metric["total_ms"] += milliseconds
            metric["samples"].append(milliseconds)

    def snapshot(self):
        with self._lock:
            result = {}
            for name, metric in self._operations.items():
                samples = sorted(metric["samples"])
                result[name] = {key: metric[key] for key in ("count", "failures", "total_ms")}
                result[name].update({"window_size": len(samples), "last_ms": metric["samples"][-1],
                                     "p50_ms": samples[len(samples) // 2],
                                     "p95_ms": samples[min(len(samples) - 1, int(len(samples) * .95))]})
        return result


# One shared metrics registry lets API and worker threads report the same
# process. This contains only bounded numeric aggregates, never cached feeds.
operation_timings = OperationTimings()


def measured(operation):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            started = perf_counter()
            failed = False
            try:
                return function(*args, **kwargs)
            except BaseException:
                failed = True
                raise
            finally:
                operation_timings.observe(operation, (perf_counter() - started) * 1000, failed)
        return wrapped
    return decorate


def performance_snapshot():
    # CPU time is cumulative for the process, not a fabricated utilization %.
    # ru_maxrss is a high-water mark, not current resident memory.
    peak_rss_bytes = None
    try:
        import resource
        import sys
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss_bytes = peak if sys.platform == "darwin" else peak * 1024
    except ImportError:  # Windows MT5: resource is unavailable.
        pass
    return {"operations": operation_timings.snapshot(), "process_cpu_seconds": process_time(),
            "peak_rss_bytes": peak_rss_bytes, "sample_capacity": operation_timings.capacity}
