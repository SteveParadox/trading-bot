"""Generate deterministic sample OHLCV files for smoke-testing the backtester.

These files are not research data. Use real Bybit candles for actual strategy
evaluation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_symbol(symbol: str, start_price: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01", periods=3500, freq="5min", tz="UTC")
    phase = np.linspace(0, 14 * np.pi, len(index))
    regime = np.sign(np.sin(phase))
    regime[regime == 0] = 1
    trend = regime * 0.0018
    pullback = np.sin(np.linspace(0, 80 * np.pi, len(index))) * 0.0007
    shocks = rng.normal(0, 0.0012, len(index))
    returns = trend + pullback + shocks
    close = start_price * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.maximum(close * rng.uniform(0.0005, 0.004, len(index)), 0.0001)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    activity = 1.0 + (np.sin(phase + 0.4) > 0.25) * 0.8
    impulse = 1.0 + (np.sin(np.linspace(0, 220 * np.pi, len(index))) > 0.85) * 1.2
    volume = rng.lognormal(mean=7.0, sigma=0.25, size=len(index)) * activity * impulse
    return pd.DataFrame(
        {
            "timestamp": index,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def main() -> None:
    output = Path("data")
    output.mkdir(exist_ok=True)
    make_symbol("SAGAUSDT", 1.10, 1).to_csv(output / "SAGAUSDT_5m.csv", index=False)
    make_symbol("BUSDT", 0.75, 2).to_csv(output / "BUSDT_5m.csv", index=False)
    make_symbol("NEARUSDT", 1.10, 3).to_csv(output / "NEARUSDT_5m.csv", index=False)
    print("Wrote sample data to data/*.csv")


if __name__ == "__main__":
    main()
