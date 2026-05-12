"""Professional Bybit USDT perpetual futures backtesting framework.

The package is intentionally additive to the live bot.  It reuses the existing
indicator module while keeping exchange simulation, risk, analytics, and
optimization isolated from live-trading code.
"""

from backtester.analytics import PerformanceAnalyzer
from backtester.config import BacktestConfig
from backtester.data import DataPortal
from backtester.engine import BacktestEngine, BacktestResult
from backtester.execution import SimulatedExchange
from backtester.risk import RiskManager
from backtester.strategy import IndicatorSignalStrategy

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "DataPortal",
    "IndicatorSignalStrategy",
    "PerformanceAnalyzer",
    "RiskManager",
    "SimulatedExchange",
]
