from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(name: str, path: Path):
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_root_config_strips_and_normalizes_env_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STOP_MODE", " ATR ")
    monkeypatch.setenv("TP_MODE", " fixed ")
    monkeypatch.setenv("ORDER_LINK_PREFIX", "  custom-prefix  ")
    module = _load_module("config", ROOT / "config.py")

    assert module.STOP_MODE == "atr"
    assert module.TP_MODE == "fixed"
    assert module.ORDER_LINK_PREFIX == "custom-prefix"


def test_root_config_rejects_invalid_float_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RISK_PER_TRADE_PCT", "not-a-number")

    with pytest.raises(ValueError, match="RISK_PER_TRADE_PCT"):
        _load_module("config", ROOT / "config.py")


def test_fxbot_settings_strip_and_validate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FX_API_KEY", "  demo-key  ")
    monkeypatch.setenv("MT5_ORDER_FILLING", " return ")
    monkeypatch.setenv("FX_ENTRY_TIMEFRAME", " 30m ")
    monkeypatch.setenv("FX_STOP_MODE", " ATR ")
    monkeypatch.setenv("FX_ACCOUNT_CURRENCY", " usd ")

    module = _load_module("fxbot.config", ROOT / "fxbot" / "config.py")

    settings = module.settings_from_env()

    assert settings.runtime.api_key == "demo-key"
    assert settings.broker.order_filling == "RETURN"
    assert settings.strategy.entry_timeframe == "30m"
    assert settings.strategy.stop_mode == "atr"
    assert settings.risk.account_currency == "USD"


def test_fxbot_settings_rejects_invalid_float_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FX_MIN_RISK_REWARD", "oops")

    module = _load_module("fxbot.config", ROOT / "fxbot" / "config.py")

    with pytest.raises(ValueError, match="FX_MIN_RISK_REWARD"):
        module.settings_from_env()
