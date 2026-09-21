from __future__ import annotations

import pytest

from forex_agent.config import AgentConfig, load_config


def test_agent_config_default_health_thresholds_present() -> None:
    config = AgentConfig()
    ht = config.health_thresholds
    assert ht["critical_min_sample"] == 100.0
    assert ht["critical_expectancy"] == -0.5
    assert ht["critical_profit_factor"] == 0.7
    assert ht["critical_max_drawdown_pct"] == 0.15
    assert ht["normal_variance_expectancy"] == 0.15
    assert ht["minimum_expectancy_r"] == 0.10
    assert config.min_sample_size == 100


def test_load_config_accepts_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_DATABASE_URL", "sqlite:///./data/custom.db")
    monkeypatch.setenv("AGENT_MIN_WIN_RATE", "0.40")
    monkeypatch.setenv("AGENT_CRITICAL_MAX_DRAWDOWN_PCT", "0.25")
    config = load_config()
    assert config.database_url == "sqlite:///./data/custom.db"
    assert config.alert_thresholds["min_win_rate"] == 0.40
    assert config.health_thresholds["critical_max_drawdown_pct"] == 0.25


def test_load_config_defaults_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_DATABASE_URL", raising=False)
    monkeypatch.delenv("AGENT_MIN_WIN_RATE", raising=False)
    monkeypatch.delenv("AGENT_CRITICAL_MAX_DRAWDOWN_PCT", raising=False)
    config = load_config()
    assert config.database_url == AgentConfig().database_url
    assert config.health_thresholds["critical_expectancy"] == -0.5
    assert config.alert_thresholds["min_profit_factor"] == 0.8
