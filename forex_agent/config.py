from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    database_url: str = "sqlite:///./data/fx_forward_test.db"
    jsonl_path: str = "data/fx_journal.jsonl"
    alert_thresholds: dict[str, float] = field(default_factory=lambda: {
        "max_drawdown_pct": 0.10,
        "min_win_rate": 0.35,
        "min_expectancy_r": -0.3,
        "max_consecutive_losses": 5,
        "max_slippage_pips": 3.0,
        "max_spread_pips": 5.0,
        "min_profit_factor": 0.8,
        "max_anomaly_z_score": 2.5,
    })
    health_thresholds: dict[str, float] = field(default_factory=lambda: {
        "critical_min_sample": 20.0,
        "critical_expectancy": -0.5,
        "critical_profit_factor": 0.7,
        "critical_max_drawdown_pct": 0.15,
        "normal_variance_expectancy": 0.15,
    })
    rolling_window_size: int = 30
    min_sample_size: int = 20
    confidence_level: float = 0.95
    research_memory_path: str = "data/research_memory.jsonl"
    # LLM settings (OpenAI primary, Ollama fallback, template if neither)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2000

    def __post_init__(self) -> None:
        if self.rolling_window_size < 1:
            raise ValueError("rolling_window_size must be >= 1")
        if self.min_sample_size < 1:
            raise ValueError("min_sample_size must be >= 1")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0 and 1")


_HEALTH_THRESHOLD_KEYS = [
    "critical_min_sample",
    "critical_expectancy",
    "critical_profit_factor",
    "critical_max_drawdown_pct",
    "normal_variance_expectancy",
]


def load_config() -> AgentConfig:
    thresholds: dict[str, float] = {}
    threshold_keys = [
        "max_drawdown_pct",
        "min_win_rate",
        "min_expectancy_r",
        "max_consecutive_losses",
        "max_slippage_pips",
        "max_spread_pips",
        "min_profit_factor",
        "max_anomaly_z_score",
    ]
    for key in threshold_keys:
        env_key = f"AGENT_{key.upper()}"
        raw = os.getenv(env_key)
        if raw is not None and raw.strip() != "":
            thresholds[key] = float(raw)

    health: dict[str, float] = {}
    for key in _HEALTH_THRESHOLD_KEYS:
        env_key = f"AGENT_{key.upper()}"
        raw = os.getenv(env_key)
        if raw is not None and raw.strip() != "":
            health[key] = float(raw)

    defaults = AgentConfig()
    merged = {**defaults.alert_thresholds, **thresholds}
    merged_health = {**defaults.health_thresholds, **health}

    return AgentConfig(
        database_url=os.getenv("AGENT_DATABASE_URL", defaults.database_url),
        jsonl_path=os.getenv("AGENT_JSONL_PATH", defaults.jsonl_path),
        alert_thresholds=merged,
        health_thresholds=merged_health,
        rolling_window_size=int(os.getenv("AGENT_ROLLING_WINDOW_SIZE", str(defaults.rolling_window_size))),
        min_sample_size=int(os.getenv("AGENT_MIN_SAMPLE_SIZE", str(defaults.min_sample_size))),
        confidence_level=float(os.getenv("AGENT_CONFIDENCE_LEVEL", str(defaults.confidence_level))),
        research_memory_path=os.getenv("AGENT_RESEARCH_MEMORY_PATH", defaults.research_memory_path),
        openai_api_key=os.getenv("OPENAI_API_KEY", defaults.openai_api_key),
        openai_model=os.getenv("OPENAI_MODEL", defaults.openai_model),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", defaults.ollama_base_url),
        ollama_model=os.getenv("OLLAMA_MODEL", defaults.ollama_model),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", str(defaults.llm_temperature))),
        llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", str(defaults.llm_max_tokens))),
    )
