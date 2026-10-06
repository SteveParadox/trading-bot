"""Shared deterministic tabular coercion for fitting and artifact inference."""
import numpy as np
import pandas as pd
from fxbot.training_dataset import FEATURE_COLUMNS

CATEGORICAL_FEATURES = [
    "symbol", "direction", "strategy_signal", "session", "account_currency",
    "news_risk", "upcoming_news_currency", "upcoming_news_impact",
    "recent_news_currency", "news_freshness_state",
]


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"dataset is missing model feature columns: {missing}")
    result = frame.loc[:, FEATURE_COLUMNS].copy()
    for column in result:
        if column in CATEGORICAL_FEATURES:
            result[column] = result[column].map(lambda value: str(value) if pd.notna(value) else np.nan).astype(object)
        else:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
    return result
