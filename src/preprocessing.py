"""Reusable preprocessing utilities for model training and prediction."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# The label is derived from current_speed / free_flow_speed, so these must never be inputs.
LABEL_SOURCE_COLUMNS = frozenset({"current_speed", "free_flow_speed", "speed_ratio", "travel_time"})

FEATURE_COLUMNS = [
    "latitude",
    "longitude",
    "confidence",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "peak_hour",
    "road_closure",
    "location_name",
]

NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "confidence",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "peak_hour",
]

CATEGORICAL_FEATURES = ["road_closure", "location_name"]


def build_preprocessor() -> ColumnTransformer:
    """Build a preprocessing transformer for numeric and categorical fields."""

    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def split_chronologically(
    featured: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split featured rows by time: earlier rows train, the most recent block tests.

    Shared by training and evaluation so both use the same holdout.
    """

    if len(featured) < 2:
        raise ValueError("At least 2 records are needed for a train/test split")
    ordered = featured.sort_values("timestamp", kind="stable")
    test_rows = max(1, round(len(ordered) * test_size))
    train, test = ordered.iloc[:-test_rows], ordered.iloc[-test_rows:]
    return (
        train[FEATURE_COLUMNS],
        test[FEATURE_COLUMNS],
        train["congestion_level"],
        test["congestion_level"],
    )
