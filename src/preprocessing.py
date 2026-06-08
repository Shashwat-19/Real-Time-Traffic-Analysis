"""Reusable preprocessing utilities for model training and prediction."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_COLUMNS = [
    "latitude",
    "longitude",
    "current_speed",
    "free_flow_speed",
    "confidence",
    "travel_time",
    "road_closure",
    "speed_ratio",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "peak_hour",
    "location_name",
]

NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "current_speed",
    "free_flow_speed",
    "confidence",
    "travel_time",
    "speed_ratio",
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
