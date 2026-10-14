"""Tests for feature lists, leakage guard and the shared chronological split."""

import pandas as pd
import pytest

from src.feature_engineering import add_traffic_features
from src.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    LABEL_SOURCE_COLUMNS,
    NUMERIC_FEATURES,
    split_chronologically,
)


def test_numeric_and_categorical_features_cover_feature_columns() -> None:
    assert NUMERIC_FEATURES + CATEGORICAL_FEATURES == FEATURE_COLUMNS


def test_no_model_feature_is_a_label_source_column() -> None:
    leaked = set(FEATURE_COLUMNS) & LABEL_SOURCE_COLUMNS

    assert leaked == set()


def _featured(count: int) -> pd.DataFrame:
    rows = [
        {
            "timestamp": pd.Timestamp("2026-06-01T00:00:00Z") + pd.Timedelta(minutes=10 * i),
            "location_name": "Delhi",
            "latitude": 28.61,
            "longitude": 77.20,
            "current_speed": 10.0 + (i % 50),
            "free_flow_speed": 50.0,
            "confidence": 0.9,
            "travel_time": 100.0,
            "road_closure": False,
        }
        for i in range(count)
    ]
    return add_traffic_features(pd.DataFrame(rows))


def test_split_uses_most_recent_block_as_test_even_when_input_is_unsorted() -> None:
    featured = _featured(50).sample(frac=1.0, random_state=0)

    x_train, x_test, y_train, y_test = split_chronologically(featured, test_size=0.2)

    assert len(x_test) == 10
    assert len(x_train) == 40
    assert list(x_train.columns) == FEATURE_COLUMNS
    # Every training row is earlier than every test row.
    train_times = featured.loc[x_train.index, "timestamp"]
    test_times = featured.loc[x_test.index, "timestamp"]
    assert train_times.max() < test_times.min()
    assert list(y_test.index) == list(x_test.index)
    assert set(y_train.index) == set(x_train.index)


def test_split_needs_at_least_two_rows() -> None:
    with pytest.raises(ValueError, match="At least 2"):
        split_chronologically(_featured(1))
