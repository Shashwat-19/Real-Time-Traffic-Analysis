"""Tests for traffic feature engineering."""

import pandas as pd

from src.feature_engineering import add_traffic_features, classify_congestion


def test_classify_congestion_thresholds() -> None:
    assert classify_congestion(0.9) == "Low"
    assert classify_congestion(0.5) == "Medium"
    assert classify_congestion(0.2) == "High"


def test_add_traffic_features_generates_expected_columns() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "timestamp": "2026-06-08T08:00:00+00:00",
                "location_name": "Delhi",
                "latitude": 28.61,
                "longitude": 77.20,
                "current_speed": 25,
                "free_flow_speed": 50,
                "confidence": 0.9,
                "travel_time": 150,
                "road_closure": False,
            }
        ]
    )
    result = add_traffic_features(dataframe)

    assert result.loc[0, "speed_ratio"] == 0.5
    assert result.loc[0, "hour"] == 8
    assert result.loc[0, "peak_hour"] == 1
    assert result.loc[0, "congestion_level"] == "Medium"
