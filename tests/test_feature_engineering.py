"""Tests for traffic feature engineering."""

import logging

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import add_traffic_features, classify_congestion


def _row(timestamp: str, **overrides: object) -> dict:
    row = {
        "timestamp": timestamp,
        "location_name": "Delhi",
        "latitude": 28.61,
        "longitude": 77.20,
        "current_speed": 25,
        "free_flow_speed": 50,
        "confidence": 0.9,
        "travel_time": 150,
        "road_closure": False,
    }
    row.update(overrides)
    return row


def test_classify_congestion_thresholds() -> None:
    assert classify_congestion(0.9) == "Low"
    assert classify_congestion(0.5) == "Medium"
    assert classify_congestion(0.2) == "High"


@pytest.mark.parametrize(
    ("speed_ratio", "expected"),
    [
        (0.75, "Low"),
        (0.7499, "Medium"),
        (0.45, "Medium"),
        (0.4499, "High"),
        (0.0, "High"),
    ],
)
def test_classify_congestion_exact_boundaries(speed_ratio: float, expected: str) -> None:
    assert classify_congestion(speed_ratio) == expected


def test_classify_congestion_rejects_missing_ratio() -> None:
    with pytest.raises(ValueError, match="missing"):
        classify_congestion(float("nan"))


def test_vectorised_labels_match_classify_congestion() -> None:
    ratios = [0.0, 0.3, 0.4499, 0.45, 0.6, 0.7499, 0.75, 0.9, 1.0]
    dataframe = pd.DataFrame(
        [_row("2026-06-08T08:00:00+00:00", current_speed=r * 50.0) for r in ratios]
    )
    result = add_traffic_features(dataframe)

    expected = [classify_congestion(r) for r in result["speed_ratio"]]
    assert list(result["congestion_level"]) == expected


def test_add_traffic_features_generates_expected_columns() -> None:
    result = add_traffic_features(pd.DataFrame([_row("2026-06-08T08:00:00+00:00")]))

    assert result.loc[0, "speed_ratio"] == 0.5
    # 08:00 UTC is 13:30 IST, which is outside the peak windows.
    assert result.loc[0, "hour"] == 13
    assert result.loc[0, "peak_hour"] == 0
    assert result.loc[0, "congestion_level"] == "Medium"


@pytest.mark.parametrize(
    ("local_time", "expected_peak"),
    [
        ("06:00", 0),
        ("07:00", 1),
        ("09:59", 1),
        ("10:00", 0),
        ("16:00", 0),
        ("17:00", 1),
        ("19:59", 1),
        ("20:00", 0),
    ],
)
def test_peak_hour_edges_in_ist(local_time: str, expected_peak: int) -> None:
    result = add_traffic_features(pd.DataFrame([_row(f"2026-06-08T{local_time}:00+05:30")]))

    assert result.loc[0, "peak_hour"] == expected_peak


def test_timestamps_are_converted_to_ist_before_extracting_hour() -> None:
    # 02:00 UTC is 07:30 IST, so it is a peak hour even though the UTC hour is 2.
    result = add_traffic_features(pd.DataFrame([_row("2026-06-08T02:00:00+00:00")]))

    assert result.loc[0, "hour"] == 7
    assert result.loc[0, "peak_hour"] == 1


def test_weekend_flag_uses_ist_day() -> None:
    saturday = add_traffic_features(pd.DataFrame([_row("2026-06-06T10:00:00+05:30")]))
    # Sunday 20:00 UTC is already Monday 01:30 IST, so it is a weekday locally.
    sunday_utc_monday_ist = add_traffic_features(
        pd.DataFrame([_row("2026-06-07T20:00:00+00:00")])
    )

    assert saturday.loc[0, "is_weekend"] == 1
    assert sunday_utc_monday_ist.loc[0, "is_weekend"] == 0


def test_missing_speed_rows_are_dropped_before_labelling(caplog: pytest.LogCaptureFixture) -> None:
    dataframe = pd.DataFrame(
        [
            _row("2026-06-08T08:00:00+00:00"),
            _row("2026-06-08T08:10:00+00:00", current_speed=None),
            _row("2026-06-08T08:20:00+00:00", free_flow_speed=None),
        ]
    )
    with caplog.at_level(logging.WARNING, logger="src.feature_engineering"):
        result = add_traffic_features(dataframe)

    assert len(result) == 1
    assert result["congestion_level"].iloc[0] == "Medium"
    assert "Dropped 2 rows" in caplog.text


def test_zero_free_flow_speed_is_dropped_not_labelled_high() -> None:
    dataframe = pd.DataFrame(
        [_row("2026-06-08T08:00:00+00:00", free_flow_speed=0), _row("2026-06-08T08:10:00+00:00")]
    )
    result = add_traffic_features(dataframe)

    assert len(result) == 1
    assert result["speed_ratio"].notna().all()


def test_speed_ratio_is_clipped_at_1_5() -> None:
    result = add_traffic_features(
        pd.DataFrame([_row("2026-06-08T08:00:00+00:00", current_speed=200, free_flow_speed=50)])
    )

    assert result.loc[0, "speed_ratio"] == 1.5
    assert result.loc[0, "congestion_level"] == "Low"


def test_missing_road_closure_is_not_treated_as_closed() -> None:
    dataframe = pd.DataFrame(
        [
            _row("2026-06-08T08:00:00+00:00", road_closure=None),
            _row("2026-06-08T08:10:00+00:00", road_closure=True),
            _row("2026-06-08T08:20:00+00:00", road_closure=False),
        ]
    )
    result = add_traffic_features(dataframe)

    assert pd.isna(result.loc[0, "road_closure"])
    assert result.loc[1, "road_closure"] is True
    assert result.loc[2, "road_closure"] is False


def test_invalid_timestamp_rows_are_dropped_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    dataframe = pd.DataFrame(
        [_row("2026-06-08T08:00:00+00:00"), _row("not-a-time")]
    )
    with caplog.at_level(logging.WARNING, logger="src.feature_engineering"):
        result = add_traffic_features(dataframe)

    assert len(result) == 1
    assert "invalid timestamps" in caplog.text


def test_empty_frame_keeps_derived_columns_with_dtypes() -> None:
    result = add_traffic_features(pd.DataFrame(columns=["timestamp", "current_speed"]))

    assert result.empty
    for column in ["speed_ratio", "hour", "day_of_week", "month", "is_weekend", "peak_hour"]:
        assert column in result.columns
    assert result["speed_ratio"].dtype == np.float64
    assert result["hour"].dtype == np.int64
    assert "congestion_level" in result.columns
