"""Feature engineering for traffic congestion modeling."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LOCAL_TIMEZONE = "Asia/Kolkata"
LOW_MIN_SPEED_RATIO = 0.75
MEDIUM_MIN_SPEED_RATIO = 0.45
MAX_SPEED_RATIO = 1.5

# A row needs these to be labelled or used as model input; otherwise it is dropped.
REQUIRED_FIELDS = ["current_speed", "free_flow_speed", "confidence", "latitude", "longitude"]

DERIVED_DTYPES = {
    "speed_ratio": float,
    "hour": int,
    "day_of_week": int,
    "month": int,
    "is_weekend": int,
    "peak_hour": int,
    "congestion_level": object,
}


def classify_congestion(speed_ratio: float) -> str:
    """Convert speed ratio into Low, Medium, or High congestion."""

    if pd.isna(speed_ratio):
        raise ValueError("speed_ratio is missing; drop the row before labelling")
    if speed_ratio >= LOW_MIN_SPEED_RATIO:
        return "Low"
    if speed_ratio >= MEDIUM_MIN_SPEED_RATIO:
        return "Medium"
    return "High"


def add_traffic_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Generate model-ready traffic features from raw observations."""

    if dataframe.empty:
        derived = [column for column in DERIVED_DTYPES if column not in dataframe.columns]
        return dataframe.reindex(columns=[*dataframe.columns, *derived]).astype(DERIVED_DTYPES)

    df = dataframe.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    invalid_timestamps = int(df["timestamp"].isna().sum())
    if invalid_timestamps:
        logger.warning("Dropped %s rows with invalid timestamps", invalid_timestamps)
    df = df.dropna(subset=["timestamp"])

    for column in ["current_speed", "free_flow_speed", "confidence", "travel_time"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # A zero or missing free-flow speed leaves the ratio undefined (NaN), never 0.
    free_flow = df["free_flow_speed"].where(df["free_flow_speed"] > 0)
    df["speed_ratio"] = (df["current_speed"] / free_flow).clip(lower=0.0, upper=MAX_SPEED_RATIO)

    incomplete = df[REQUIRED_FIELDS].isna().any(axis=1) | df["speed_ratio"].isna()
    if incomplete.any():
        logger.warning("Dropped %s rows with missing speed or fields", int(incomplete.sum()))
    df = df[~incomplete].copy()

    local_time = df["timestamp"].dt.tz_convert(LOCAL_TIMEZONE)
    df["hour"] = local_time.dt.hour
    df["day_of_week"] = local_time.dt.dayofweek
    df["month"] = local_time.dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["peak_hour"] = (df["hour"].between(7, 9) | df["hour"].between(17, 19)).astype(int)

    # Same thresholds as classify_congestion, applied to whole columns.
    df["congestion_level"] = np.select(
        [df["speed_ratio"] >= LOW_MIN_SPEED_RATIO, df["speed_ratio"] >= MEDIUM_MIN_SPEED_RATIO],
        ["Low", "Medium"],
        default="High",
    )
    return df


def ensure_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Featurise raw rows once; frames that already carry the label are returned as-is."""

    if "congestion_level" in dataframe.columns:
        return dataframe.copy()
    return add_traffic_features(dataframe)
