"""Feature engineering for traffic congestion modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd


def classify_congestion(speed_ratio: float) -> str:
    """Convert speed ratio into Low, Medium, or High congestion."""

    if speed_ratio >= 0.75:
        return "Low"
    if speed_ratio >= 0.45:
        return "Medium"
    return "High"


def add_traffic_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Generate model-ready traffic features from raw observations."""

    if dataframe.empty:
        return dataframe.copy()

    df = dataframe.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])

    df["current_speed"] = pd.to_numeric(df["current_speed"], errors="coerce").fillna(0.0)
    df["free_flow_speed"] = pd.to_numeric(df["free_flow_speed"], errors="coerce").fillna(0.0)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    df["travel_time"] = pd.to_numeric(df["travel_time"], errors="coerce").fillna(0.0)
    df["road_closure"] = df["road_closure"].astype(bool)

    df["speed_ratio"] = np.where(
        df["free_flow_speed"] > 0,
        df["current_speed"] / df["free_flow_speed"],
        0.0,
    )
    df["speed_ratio"] = df["speed_ratio"].clip(lower=0.0, upper=1.5)
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["peak_hour"] = df["hour"].between(7, 10, inclusive="left") | df["hour"].between(
        17, 20, inclusive="left"
    )
    df["peak_hour"] = df["peak_hour"].astype(int)
    df["congestion_level"] = df["speed_ratio"].apply(classify_congestion)
    return df
