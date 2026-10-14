"""FastAPI service for traffic records and congestion prediction."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.config import get_settings
from src.data_storage import TrafficDataRepository
from src.feature_engineering import add_traffic_features
from src.model_artifacts import ModelIntegrityError
from src.predict import TrafficPredictionInput, TrafficPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Real-Time Traffic Congestion Prediction API",
    description="Traffic history and congestion prediction APIs backed by TomTom traffic data.",
    version="1.0.0",
)

_predictor: TrafficPredictor | None = None


class PredictionRequest(BaseModel):
    """Prediction request body."""

    location_name: str = Field(..., examples=["New Delhi"])
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    current_speed: float = Field(..., ge=0)
    free_flow_speed: float = Field(..., gt=0)
    confidence: float = Field(..., ge=0, le=1)
    travel_time: float = Field(..., ge=0)
    road_closure: bool = False
    timestamp: datetime | None = None


def repository() -> TrafficDataRepository:
    """Build the configured repository."""

    settings = get_settings()
    return TrafficDataRepository(settings.raw_data_path)


def get_predictor() -> TrafficPredictor:
    """Return the shared predictor, created on first use. A missing model is not cached."""

    global _predictor
    if _predictor is None:
        _predictor = TrafficPredictor(get_settings().model_path)
    return _predictor


def reset_predictor() -> None:
    """Drop the cached predictor so the next request builds a new one (used by tests)."""

    global _predictor
    _predictor = None


def _sorted_rows(since: datetime | None = None) -> pd.DataFrame:
    """Read raw rows in time order, optionally keeping only those at or after `since`."""

    rows = repository().read_raw()
    rows = rows.assign(timestamp=pd.to_datetime(rows["timestamp"], errors="coerce", utc=True))
    rows = rows.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable")
    if since is not None:
        cutoff = pd.Timestamp(since)
        cutoff = cutoff.tz_localize(UTC) if cutoff.tzinfo is None else cutoff.tz_convert(UTC)
        rows = rows[rows["timestamp"] >= cutoff]
    return rows


def _to_records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    """Coerce coordinates, drop rows without them and convert NaN to null for JSON."""

    dataframe = dataframe.assign(
        latitude=pd.to_numeric(dataframe["latitude"], errors="coerce"),
        longitude=pd.to_numeric(dataframe["longitude"], errors="coerce"),
    ).dropna(subset=["latitude", "longitude"])
    return dataframe.astype(object).where(dataframe.notna(), None).to_dict(orient="records")


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health."""

    return {"status": "ok", "timestamp": datetime.now(UTC).isoformat()}


@app.get("/traffic/latest")
def latest_traffic(limit: int = Query(20, ge=1, le=500)) -> list[dict[str, Any]]:
    """Return the newest observation for each location (at most `limit` locations)."""

    try:
        rows = _sorted_rows().groupby("location_name", sort=False).tail(1).tail(limit)
        if rows.empty:
            return []
        features = add_traffic_features(rows)
        return _to_records(features.sort_values("timestamp", ascending=False))
    except Exception as exc:
        logger.exception("Failed to read latest traffic")
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@app.get("/traffic/history")
def traffic_history(
    limit: int = Query(100, ge=1, le=5000),
    since: datetime | None = Query(None, description="Only rows at or after this time"),
) -> list[dict[str, Any]]:
    """Return the newest observations, newest first, optionally from `since` onwards."""

    try:
        rows = _sorted_rows(since).tail(limit)
        if rows.empty:
            return []
        features = add_traffic_features(rows)
        return _to_records(features.sort_values("timestamp", ascending=False))
    except Exception as exc:
        logger.exception("Failed to read traffic history")
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    """Predict congestion level for supplied traffic values."""

    try:
        payload = TrafficPredictionInput(
            **request.model_dump(exclude={"timestamp"}),
            timestamp=request.timestamp.isoformat() if request.timestamp else None,
        )
        return get_predictor().predict(payload)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503, detail="Model is not available. Train the model first."
        ) from exc
    except ValueError as exc:
        # Invalid timestamp or missing/zero speeds: the message names the field.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelIntegrityError as exc:
        # Checksum or metadata mismatch is a server-side fault: log it, never echo it.
        logger.error("Model integrity check failed: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal server error") from exc
