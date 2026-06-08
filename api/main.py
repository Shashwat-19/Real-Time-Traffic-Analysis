"""FastAPI service for traffic records and congestion prediction."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.config import get_settings
from src.data_storage import TrafficDataRepository
from src.feature_engineering import add_traffic_features
from src.predict import TrafficPredictionInput, TrafficPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Real-Time Traffic Congestion Prediction API",
    description="Traffic history and congestion prediction APIs backed by TomTom traffic data.",
    version="1.0.0",
)


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
    timestamp: str | None = None


def repository() -> TrafficDataRepository:
    """Build the configured repository."""

    settings = get_settings()
    return TrafficDataRepository(settings.raw_data_path, settings.processed_data_path)


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health."""

    return {"status": "ok", "timestamp": datetime.now(UTC).isoformat()}


@app.get("/traffic/latest")
def latest_traffic(limit: int = Query(20, ge=1, le=500)) -> list[dict[str, Any]]:
    """Return the latest traffic observations."""

    dataframe = add_traffic_features(repository().read_raw())
    if dataframe.empty:
        return []
    return dataframe.sort_values("timestamp", ascending=False).head(limit).to_dict(orient="records")


@app.get("/traffic/history")
def traffic_history(limit: int = Query(100, ge=1, le=5000)) -> list[dict[str, Any]]:
    """Return historical traffic observations."""

    dataframe = add_traffic_features(repository().read_raw())
    if dataframe.empty:
        return []
    return dataframe.sort_values("timestamp", ascending=False).head(limit).to_dict(orient="records")


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    """Predict congestion level for supplied traffic values."""

    settings = get_settings()
    try:
        predictor = TrafficPredictor(settings.model_path)
        payload = TrafficPredictionInput(**request.model_dump())
        return predictor.predict(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail="Prediction failed") from exc
