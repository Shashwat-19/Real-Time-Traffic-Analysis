"""Prediction pipeline for congestion inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.feature_engineering import add_traffic_features
from src.preprocessing import FEATURE_COLUMNS


@dataclass
class TrafficPredictionInput:
    """Input data required for congestion prediction."""

    location_name: str
    latitude: float
    longitude: float
    current_speed: float
    free_flow_speed: float
    confidence: float
    travel_time: float
    road_closure: bool = False
    timestamp: str | None = None


class TrafficPredictor:
    """Load a persisted model and generate congestion predictions."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        """Lazily load the model from disk."""

        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            self._model = joblib.load(self.model_path)
        return self._model

    def predict(self, payload: TrafficPredictionInput) -> dict[str, Any]:
        """Predict the congestion class and probability scores."""

        record = asdict(payload)
        record["timestamp"] = record["timestamp"] or datetime.now(UTC).isoformat()
        features = add_traffic_features(pd.DataFrame([record]))[FEATURE_COLUMNS]
        prediction = str(self.model.predict(features)[0])

        probabilities: dict[str, float] = {}
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features)[0]
            probabilities = {
                str(label): float(probability)
                for label, probability in zip(self.model.classes_, proba, strict=False)
            }

        return {
            "prediction": prediction,
            "probabilities": probabilities,
            "features": features.iloc[0].to_dict(),
        }
