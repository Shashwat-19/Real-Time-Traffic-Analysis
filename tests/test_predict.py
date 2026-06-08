"""Tests for the prediction pipeline."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.feature_engineering import add_traffic_features
from src.predict import TrafficPredictionInput, TrafficPredictor
from src.preprocessing import FEATURE_COLUMNS, build_preprocessor


def test_predictor_returns_congestion_class(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        [
            {
                "timestamp": "2026-06-08T08:00:00+00:00",
                "location_name": "Delhi",
                "latitude": 28.61,
                "longitude": 77.20,
                "current_speed": 20,
                "free_flow_speed": 50,
                "confidence": 0.9,
                "travel_time": 180,
                "road_closure": False,
            },
            {
                "timestamp": "2026-06-08T11:00:00+00:00",
                "location_name": "Delhi",
                "latitude": 28.61,
                "longitude": 77.20,
                "current_speed": 48,
                "free_flow_speed": 50,
                "confidence": 0.9,
                "travel_time": 80,
                "road_closure": False,
            },
        ]
    )
    featured = add_traffic_features(raw)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]
    )
    pipeline.fit(featured[FEATURE_COLUMNS], featured["congestion_level"])
    model_path = tmp_path / "traffic_model.pkl"
    joblib.dump(pipeline, model_path)

    result = TrafficPredictor(model_path).predict(
        TrafficPredictionInput(
            location_name="Delhi",
            latitude=28.61,
            longitude=77.20,
            current_speed=18,
            free_flow_speed=50,
            confidence=0.9,
            travel_time=190,
            road_closure=False,
        )
    )

    assert result["prediction"] in {"Low", "Medium", "High"}
