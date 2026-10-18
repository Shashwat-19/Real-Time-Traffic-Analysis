"""Tests for the prediction pipeline."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import src.model_artifacts as model_artifacts
from src.feature_engineering import add_traffic_features
from src.model_artifacts import (
    ModelIntegrityError,
    metadata_path,
    save_model_artifacts,
    sha256_file,
    sidecar_path,
)
from src.predict import TrafficPredictionInput, TrafficPredictor
from src.preprocessing import FEATURE_COLUMNS, build_preprocessor


def _input(**overrides: object) -> TrafficPredictionInput:
    values = {
        "location_name": "Delhi",
        "latitude": 28.61,
        "longitude": 77.20,
        "current_speed": 18,
        "free_flow_speed": 50,
        "confidence": 0.9,
        "travel_time": 190,
        "timestamp": "2026-06-08T13:00:00+05:30",
    }
    values.update(overrides)
    return TrafficPredictionInput(**values)


@pytest.fixture
def saved_model(tmp_path: Path) -> Path:
    """Save a deterministic model: 'High' in IST peak hours, 'Low' otherwise."""

    rows = []
    for day in range(1, 11):
        for hour in (8, 13):
            rows.append(
                {
                    "timestamp": f"2026-06-{day:02d}T{hour:02d}:00:00+05:30",
                    "location_name": "Delhi",
                    "latitude": 28.61,
                    "longitude": 77.20,
                    "current_speed": 20.0 + day,
                    "free_flow_speed": 50.0,
                    "confidence": 0.9,
                    "travel_time": 150.0,
                }
            )
    featured = add_traffic_features(pd.DataFrame(rows))
    labels = np.where(featured["peak_hour"] == 1, "High", "Low")
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", DecisionTreeClassifier(random_state=0)),
        ]
    )
    pipeline.fit(featured[FEATURE_COLUMNS], labels)
    model_path = tmp_path / "models" / "traffic_model.pkl"
    save_model_artifacts(pipeline, model_path, {"best_model": "test"})
    return model_path


def test_predictor_returns_exact_class_and_probabilities(saved_model: Path) -> None:
    peak = TrafficPredictor(saved_model).predict(_input(timestamp="2026-06-08T18:00:00+05:30"))
    off_peak = TrafficPredictor(saved_model).predict(_input(timestamp="2026-06-08T13:00:00+05:30"))

    assert peak["prediction"] == "High"
    assert off_peak["prediction"] == "Low"
    for result in (peak, off_peak):
        assert set(result["probabilities"]) == {"High", "Low"}
        assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    assert peak["probabilities"]["High"] == pytest.approx(1.0)


def test_missing_model_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        TrafficPredictor(tmp_path / "absent.pkl").predict(_input())


def test_tampered_model_is_rejected_before_unpickling(
    saved_model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved_model.write_bytes(b"tampered")

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("joblib.load ran before the checksum check")

    monkeypatch.setattr(model_artifacts.joblib, "load", fail)
    with pytest.raises(ModelIntegrityError, match="Checksum mismatch"):
        TrafficPredictor(saved_model).predict(_input())


def test_missing_sidecar_is_rejected(saved_model: Path) -> None:
    sidecar_path(saved_model).unlink()

    with pytest.raises(ModelIntegrityError, match="Checksum file"):
        TrafficPredictor(saved_model).predict(_input())


def test_metadata_with_other_feature_columns_is_rejected(saved_model: Path) -> None:
    meta_file = metadata_path(saved_model)
    metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    metadata["feature_columns"] = metadata["feature_columns"][:-1]
    meta_file.write_text(json.dumps(metadata), encoding="utf-8")
    sidecar_path(saved_model).write_text(sha256_file(saved_model) + "\n", encoding="utf-8")

    with pytest.raises(ModelIntegrityError, match="feature columns"):
        TrafficPredictor(saved_model).predict(_input())


def test_invalid_timestamp_raises_value_error(saved_model: Path) -> None:
    with pytest.raises(ValueError, match="Invalid timestamp"):
        TrafficPredictor(saved_model).predict(_input(timestamp="not-a-time"))


def test_missing_speed_raises_value_error(saved_model: Path) -> None:
    with pytest.raises(ValueError, match="speed"):
        TrafficPredictor(saved_model).predict(_input(current_speed=float("nan")))


def test_default_timestamp_is_used_when_none(saved_model: Path) -> None:
    result = TrafficPredictor(saved_model).predict(_input(timestamp=None))

    assert result["prediction"] in {"High", "Low"}
    assert "hour" in result["features"]
