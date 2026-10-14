"""Tests for model training on small synthetic data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

import src.feature_engineering as feature_engineering
import src.train as train_module
from src.feature_engineering import add_traffic_features
from src.model_artifacts import metadata_path, sidecar_path
from src.predict import TrafficPredictionInput, TrafficPredictor
from src.preprocessing import FEATURE_COLUMNS
from src.train import XGBoostLabelClassifier, train_models


def _raw(count: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-06-01T00:00:00Z")
    rows = []
    for i in range(count):
        delhi = i % 2 == 1
        rows.append(
            {
                "timestamp": (start + pd.Timedelta(minutes=10 * i)).isoformat(),
                "location_name": "Delhi" if delhi else "Mumbai",
                "latitude": 28.61 if delhi else 19.07,
                "longitude": 77.20 if delhi else 72.87,
                "current_speed": float(rng.uniform(5, 55)),
                "free_flow_speed": 50.0,
                "confidence": 0.9,
                "travel_time": 100.0,
                "road_closure": False,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("trained")
    model_path = root / "models" / "nested" / "traffic_model.pkl"
    metrics_path = root / "outputs" / "metrics" / "model_metrics.json"
    metrics = train_models(_raw(), model_path, metrics_path)
    return {"metrics": metrics, "model_path": model_path, "metrics_path": metrics_path}


def test_train_models_reports_expected_metric_keys(trained: dict) -> None:
    metrics = trained["metrics"]

    assert {"best_model", "best_test_f1_macro", "split", "models"} <= set(metrics)
    assert set(metrics["models"]) == {"Logistic Regression", "Random Forest", "XGBoost", "SVM"}
    for entry in metrics["models"].values():
        assert {"cv_f1_macro_mean", "cv_f1_macro_std", "training_rows"} <= set(entry)
    best = metrics["models"][metrics["best_model"]]
    assert {"test_accuracy", "test_f1_macro", "classification_report"} <= set(best)
    assert best["test_f1_macro"] == metrics["best_test_f1_macro"]


def test_best_model_is_argmax_of_cv_score_and_test_scored_once(trained: dict) -> None:
    metrics = trained["metrics"]
    cv_means = {name: entry["cv_f1_macro_mean"] for name, entry in metrics["models"].items()}

    assert metrics["best_model"] == max(cv_means, key=cv_means.get)
    for name, entry in metrics["models"].items():
        if name != metrics["best_model"]:
            assert "test_f1_macro" not in entry


def test_split_is_chronological_and_saved(trained: dict) -> None:
    split = trained["metrics"]["split"]

    assert split["method"] == "chronological"
    assert split["test_rows"] == 24
    assert split["train_rows"] == 96
    saved = json.loads(trained["metrics_path"].read_text(encoding="utf-8"))
    assert saved["best_model"] == trained["metrics"]["best_model"]


def test_outputs_are_written_with_sidecar_and_metadata(trained: dict) -> None:
    model_path = trained["model_path"]

    assert model_path.exists()
    assert sidecar_path(model_path).exists()
    # Pickles must not reference __main__, or loading from the API or another process fails.
    assert b"__main__" not in model_path.read_bytes()
    metadata = json.loads(metadata_path(model_path).read_text(encoding="utf-8"))
    assert metadata["feature_columns"] == FEATURE_COLUMNS
    assert metadata["best_model"] == trained["metrics"]["best_model"]
    assert set(metadata["classes"]) == {"Low", "Medium", "High"}
    assert "scikit-learn" in metadata["library_versions"]


def test_saved_model_reloads_through_predictor(trained: dict) -> None:
    result = TrafficPredictor(trained["model_path"]).predict(
        TrafficPredictionInput(
            location_name="Delhi",
            latitude=28.61,
            longitude=77.20,
            current_speed=20,
            free_flow_speed=50,
            confidence=0.9,
            travel_time=100,
            timestamp="2026-06-08T18:00:00+05:30",
        )
    )

    assert result["prediction"] in {"Low", "Medium", "High"}
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)


def test_train_models_does_not_refeaturise_a_featured_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    featured = add_traffic_features(_raw())

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("featurised twice")

    monkeypatch.setattr(feature_engineering, "add_traffic_features", fail)
    metrics = train_models(featured, tmp_path / "m.pkl", tmp_path / "metrics.json")

    assert metrics["split"]["test_rows"] == 24


def test_train_models_rejects_too_little_data(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="At least 25 records"):
        train_models(_raw(count=10), tmp_path / "m.pkl", tmp_path / "metrics.json")


def test_svm_training_rows_are_capped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(train_module, "SVM_MAX_TRAIN_ROWS", 30)

    metrics = train_models(_raw(), tmp_path / "m.pkl", tmp_path / "metrics.json")

    assert metrics["models"]["SVM"]["training_rows"] == 30
    assert metrics["models"]["XGBoost"]["training_rows"] == 96


def test_xgboost_wrapper_predicts_string_labels_inside_pipeline() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 3))
    y = np.array(["Low", "Medium", "High"] * 20)
    pipeline = Pipeline(steps=[("scaler", StandardScaler()), ("model", XGBoostLabelClassifier())])

    pipeline.fit(x, y, model__sample_weight=compute_sample_weight("balanced", y))

    assert list(pipeline.classes_) == ["High", "Low", "Medium"]
    assert set(pipeline.predict(x)) <= {"Low", "Medium", "High"}
    assert pipeline.predict_proba(x).shape == (60, 3)
