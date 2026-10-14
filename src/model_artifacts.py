"""Save and verify trained model artifacts: pickle, SHA-256 sidecar and metadata."""

from __future__ import annotations

import hashlib
import hmac
import json
import sys
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier

from src.preprocessing import FEATURE_COLUMNS


# Defined here, not in train.py, so pickles reference src.model_artifacts rather than __main__.
class XGBoostLabelClassifier(ClassifierMixin, BaseEstimator):
    """XGBoost classifier that takes string labels and works inside sklearn Pipelines.

    xgboost 2.x needs integer class codes, and subclassing XGBClassifier breaks sklearn 1.6
    tag lookup in Pipeline.predict, so this wraps it and maps the labels itself.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        self.model_ = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_estimators=150,
            learning_rate=0.08,
            max_depth=4,
            random_state=self.random_state,
        )
        self.model_.fit(X, np.searchsorted(self.classes_, y), sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.classes_[self.model_.predict(X).astype(int)]

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

LIBRARIES = ["scikit-learn", "xgboost", "pandas", "numpy", "joblib"]


class ModelIntegrityError(RuntimeError):
    """The model file, its checksum or its metadata is missing or does not match."""


def sidecar_path(model_path: Path) -> Path:
    """Path of the SHA-256 sidecar next to the model."""

    return model_path.with_name(model_path.name + ".sha256")


def metadata_path(model_path: Path) -> Path:
    """Path of the metadata JSON next to the model."""

    return model_path.with_name(model_path.stem + ".meta.json")


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_model_artifacts(
    pipeline: Any, model_path: Path, details: dict[str, Any]
) -> dict[str, Any]:
    """Write the model, its SHA-256 sidecar and metadata (features, classes, versions)."""

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metadata = {
        **details,
        "feature_columns": list(FEATURE_COLUMNS),
        "classes": [str(label) for label in pipeline.classes_],
        "library_versions": {
            "python": sys.version.split()[0],
            **{name: importlib_metadata.version(name) for name in LIBRARIES},
        },
        "created_at": datetime.now(UTC).isoformat(),
    }
    metadata_path(model_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    sidecar_path(model_path).write_text(sha256_file(model_path) + "\n", encoding="utf-8")
    return metadata


def load_verified_model(model_path: Path) -> tuple[Any, dict[str, Any]]:
    """Check the checksum and metadata, then unpickle the model.

    The checksum is verified before joblib.load runs, because unpickling executes code.
    """

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    sidecar = sidecar_path(model_path)
    if not sidecar.exists():
        raise ModelIntegrityError(f"Checksum file {sidecar} is missing; retrain the model")
    expected = sidecar.read_text(encoding="utf-8").strip()
    if not hmac.compare_digest(sha256_file(model_path), expected):
        raise ModelIntegrityError(f"Checksum mismatch for {model_path}; refusing to load it")

    meta_file = metadata_path(model_path)
    if not meta_file.exists():
        raise ModelIntegrityError(f"Metadata file {meta_file} is missing; retrain the model")
    metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    if metadata.get("feature_columns") != FEATURE_COLUMNS:
        raise ModelIntegrityError("Model feature columns do not match this code; retrain the model")
    # Pickle is needed for the sklearn pipeline. Only the service-owned model path is loaded,
    # and the checksum above rejects a changed file. It does not authenticate a deliberate
    # replacement of both the model and its sidecar.
    return joblib.load(model_path), metadata
