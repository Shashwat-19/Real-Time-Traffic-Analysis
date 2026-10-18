"""Train and compare traffic congestion models."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from src.config import get_settings
from src.feature_engineering import ensure_features
from src.model_artifacts import XGBoostLabelClassifier, save_model_artifacts
from src.preprocessing import build_preprocessor, exclude_repeats, split_chronologically

logger = logging.getLogger(__name__)

MIN_RECORDS = 25
MAX_CV_FOLDS = 5
# SVC training cost grows faster than linearly with rows, so SVM only sees the most recent rows.
SVM_MAX_TRAIN_ROWS = 3000


def build_models(random_state: int = 42) -> dict[str, object]:
    """Create candidate estimators."""

    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, random_state=random_state, class_weight="balanced"
        ),
        "XGBoost": XGBoostLabelClassifier(random_state=random_state),
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def _pipeline(estimator: object) -> Pipeline:
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", estimator)])


def _fit(name: str, pipeline: Pipeline, x: pd.DataFrame, y: pd.Series) -> Pipeline:
    if name == "XGBoost":
        # XGBoost has no class_weight option, so balance it with sample weights to compare fairly.
        pipeline.fit(x, y, model__sample_weight=compute_sample_weight("balanced", y))
    else:
        pipeline.fit(x, y)
    return pipeline


def _training_rows(name: str, x: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    if name == "SVM":
        return x.tail(SVM_MAX_TRAIN_ROWS), y.tail(SVM_MAX_TRAIN_ROWS)
    return x, y


def _time_series_cv_scores(
    name: str, estimator: object, x: pd.DataFrame, y: pd.Series, n_splits: int
) -> np.ndarray:
    """Macro-F1 per forward-chaining fold: train on the past, validate on the next block."""

    scores = []
    for train_index, valid_index in TimeSeriesSplit(n_splits=n_splits).split(x):
        if y.iloc[train_index].nunique() < 2:
            continue  # an early fold can contain one class only, so it cannot be fitted
        pipeline = _fit(name, _pipeline(clone(estimator)), x.iloc[train_index], y.iloc[train_index])
        predictions = pipeline.predict(x.iloc[valid_index])
        scores.append(f1_score(y.iloc[valid_index], predictions, average="macro"))
    if not scores:
        raise ValueError("Not enough chronological data to cross-validate; add more records")
    return np.array(scores)


def train_models(
    dataframe: pd.DataFrame,
    model_path: Path,
    metrics_path: Path,
    random_state: int = 42,
) -> dict[str, object]:
    """Train candidate models and persist the best pipeline.

    Rows are split by time (the most recent block is the test set). Models are ranked
    by mean macro-F1 across TimeSeriesSplit folds of the training block. The test block
    is scored once, for the selected model only.
    """

    featured = ensure_features(exclude_repeats(dataframe))
    if len(featured) < MIN_RECORDS:
        raise ValueError(
            f"At least {MIN_RECORDS} records are needed for a train/test split and CV, "
            f"got {len(featured)}"
        )
    if featured["congestion_level"].nunique() < 2:
        raise ValueError("Training requires at least two congestion classes")

    x_train, x_test, y_train, y_test = split_chronologically(featured)
    n_splits = min(MAX_CV_FOLDS, len(x_train) // 10)

    candidates = build_models(random_state)
    results: dict[str, dict[str, Any]] = {}
    for name, estimator in candidates.items():
        fit_x, fit_y = _training_rows(name, x_train, y_train)
        cv_scores = _time_series_cv_scores(name, estimator, fit_x, fit_y, n_splits)
        results[name] = {
            "cv_f1_macro_mean": float(cv_scores.mean()),
            "cv_f1_macro_std": float(cv_scores.std()),
            "training_rows": len(fit_x),
        }

    best_name = max(results, key=lambda name: results[name]["cv_f1_macro_mean"])
    fit_x, fit_y = _training_rows(best_name, x_train, y_train)
    best_pipeline = _fit(best_name, _pipeline(candidates[best_name]), fit_x, fit_y)

    predictions = best_pipeline.predict(x_test)
    test_f1 = float(f1_score(y_test, predictions, average="macro"))
    results[best_name].update(
        {
            "test_accuracy": float(accuracy_score(y_test, predictions)),
            "test_f1_macro": test_f1,
            "classification_report": classification_report(y_test, predictions, output_dict=True),
        }
    )

    split = {
        "method": "chronological",
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "cv_folds": n_splits,
    }
    save_model_artifacts(
        best_pipeline,
        model_path,
        {"best_model": best_name, "split": split, "random_state": random_state},
    )

    metrics = {
        "best_model": best_name,
        "best_test_f1_macro": test_f1,
        "split": split,
        "models": results,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Saved best model %s to %s", best_name, model_path)
    return metrics


def main() -> None:
    """CLI entry point for model training."""

    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    if not settings.raw_data_path.exists():
        raise FileNotFoundError(f"Raw data not found at {settings.raw_data_path}; run collector")
    raw_df = pd.read_csv(settings.raw_data_path)
    featured = ensure_features(raw_df)
    settings.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(settings.processed_data_path, index=False)
    metrics = train_models(featured, settings.model_path, settings.metrics_path)
    logger.info("Training complete: %s", metrics["best_model"])


if __name__ == "__main__":
    main()
