"""Train and compare traffic congestion models."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.config import get_settings
from src.feature_engineering import add_traffic_features
from src.preprocessing import FEATURE_COLUMNS, build_preprocessor

logger = logging.getLogger(__name__)


def build_models(random_state: int = 42) -> dict[str, object]:
    """Create candidate estimators."""

    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, random_state=random_state, class_weight="balanced"
        ),
        "XGBoost": XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_estimators=150,
            learning_rate=0.08,
            max_depth=4,
            random_state=random_state,
        ),
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def train_models(
    dataframe: pd.DataFrame,
    model_path: Path,
    metrics_path: Path,
    random_state: int = 42,
) -> dict[str, object]:
    """Train candidate models and persist the best pipeline."""

    featured = add_traffic_features(dataframe)
    if len(featured) < 12:
        raise ValueError("At least 12 records are recommended for train/test split and CV")

    x = featured[FEATURE_COLUMNS]
    y = featured["congestion_level"]
    if y.nunique() < 2:
        raise ValueError("Training requires at least two congestion classes")

    stratify = y if y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state, stratify=stratify
    )

    cv_splits = max(2, min(5, int(y_train.value_counts().min())))
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    results: dict[str, dict[str, object]] = {}
    best_name = ""
    best_pipeline: Pipeline | None = None
    best_score = -1.0

    for name, estimator in build_models(random_state).items():
        pipeline = Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", estimator)])
        cv_scores = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="f1_macro")
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        test_f1 = f1_score(y_test, predictions, average="macro")
        results[name] = {
            "cv_f1_macro_mean": float(cv_scores.mean()),
            "cv_f1_macro_std": float(cv_scores.std()),
            "test_accuracy": float(accuracy_score(y_test, predictions)),
            "test_f1_macro": float(test_f1),
            "classification_report": classification_report(y_test, predictions, output_dict=True),
        }
        if test_f1 > best_score:
            best_name = name
            best_pipeline = pipeline
            best_score = test_f1

    if best_pipeline is None:
        raise RuntimeError("No model was trained")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_path)

    metrics = {"best_model": best_name, "best_test_f1_macro": best_score, "models": results}
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Saved best model %s to %s", best_name, model_path)
    return metrics


def main() -> None:
    """CLI entry point for model training."""

    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    raw_df = pd.read_csv(settings.raw_data_path)
    featured = add_traffic_features(raw_df)
    featured.to_csv(settings.processed_data_path, index=False)
    metrics = train_models(featured, settings.model_path, settings.metrics_path)
    logger.info("Training complete: %s", metrics["best_model"])


if __name__ == "__main__":
    main()
