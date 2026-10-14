"""Evaluate a saved traffic congestion model."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

from src.config import get_settings
from src.feature_engineering import ensure_features
from src.model_artifacts import load_verified_model
from src.preprocessing import split_chronologically


def evaluate_model(dataframe: pd.DataFrame, model_path: Path) -> dict:
    """Evaluate a persisted model on the same chronological test block training used."""

    featured = ensure_features(dataframe)
    _, x_test, _, y_test = split_chronologically(featured)
    model, _ = load_verified_model(model_path)
    predictions = model.predict(x_test)
    return classification_report(y_test, predictions, output_dict=True)


def main() -> None:
    """CLI entry point for model evaluation."""

    settings = get_settings()
    if not settings.raw_data_path.exists():
        raise FileNotFoundError(f"Raw data not found at {settings.raw_data_path}; run collector")
    dataframe = pd.read_csv(settings.raw_data_path)
    report = evaluate_model(dataframe, settings.model_path)
    output_path = settings.metrics_path.with_name("evaluation_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved evaluation report to {output_path}")


if __name__ == "__main__":
    main()
