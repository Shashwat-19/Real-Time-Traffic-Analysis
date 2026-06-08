"""Evaluate a saved traffic congestion model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.config import get_settings
from src.feature_engineering import add_traffic_features
from src.preprocessing import FEATURE_COLUMNS


def evaluate_model(dataframe: pd.DataFrame, model_path: Path) -> dict:
    """Evaluate a persisted model on a holdout split."""

    featured = add_traffic_features(dataframe)
    x = featured[FEATURE_COLUMNS]
    y = featured["congestion_level"]
    _, x_test, _, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    model = joblib.load(model_path)
    predictions = model.predict(x_test)
    return classification_report(y_test, predictions, output_dict=True)


def main() -> None:
    """CLI entry point for model evaluation."""

    settings = get_settings()
    dataframe = pd.read_csv(settings.raw_data_path)
    report = evaluate_model(dataframe, settings.model_path)
    output_path = settings.metrics_path.with_name("evaluation_report.json")
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved evaluation report to {output_path}")


if __name__ == "__main__":
    main()
