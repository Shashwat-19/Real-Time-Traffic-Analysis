"""Tests for evaluating a saved model on the training holdout."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

import src.evaluate as evaluate_module
from src.config import get_settings
from src.evaluate import evaluate_model
from src.model_artifacts import ModelIntegrityError
from src.train import train_models
from tests.test_train import _raw


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("evaluate")
    raw = _raw()
    model_path = root / "traffic_model.pkl"
    metrics = train_models(raw, model_path, root / "metrics.json")
    return {"raw": raw, "model_path": model_path, "metrics": metrics, "root": root}


def test_evaluate_model_scores_the_training_holdout(trained: dict) -> None:
    report = evaluate_model(trained["raw"], trained["model_path"])
    best = trained["metrics"]["models"][trained["metrics"]["best_model"]]

    # Same rows as training's test block, so support and accuracy must match exactly.
    assert report["macro avg"]["support"] == trained["metrics"]["split"]["test_rows"]
    assert report["accuracy"] == pytest.approx(best["test_accuracy"])


def test_evaluate_model_missing_model_raises(trained: dict) -> None:
    with pytest.raises(FileNotFoundError):
        evaluate_model(trained["raw"], trained["root"] / "absent.pkl")


def test_evaluate_model_rejects_tampered_model(trained: dict, tmp_path: Path) -> None:
    model_path = tmp_path / "traffic_model.pkl"
    train_models(trained["raw"], model_path, tmp_path / "metrics.json")
    model_path.write_bytes(b"tampered")

    with pytest.raises(ModelIntegrityError):
        evaluate_model(trained["raw"], model_path)


def test_main_writes_report_and_reports_missing_raw_data(
    trained: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_path = tmp_path / "raw.csv"
    trained["raw"].to_csv(raw_path, index=False)
    settings = replace(
        get_settings(),
        raw_data_path=raw_path,
        model_path=trained["model_path"],
        metrics_path=tmp_path / "reports" / "model_metrics.json",
    )
    monkeypatch.setattr(evaluate_module, "get_settings", lambda: settings)

    evaluate_module.main()
    report = json.loads((tmp_path / "reports" / "evaluation_report.json").read_text("utf-8"))
    assert "accuracy" in report

    monkeypatch.setattr(
        evaluate_module, "get_settings", lambda: replace(settings, raw_data_path=tmp_path / "x.csv")
    )
    with pytest.raises(FileNotFoundError, match="Raw data not found"):
        evaluate_module.main()
