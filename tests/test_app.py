"""Dashboard prediction-form error handling via AppTest, with a stub predictor."""

from pathlib import Path
from typing import Any

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import src.predict as predict_module
from src.model_artifacts import ModelIntegrityError

APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "app.py")


class StubPredictor:
    error: Exception | None = None

    def __init__(self, model_path: Any) -> None:
        pass

    def predict(self, payload: Any) -> dict[str, Any]:
        if StubPredictor.error is not None:
            raise StubPredictor.error
        return {"prediction": "Low", "probabilities": {"Low": 1.0}, "features": {}}


def submit_form(monkeypatch, error: Exception | None) -> AppTest:
    StubPredictor.error = error
    monkeypatch.setattr(predict_module, "TrafficPredictor", StubPredictor)
    st.cache_resource.clear()
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    at.button[0].click().run()
    return at


@pytest.fixture(autouse=True)
def reset_stub():
    yield
    StubPredictor.error = None


def test_valid_prediction_shows_success(monkeypatch):
    at = submit_form(monkeypatch, None)

    assert not at.exception
    assert any("Predicted congestion level: Low" in s.value for s in at.success)


def test_value_error_is_shown_as_invalid_input(monkeypatch):
    at = submit_form(monkeypatch, ValueError("Invalid timestamp: 'x'"))

    assert not at.exception
    assert at.error[0].value == "Invalid input: Invalid timestamp: 'x'"


def test_model_integrity_error_shows_generic_message(monkeypatch):
    at = submit_form(monkeypatch, ModelIntegrityError("Checksum mismatch for /x/model.pkl"))

    assert not at.exception
    assert "Checksum" not in at.error[0].value
    assert "Prediction failed" in at.error[0].value
