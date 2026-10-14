"""FastAPI endpoint tests with temporary data and a stub predictor (no real model file)."""

from __future__ import annotations

import dataclasses
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from src.config import get_settings
from src.data_storage import RAW_COLUMNS
from src.model_artifacts import ModelIntegrityError


class StubPredictor:
    """Stands in for TrafficPredictor; counts how often it is constructed."""

    instances = 0
    error: Exception | None = None

    def __init__(self, model_path: Any) -> None:
        StubPredictor.instances += 1

    def predict(self, payload: Any) -> dict[str, Any]:
        if StubPredictor.error is not None:
            raise StubPredictor.error
        return {
            "prediction": "Medium",
            "probabilities": {"High": 0.2, "Low": 0.3, "Medium": 0.5},
            "features": {},
        }


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """Point the API at a temporary raw file and model path."""

    base = dataclasses.replace(
        get_settings(),
        raw_data_path=tmp_path / "raw" / "traffic_data.csv",
        processed_data_path=tmp_path / "processed" / "features.csv",
        model_path=tmp_path / "models" / "traffic_model.pkl",
    )
    monkeypatch.setattr(api_main, "get_settings", lambda: base)
    api_main.reset_predictor()
    yield base
    api_main.reset_predictor()


@pytest.fixture
def client(settings):
    return TestClient(api_main.app)


@pytest.fixture
def stub_predictor(monkeypatch):
    StubPredictor.instances = 0
    StubPredictor.error = None
    monkeypatch.setattr(api_main, "TrafficPredictor", StubPredictor)
    api_main.reset_predictor()
    yield StubPredictor
    api_main.reset_predictor()


def raw_row(timestamp, location="Pune", latitude=18.5, longitude=73.8, speed=30.0):
    return {
        "timestamp": timestamp,
        "location_name": location,
        "latitude": latitude,
        "longitude": longitude,
        "current_speed": speed,
        "free_flow_speed": 60.0,
        "confidence": 0.9,
        "travel_time": 100.0,
        "road_closure": False,
    }


def write_raw(settings, rows):
    settings.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=RAW_COLUMNS).to_csv(settings.raw_data_path, index=False)


PREDICT_BODY = {
    "location_name": "Pune",
    "latitude": 18.5,
    "longitude": 73.8,
    "current_speed": 30,
    "free_flow_speed": 60,
    "confidence": 0.9,
    "travel_time": 100,
    "road_closure": False,
    "timestamp": "2026-06-08T08:00:00+00:00",
}


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_latest_empty_when_no_file(client):
    response = client.get("/traffic/latest")

    assert response.status_code == 200
    assert response.json() == []


def test_latest_returns_newest_row_per_location(client, settings):
    write_raw(
        settings,
        [
            raw_row("2026-06-08T08:00:00+00:00", speed=10.0),
            raw_row("2026-06-08T09:00:00+00:00", speed=45.0),
            raw_row("2026-06-08T08:30:00+00:00", location="Delhi", speed=20.0),
        ],
    )

    data = client.get("/traffic/latest").json()

    assert sorted(item["location_name"] for item in data) == ["Delhi", "Pune"]
    pune = next(item for item in data if item["location_name"] == "Pune")
    assert pune["current_speed"] == 45.0


def test_latest_drops_rows_with_missing_coordinates(client, settings):
    write_raw(
        settings,
        [
            raw_row("2026-06-08T08:00:00+00:00", location="Pune"),
            raw_row("2026-06-08T08:00:00+00:00", location="Delhi", latitude=None),
        ],
    )

    data = client.get("/traffic/latest").json()

    assert [item["location_name"] for item in data] == ["Pune"]


@pytest.mark.parametrize("limit", [0, 501])
def test_latest_limit_bounds(client, limit):
    assert client.get("/traffic/latest", params={"limit": limit}).status_code == 422


def test_history_empty_when_no_file(client):
    response = client.get("/traffic/history")

    assert response.status_code == 200
    assert response.json() == []


def test_history_returns_newest_rows_up_to_limit(client, settings):
    write_raw(
        settings,
        [raw_row(f"2026-06-08T0{hour}:00:00+00:00") for hour in range(1, 6)],
    )

    data = client.get("/traffic/history", params={"limit": 2}).json()

    assert [item["timestamp"][11:13] for item in data] == ["05", "04"]


def test_history_since_filter_is_inclusive(client, settings):
    write_raw(
        settings,
        [
            raw_row("2026-06-08T08:00:00+00:00"),
            raw_row("2026-06-08T10:00:00+00:00"),
            raw_row("2026-06-08T12:00:00+00:00"),
        ],
    )

    aware = client.get("/traffic/history", params={"since": "2026-06-08T10:00:00+00:00"})
    naive = client.get("/traffic/history", params={"since": "2026-06-08T10:00:00"})

    assert len(aware.json()) == 2
    assert len(naive.json()) == 2


def test_history_rejects_bad_since(client):
    response = client.get("/traffic/history", params={"since": "yesterday"})

    assert response.status_code == 422


@pytest.mark.parametrize("limit", [0, 5001])
def test_history_limit_bounds(client, limit):
    assert client.get("/traffic/history", params={"limit": limit}).status_code == 422


def test_history_nan_coordinates_return_valid_json(client, settings):
    write_raw(
        settings,
        [
            raw_row("2026-06-08T08:00:00+00:00"),
            raw_row("2026-06-08T09:00:00+00:00", latitude="not-a-number"),
        ],
    )

    response = client.get("/traffic/history")

    assert response.status_code == 200
    assert [item["latitude"] for item in response.json()] == [18.5]


def test_traffic_read_failure_returns_generic_500(client, monkeypatch):
    def broken_repository():
        raise RuntimeError("disk on fire /secret/path")

    monkeypatch.setattr(api_main, "repository", broken_repository)

    response = client.get("/traffic/history")

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "secret" not in response.text


def test_predict_success_and_predictor_is_built_once(client, stub_predictor):
    first = client.post("/predict", json=PREDICT_BODY)
    second = client.post("/predict", json=PREDICT_BODY)

    assert first.status_code == 200
    assert second.status_code == 200
    body = first.json()
    assert body["prediction"] == "Medium"
    assert sum(body["probabilities"].values()) == pytest.approx(1.0)
    assert stub_predictor.instances == 1


def test_predict_invalid_timestamp_is_422_naming_field(client, stub_predictor):
    response = client.post("/predict", json={**PREDICT_BODY, "timestamp": "not a time"})

    assert response.status_code == 422
    assert "timestamp" in response.text


def test_predict_latitude_out_of_range_is_422(client, stub_predictor):
    response = client.post("/predict", json={**PREDICT_BODY, "latitude": 91})

    assert response.status_code == 422
    assert "latitude" in response.text


def test_predict_missing_model_is_503(client, settings):
    response = client.post("/predict", json=PREDICT_BODY)

    assert response.status_code == 503
    assert response.json() == {"detail": "Model is not available. Train the model first."}


def test_predict_model_integrity_error_is_generic_500(client, stub_predictor):
    stub_predictor.error = ModelIntegrityError("Checksum mismatch for /x/model.pkl")

    response = client.post("/predict", json=PREDICT_BODY)

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "Checksum" not in response.text


@pytest.mark.parametrize(
    "message",
    ["Invalid timestamp: 'x'", "Prediction input has missing or non-numeric speed values"],
)
def test_predict_value_error_is_422_with_message(client, stub_predictor, message):
    stub_predictor.error = ValueError(message)

    response = client.post("/predict", json=PREDICT_BODY)

    assert response.status_code == 422
    assert response.json() == {"detail": message}


def test_predict_unexpected_error_is_generic_500(client, stub_predictor):
    stub_predictor.error = RuntimeError("internal detail /private/model.pkl")

    response = client.post("/predict", json=PREDICT_BODY)

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "internal detail" not in response.text
    assert "Traceback" not in response.text
