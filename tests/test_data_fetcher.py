"""Tests for TomTom response parsing, fetch error handling and retries."""

from typing import Any

import pytest
import requests

from src.config import Location
from src.data_fetcher import TomTomTrafficFetcher

DELHI = Location("Delhi", 28.61, 77.20)
SECRET_KEY = "SECRET-KEY-123"


def valid_payload(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "currentSpeed": 32,
        "freeFlowSpeed": 55,
        "confidence": 0.93,
        "currentTravelTime": 120,
        "roadClosure": False,
        "frc": "FRC2",
        "freeFlowTravelTime": 90,
        "coordinates": {"coordinate": [{"latitude": 28.6, "longitude": 77.2}]},
    }
    data.update(overrides)
    return {"flowSegmentData": data}


def test_parse_response_extracts_required_fields() -> None:
    payload = {
        "flowSegmentData": {
            "currentSpeed": 32,
            "freeFlowSpeed": 55,
            "confidence": 0.93,
            "currentTravelTime": 120,
            "roadClosure": False,
            "frc": "FRC2",
            "freeFlowTravelTime": 90,
            "coordinates": {"coordinate": [{"latitude": 28.6, "longitude": 77.2}]},
        }
    }
    result = TomTomTrafficFetcher.parse_response(payload, Location("Delhi", 28.61, 77.20))

    assert result["location_name"] == "Delhi"
    assert result["latitude"] == 28.6
    assert result["longitude"] == 77.2
    assert result["current_speed"] == 32.0
    assert result["free_flow_speed"] == 55.0
    assert result["confidence"] == 0.93
    assert result["travel_time"] == 120.0
    assert result["road_closure"] is False


@pytest.mark.parametrize(
    "field", ["currentSpeed", "freeFlowSpeed", "confidence", "currentTravelTime"]
)
def test_parse_response_rejects_null_or_missing_required_field(field: str) -> None:
    null_payload = valid_payload(**{field: None})
    missing_payload = valid_payload()
    del missing_payload["flowSegmentData"][field]

    for payload in (null_payload, missing_payload):
        with pytest.raises(ValueError, match=field):
            TomTomTrafficFetcher.parse_response(payload, DELHI)


def test_parse_response_rejects_non_numeric_speed() -> None:
    with pytest.raises(ValueError, match="currentSpeed"):
        TomTomTrafficFetcher.parse_response(valid_payload(currentSpeed="32"), DELHI)


@pytest.mark.parametrize("payload", [None, [], "text", {"flowSegmentData": ["x"]}])
def test_parse_response_rejects_wrong_payload_shape(payload: Any) -> None:
    with pytest.raises(ValueError):
        TomTomTrafficFetcher.parse_response(payload, DELHI)


def test_parse_response_falls_back_to_location_for_empty_coordinates() -> None:
    for coordinates in ({"coordinate": []}, {}, None):
        result = TomTomTrafficFetcher.parse_response(valid_payload(coordinates=coordinates), DELHI)

        assert result["latitude"] == DELHI.latitude
        assert result["longitude"] == DELHI.longitude


def test_parse_response_rejects_malformed_coordinates() -> None:
    with pytest.raises(ValueError):
        TomTomTrafficFetcher.parse_response(valid_payload(coordinates={"coordinate": "x"}), DELHI)


@pytest.mark.parametrize("road_closure", [None, "false", 0])
def test_parse_response_rejects_missing_or_non_boolean_road_closure(road_closure: Any) -> None:
    payload = valid_payload()
    if road_closure is None:
        del payload["flowSegmentData"]["roadClosure"]
    else:
        payload["flowSegmentData"]["roadClosure"] = road_closure

    with pytest.raises(ValueError, match="roadClosure"):
        TomTomTrafficFetcher.parse_response(payload, DELHI)


@pytest.mark.parametrize("confidence", [-0.1, 1.5])
def test_parse_response_rejects_confidence_outside_unit_interval(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        TomTomTrafficFetcher.parse_response(valid_payload(confidence=confidence), DELHI)


def test_parse_response_rejects_missing_flow_segment_data() -> None:
    error_body = {
        "error": "Forbidden",
        "httpStatusCode": 403,
        "detailedError": {"code": "FORBIDDEN", "message": "Invalid API key"},
    }

    with pytest.raises(ValueError, match="flowSegmentData"):
        TomTomTrafficFetcher.parse_response(error_body, DELHI)


class FakeResponse:
    def __init__(self, status_code: int, payload: Any = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            # Mirrors requests: the message contains the full URL, key included.
            raise requests.HTTPError(f"{self.status_code} for url https://x?key={SECRET_KEY}")

    def json(self) -> Any:
        return self._payload


def fake_get(monkeypatch: pytest.MonkeyPatch, responses: list[Any]) -> list[dict[str, Any]]:
    """Replace requests.get with a scripted sequence; returns the recorded calls."""

    calls: list[dict[str, Any]] = []

    def get(url: str, params: dict[str, Any], timeout: int) -> Any:
        calls.append({"url": url, "params": params, "timeout": timeout})
        item = responses[len(calls) - 1]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(requests, "get", get)
    return calls


def test_fetch_many_drops_bad_payload_and_keeps_good_locations(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake_get(
        monkeypatch,
        [FakeResponse(200, ["not", "an", "object"]), FakeResponse(200, valid_payload())],
    )
    fetcher = TomTomTrafficFetcher(SECRET_KEY, sleep=lambda _: None)

    records = fetcher.fetch_many((DELHI, Location("Mumbai", 19.07, 72.87)))

    assert [record["location_name"] for record in records] == ["Mumbai"]
    assert "Dropped Delhi: ValueError" in caplog.text


def test_fetch_many_never_logs_the_key(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    error = requests.ConnectionError(f"connection failed for https://x?key={SECRET_KEY}")
    fake_get(monkeypatch, [error])
    fetcher = TomTomTrafficFetcher(SECRET_KEY, sleep=lambda _: None)

    records = fetcher.fetch_many((DELHI,))

    assert records == []
    assert "Dropped Delhi: ConnectionError" in caplog.text
    assert SECRET_KEY not in caplog.text
    assert SECRET_KEY not in capsys.readouterr().err


def test_fetch_location_sends_key_as_params(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_get(monkeypatch, [FakeResponse(200, valid_payload())])
    fetcher = TomTomTrafficFetcher(SECRET_KEY, sleep=lambda _: None)

    fetcher.fetch_location(DELHI)

    assert calls[0]["params"]["key"] == SECRET_KEY
    assert SECRET_KEY not in calls[0]["url"]


def test_fetch_location_retries_429_and_5xx_with_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_get(
        monkeypatch,
        [FakeResponse(429), FakeResponse(503), FakeResponse(200, valid_payload())],
    )
    sleeps: list[float] = []
    fetcher = TomTomTrafficFetcher(SECRET_KEY, sleep=sleeps.append)

    record = fetcher.fetch_location(DELHI)

    assert record["current_speed"] == 32.0
    assert len(calls) == 3
    assert sleeps == [1, 2]


def test_fetch_location_gives_up_after_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_get(monkeypatch, [FakeResponse(500)] * 3)
    sleeps: list[float] = []
    fetcher = TomTomTrafficFetcher(SECRET_KEY, sleep=sleeps.append)

    with pytest.raises(requests.HTTPError):
        fetcher.fetch_location(DELHI)

    assert len(calls) == 3
    assert sleeps == [1, 2]


def test_fetch_location_does_not_retry_client_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_get(monkeypatch, [FakeResponse(403)])
    sleeps: list[float] = []
    fetcher = TomTomTrafficFetcher(SECRET_KEY, sleep=sleeps.append)

    with pytest.raises(requests.HTTPError):
        fetcher.fetch_location(DELHI)

    assert len(calls) == 1
    assert sleeps == []


ERROR_BODY = {
    "error": "Forbidden",
    "httpStatusCode": 403,
    "detailedError": {"code": "FORBIDDEN", "message": "Invalid API key"},
}


@pytest.mark.parametrize(
    ("status", "exc_name"), [(200, "ValueError"), (403, "HTTPError"), (429, "HTTPError")]
)
def test_fetch_many_drops_documented_error_body(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, status: int, exc_name: str
) -> None:
    fake_get(monkeypatch, [FakeResponse(status, ERROR_BODY)] * 3)
    fetcher = TomTomTrafficFetcher(SECRET_KEY, sleep=lambda _: None)

    assert fetcher.fetch_many((DELHI,)) == []
    assert f"Dropped Delhi: {exc_name}" in caplog.text
