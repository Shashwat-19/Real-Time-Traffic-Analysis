"""Tests for TomTom response parsing."""

from src.config import Location
from src.data_fetcher import TomTomTrafficFetcher


def test_parse_response_extracts_required_fields() -> None:
    payload = {
        "flowSegmentData": {
            "currentSpeed": 32,
            "freeFlowSpeed": 55,
            "confidence": 0.93,
            "currentTravelTime": 120,
            "roadClosure": False,
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
