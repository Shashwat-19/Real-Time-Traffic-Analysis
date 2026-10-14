"""TomTom Traffic Flow API integration."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import requests

from src.config import Location

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3


def _object(value: Any, field: str) -> dict[str, Any]:
    """Return a JSON object, treating null as empty. Anything else is a bad payload."""

    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field} is not an object")
    return value


def _number(value: Any, field: str) -> float:
    """Return a JSON number as float. Strings, booleans and lists are rejected."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} is not a number")
    return float(value)


def _required_number(data: dict[str, Any], field: str) -> float:
    """Return a required number. A missing or null field is an error, never a default."""

    if data.get(field) is None:
        raise ValueError(f"missing {field}")
    return _number(data[field], field)


class TomTomTrafficFetcher:
    """Fetch traffic flow segment data from TomTom."""

    BASE_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"

    def __init__(
        self,
        api_key: str,
        timeout_seconds: int = 20,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise ValueError("TomTom API key is required")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._sleep = sleep

    def fetch_location(self, location: Location) -> dict[str, Any]:
        """Fetch and normalize one traffic record for a location."""

        params = {
            "point": f"{location.latitude},{location.longitude}",
            "unit": "KMPH",
            "key": self.api_key,
        }
        response = requests.get(self.BASE_URL, params=params, timeout=self.timeout_seconds)
        for attempt in range(1, MAX_ATTEMPTS):
            if response.status_code != 429 and response.status_code < 500:
                break
            # Backoff of 1s, then 2s, before each retry.
            self._sleep(2 ** (attempt - 1))
            response = requests.get(self.BASE_URL, params=params, timeout=self.timeout_seconds)
        response.raise_for_status()
        return self.parse_response(response.json(), location)

    @staticmethod
    def parse_response(payload: Any, location: Location) -> dict[str, Any]:
        """Extract the fields required by the downstream pipeline.

        Raises ValueError for a malformed payload or a missing required field.
        Coordinates fall back to the configured location; speeds never do.
        """

        data = _object(_object(payload, "payload").get("flowSegmentData", payload), "data")
        coordinates = _object(data.get("coordinates"), "coordinates")
        points = coordinates.get("coordinate") or []
        if not isinstance(points, list):
            raise ValueError("coordinate is not a list")
        point = _object(points[0] if points else None, "coordinate")

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "location_name": location.name,
            "latitude": location.latitude
            if point.get("latitude") is None
            else _number(point["latitude"], "latitude"),
            "longitude": location.longitude
            if point.get("longitude") is None
            else _number(point["longitude"], "longitude"),
            "current_speed": _required_number(data, "currentSpeed"),
            "free_flow_speed": _required_number(data, "freeFlowSpeed"),
            "confidence": _required_number(data, "confidence"),
            "travel_time": _required_number(data, "currentTravelTime"),
            "road_closure": bool(data.get("roadClosure", False)),
        }

    def fetch_many(self, locations: tuple[Location, ...]) -> list[dict[str, Any]]:
        """Fetch traffic records for all configured locations."""

        records: list[dict[str, Any]] = []
        for location in locations:
            try:
                records.append(self.fetch_location(location))
            except (requests.RequestException, ValueError) as exc:
                # Log the class only: str(exc) can contain the request URL with the key.
                logger.warning("Dropped %s: %s", location.name, type(exc).__name__)
        return records
