"""TomTom Traffic Flow API integration."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import requests

from src.config import Location

logger = logging.getLogger(__name__)


class TomTomTrafficFetcher:
    """Fetch traffic flow segment data from TomTom."""

    BASE_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"

    def __init__(self, api_key: str, timeout_seconds: int = 20) -> None:
        if not api_key:
            raise ValueError("TomTom API key is required")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def fetch_location(self, location: Location) -> dict[str, Any]:
        """Fetch and normalize one traffic record for a location."""

        params = {
            "point": f"{location.latitude},{location.longitude}",
            "unit": "KMPH",
            "key": self.api_key,
        }
        response = requests.get(self.BASE_URL, params=params, timeout=self.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        return self.parse_response(payload, location)

    @staticmethod
    def parse_response(payload: dict[str, Any], location: Location) -> dict[str, Any]:
        """Extract the fields required by the downstream pipeline."""

        data = payload.get("flowSegmentData", payload)
        coordinates = data.get("coordinates", {}).get("coordinate", [])
        coordinate = coordinates[0] if coordinates else {}

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "location_name": location.name,
            "latitude": float(coordinate.get("latitude", location.latitude)),
            "longitude": float(coordinate.get("longitude", location.longitude)),
            "current_speed": float(data.get("currentSpeed", 0.0)),
            "free_flow_speed": float(data.get("freeFlowSpeed", 0.0)),
            "confidence": float(data.get("confidence", 0.0)),
            "travel_time": float(data.get("currentTravelTime", 0.0)),
            "road_closure": bool(data.get("roadClosure", False)),
        }

    def fetch_many(self, locations: tuple[Location, ...]) -> list[dict[str, Any]]:
        """Fetch traffic records for all configured locations."""

        records: list[dict[str, Any]] = []
        for location in locations:
            try:
                records.append(self.fetch_location(location))
            except requests.RequestException:
                logger.exception("Failed to fetch traffic data for %s", location.name)
        return records
