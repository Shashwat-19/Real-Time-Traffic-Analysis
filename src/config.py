"""Application configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Location:
    """A configured traffic observation location."""

    name: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    tomtom_api_key: str | None
    locations: tuple[Location, ...]
    raw_data_path: Path
    processed_data_path: Path
    model_path: Path
    metrics_path: Path
    fetch_interval_minutes: int


def resolve_path(value: str) -> Path:
    """Resolve project-relative paths while preserving absolute paths."""

    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_locations(value: str | None) -> tuple[Location, ...]:
    """Parse LOCATIONS as name:latitude:longitude comma-separated values."""

    if not value:
        return (
            Location("New Delhi", 28.6139, 77.2090),
            Location("Mumbai", 19.0760, 72.8777),
            Location("Bengaluru", 12.9716, 77.5946),
        )

    locations: list[Location] = []
    for item in value.split(","):
        parts = item.strip().split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid location format: {item!r}")
        name, latitude, longitude = parts
        locations.append(Location(name.strip(), float(latitude), float(longitude)))
    return tuple(locations)


def get_settings() -> Settings:
    """Create application settings from the current environment."""

    return Settings(
        tomtom_api_key=os.getenv("TOMTOM_API_KEY"),
        locations=parse_locations(os.getenv("LOCATIONS")),
        raw_data_path=resolve_path(os.getenv("RAW_DATA_PATH", "data/raw/traffic_data.csv")),
        processed_data_path=resolve_path(
            os.getenv("PROCESSED_DATA_PATH", "data/processed/traffic_features.csv")
        ),
        model_path=resolve_path(os.getenv("MODEL_PATH", "models/traffic_model.pkl")),
        metrics_path=resolve_path(os.getenv("METRICS_PATH", "outputs/metrics/model_metrics.json")),
        fetch_interval_minutes=int(os.getenv("FETCH_INTERVAL_MINUTES", "10")),
    )
