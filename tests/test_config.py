"""Tests for environment-based settings."""

from pathlib import Path

import pytest

from src.config import PROJECT_ROOT, get_settings, parse_locations

SETTING_VARS = [
    "TOMTOM_API_KEY",
    "LOCATIONS",
    "FETCH_INTERVAL_MINUTES",
    "RAW_DATA_PATH",
    "PROCESSED_DATA_PATH",
    "MODEL_PATH",
    "METRICS_PATH",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in SETTING_VARS:
        monkeypatch.delenv(name, raising=False)


def test_get_settings_defaults() -> None:
    settings = get_settings()

    assert settings.tomtom_api_key is None
    assert [location.name for location in settings.locations] == [
        "New Delhi",
        "Mumbai",
        "Bengaluru",
    ]
    assert settings.fetch_interval_minutes == 10
    assert settings.raw_data_path == PROJECT_ROOT / "data/raw/traffic_data.csv"
    assert settings.processed_data_path == PROJECT_ROOT / "data/processed/traffic_features.csv"


def test_get_settings_reads_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TOMTOM_API_KEY", "key")
    monkeypatch.setenv("LOCATIONS", "Pune:18.5:73.8")
    monkeypatch.setenv("FETCH_INTERVAL_MINUTES", "5")
    monkeypatch.setenv("RAW_DATA_PATH", str(tmp_path / "raw.csv"))

    settings = get_settings()

    assert settings.tomtom_api_key == "key"
    assert settings.locations[0].name == "Pune"
    assert settings.fetch_interval_minutes == 5
    assert settings.raw_data_path == tmp_path / "raw.csv"


@pytest.mark.parametrize("value", ["0", "-1"])
def test_get_settings_rejects_interval_below_one(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("FETCH_INTERVAL_MINUTES", value)

    with pytest.raises(ValueError, match="at least 1"):
        get_settings()


def test_get_settings_rejects_non_integer_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FETCH_INTERVAL_MINUTES", "ten")

    with pytest.raises(ValueError):
        get_settings()


def test_parse_locations_empty_uses_defaults() -> None:
    assert len(parse_locations(None)) == 3
    assert len(parse_locations("")) == 3


@pytest.mark.parametrize("value", ["Pune:18.5", "Pune:18.5:73.8:1", "Pune:north:73.8"])
def test_parse_locations_rejects_bad_input(value: str) -> None:
    with pytest.raises(ValueError):
        parse_locations(value)


def test_parse_locations_strips_whitespace() -> None:
    (location,) = parse_locations(" Pune : 18.5 : 73.8 ")

    assert location.name == "Pune"
    assert location.latitude == 18.5
    assert location.longitude == 73.8
