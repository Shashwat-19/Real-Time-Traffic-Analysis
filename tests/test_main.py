"""Tests for the collector entry point and scheduler loop."""

import datetime
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import schedule

import main


class StopLoop(Exception):
    """Raised from the patched sleep to end the endless scheduler loop."""


class FakeFetcher:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def fetch_many(self, locations: Any) -> list[dict[str, Any]]:
        return [
            {
                "timestamp": "2026-10-14T10:00:00+00:00",
                "location_name": "Pune",
                "latitude": 18.5,
                "longitude": 73.8,
                "current_speed": 30.0,
                "free_flow_speed": 50.0,
                "confidence": 0.9,
                "travel_time": 100.0,
                "road_closure": False,
            }
        ]


@pytest.fixture
def collector_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    raw_path = tmp_path / "raw" / "traffic.csv"
    monkeypatch.setenv("TOMTOM_API_KEY", "test-key")
    monkeypatch.setenv("LOCATIONS", "Pune:18.5:73.8")
    monkeypatch.setenv("RAW_DATA_PATH", str(raw_path))
    monkeypatch.setenv("FETCH_INTERVAL_MINUTES", "1")
    monkeypatch.setattr(main, "TomTomTrafficFetcher", FakeFetcher)
    return raw_path


def test_collect_once_writes_fetched_records(collector_env: Path) -> None:
    assert main.collect_once() == 1
    assert len(pd.read_csv(collector_env)) == 1


def test_collect_once_requires_api_key(
    collector_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TOMTOM_API_KEY")

    with pytest.raises(RuntimeError, match="TOMTOM_API_KEY"):
        main.collect_once()


def test_collect_safely_logs_failure_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def failing_collect() -> int:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(main, "collect_once", failing_collect)

    with caplog.at_level(logging.ERROR):
        main.collect_safely()

    assert "Collection cycle failed" in caplog.text
    assert "disk on fire" in caplog.text


def test_run_scheduler_keeps_running_after_failed_cycles(
    collector_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[int] = []

    def failing_collect() -> int:
        calls.append(1)
        raise RuntimeError("bad response")

    scheduler = schedule.Scheduler()
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 3:
            raise StopLoop
        # Make the scheduled job due so the next run_pending() executes it.
        for job in scheduler.jobs:
            job.next_run = datetime.datetime.now()

    monkeypatch.setattr(main, "schedule", scheduler)
    monkeypatch.setattr(main, "collect_once", failing_collect)
    monkeypatch.setattr(main.time, "sleep", fake_sleep)

    with caplog.at_level(logging.ERROR), pytest.raises(StopLoop):
        main.run_scheduler()

    # One startup call plus two scheduled calls, all failing, and the loop kept going.
    assert len(calls) == 3
    assert len(sleeps) == 3
    assert caplog.text.count("Collection cycle failed") == 3
