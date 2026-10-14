"""Tests for the CSV traffic repository."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.data_storage import RAW_COLUMNS, TrafficDataRepository


def record(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "timestamp": "2026-10-14T10:00:00+00:00",
        "location_name": "Delhi",
        "latitude": 28.6,
        "longitude": 77.2,
        "current_speed": 32.0,
        "free_flow_speed": 55.0,
        "confidence": 0.9,
        "travel_time": 120.0,
        "road_closure": False,
    }
    row.update(overrides)
    return row


@pytest.fixture
def raw_path(tmp_path: Path) -> Path:
    return tmp_path / "raw" / "traffic.csv"


def test_read_raw_missing_file_does_not_create_directory(raw_path: Path) -> None:
    frame = TrafficDataRepository(raw_path).read_raw()

    assert frame.empty
    assert list(frame.columns) == RAW_COLUMNS
    assert not raw_path.parent.exists()


def test_append_creates_directory_and_writes_rows(raw_path: Path) -> None:
    written = TrafficDataRepository(raw_path).append_raw_records(
        [record(), record(location_name="Mumbai")]
    )

    assert written == 2
    assert len(TrafficDataRepository(raw_path).read_raw()) == 2


def test_append_with_no_records_writes_nothing(raw_path: Path) -> None:
    assert TrafficDataRepository(raw_path).append_raw_records([]) == 0
    assert not raw_path.parent.exists()


def test_append_to_empty_existing_file(raw_path: Path) -> None:
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text("")

    written = TrafficDataRepository(raw_path).append_raw_records([record()])

    assert written == 1
    assert len(pd.read_csv(raw_path)) == 1


@pytest.mark.parametrize(
    "content",
    [
        b'timestamp,location_name\n"unterminated',  # pandas ParserError
        b"\xff\xfe\x00\x81bad",  # not UTF-8
        b"a,b\n1,2\n",  # wrong schema
    ],
)
def test_append_moves_corrupt_file_aside_and_keeps_collecting(
    raw_path: Path, content: bytes
) -> None:
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(content)

    written = TrafficDataRepository(raw_path).append_raw_records([record()])

    assert written == 1
    assert len(pd.read_csv(raw_path)) == 1
    backups = list(raw_path.parent.glob("traffic.csv.corrupt-*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == content


def test_repeated_values_are_skipped_even_with_new_timestamp(raw_path: Path) -> None:
    repo = TrafficDataRepository(raw_path)
    repo.append_raw_records([record()])

    written = repo.append_raw_records([record(timestamp="2026-10-14T10:10:00+00:00")])

    assert written == 0
    assert len(repo.read_raw()) == 1


def test_changed_values_are_kept(raw_path: Path) -> None:
    repo = TrafficDataRepository(raw_path)
    repo.append_raw_records([record()])

    written = repo.append_raw_records([record(current_speed=12.0)])

    assert written == 1
    assert len(repo.read_raw()) == 2


def test_repeat_within_one_batch_is_skipped(raw_path: Path) -> None:
    written = TrafficDataRepository(raw_path).append_raw_records(
        [record(), record(timestamp="2026-10-14T10:01:00+00:00")]
    )

    assert written == 1


def test_same_values_for_different_locations_are_both_kept(raw_path: Path) -> None:
    written = TrafficDataRepository(raw_path).append_raw_records(
        [record(), record(location_name="Mumbai")]
    )

    assert written == 2


def test_return_to_earlier_value_after_change_is_kept(raw_path: Path) -> None:
    repo = TrafficDataRepository(raw_path)
    repo.append_raw_records([record(current_speed=32.0)])
    repo.append_raw_records([record(current_speed=12.0)])

    written = repo.append_raw_records([record(current_speed=32.0)])

    assert written == 1
    assert len(repo.read_raw()) == 3


def test_failed_write_leaves_original_file_intact(
    raw_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = TrafficDataRepository(raw_path)
    repo.append_raw_records([record()])
    original = raw_path.read_text()

    def partial_write_then_fail(self: pd.DataFrame, path: Path, **kwargs: Any) -> None:
        Path(path).write_text("timestamp,loc")  # simulate a crash mid-write
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", partial_write_then_fail)
    with pytest.raises(OSError):
        repo.append_raw_records([record(current_speed=5.0)])

    assert raw_path.read_text() == original
    assert list(raw_path.parent.glob("*.tmp")) == []
