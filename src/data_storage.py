"""CSV-backed traffic data repository.

The repository interface keeps file persistence isolated so a PostgreSQL
implementation can replace it later without changing API or ML code.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_COLUMNS = [
    "timestamp",
    "location_name",
    "latitude",
    "longitude",
    "current_speed",
    "free_flow_speed",
    "confidence",
    "travel_time",
    "road_closure",
]

# A record is a repeat when its measured values equal the previous record for the same
# location. TomTom sends no observation time and our timestamp is the fetch time, so
# timestamp is not compared; a repeated cached speed is what would otherwise inflate the data.
VALUE_COLUMNS = [column for column in RAW_COLUMNS if column != "timestamp"]


class TrafficDataRepository:
    """Persist and read traffic observations from CSV files."""

    def __init__(self, raw_path: Path, processed_path: Path | None = None) -> None:
        # processed_path is accepted for existing callers and is not used by the raw store.
        self.raw_path = raw_path
        self.processed_path = processed_path

    def append_raw_records(self, records: Iterable[dict]) -> int:
        """Append new raw records, skipping repeats. Returns the number written."""

        new_df = pd.DataFrame(list(records), columns=RAW_COLUMNS)
        if new_df.empty:
            return 0

        existing_df = self._load_existing()
        combined = pd.concat(
            [frame for frame in (existing_df, new_df) if not frame.empty], ignore_index=True
        )
        is_new = combined.index >= len(existing_df)
        values = combined[VALUE_COLUMNS]
        previous = values.groupby(combined["location_name"]).shift()
        repeat = (values == previous).all(axis=1) & is_new
        self._write_atomic(combined[~repeat])

        skipped = int(repeat.sum())
        written = len(new_df) - skipped
        logger.info("Persisted %s new records; skipped %s repeats", written, skipped)
        return written

    def read_raw(self) -> pd.DataFrame:
        """Read all raw records. A missing or empty file reads as no rows."""

        if not self.raw_path.exists():
            return pd.DataFrame(columns=RAW_COLUMNS)
        try:
            return pd.read_csv(self.raw_path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=RAW_COLUMNS)

    def _load_existing(self) -> pd.DataFrame:
        """Return stored rows. An unreadable or foreign-schema file is moved aside."""

        try:
            existing = self.read_raw()
        except (pd.errors.ParserError, UnicodeDecodeError):
            existing = None
        if existing is None or list(existing.columns) != RAW_COLUMNS:
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
            backup = self.raw_path.with_name(f"{self.raw_path.name}.corrupt-{stamp}")
            os.replace(self.raw_path, backup)
            logger.error("Raw data file was unreadable or malformed; moved it to %s", backup)
            return pd.DataFrame(columns=RAW_COLUMNS)
        return existing

    def _write_atomic(self, dataframe: pd.DataFrame) -> None:
        """Write via a temp file in the same directory, so a crash never truncates the data."""

        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.raw_path.with_name(f"{self.raw_path.name}.tmp")
        try:
            dataframe.to_csv(tmp_path, index=False)
            os.replace(tmp_path, self.raw_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
