"""CSV-backed traffic data repository.

The repository interface keeps file persistence isolated so a PostgreSQL
implementation can replace it later without changing API or ML code.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
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


class TrafficDataRepository:
    """Persist and read traffic observations from CSV files."""

    def __init__(self, raw_path: Path, processed_path: Path | None = None) -> None:
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        if self.processed_path:
            self.processed_path.parent.mkdir(parents=True, exist_ok=True)

    def append_raw_records(self, records: Iterable[dict]) -> int:
        """Append new raw records while dropping duplicate observations."""

        new_df = pd.DataFrame(list(records), columns=RAW_COLUMNS)
        if new_df.empty:
            return 0

        existing_df = self.read_raw()
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        before = len(combined)
        combined = combined.drop_duplicates(
            subset=["timestamp", "location_name", "latitude", "longitude"], keep="last"
        )
        combined.to_csv(self.raw_path, index=False)
        written = len(combined) - len(existing_df)
        logger.info(
            "Persisted %s new records and removed %s duplicates",
            written,
            before - len(combined),
        )
        return max(written, 0)

    def read_raw(self) -> pd.DataFrame:
        """Read all raw records."""

        if not self.raw_path.exists():
            return pd.DataFrame(columns=RAW_COLUMNS)
        return pd.read_csv(self.raw_path)

    def save_processed(self, dataframe: pd.DataFrame) -> None:
        """Save processed feature data."""

        if self.processed_path is None:
            raise ValueError("Processed path is not configured")
        dataframe.to_csv(self.processed_path, index=False)

    def read_processed(self) -> pd.DataFrame:
        """Read processed feature data."""

        if self.processed_path is None or not self.processed_path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.processed_path)
