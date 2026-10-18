"""CSV-backed traffic data repository.

The repository interface keeps file persistence isolated so a PostgreSQL
implementation can replace it later without changing API or ML code.
"""

from __future__ import annotations

import io
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

# Every reading is stored. is_repeat flags a reading whose values equal the previous stored
# reading for the same location; training and evaluation exclude those rows.
STORED_COLUMNS = [*RAW_COLUMNS, "is_repeat"]

_TAIL_BLOCK_BYTES = 64 * 1024  # bytes read per step when scanning backwards for a tail


class TrafficDataRepository:
    """Persist and read traffic observations from CSV files."""

    def __init__(self, raw_path: Path, processed_path: Path | None = None) -> None:
        # processed_path is accepted for existing callers and is not used by the raw store.
        self.raw_path = raw_path
        self.processed_path = processed_path

    def append_raw_records(self, records: Iterable[dict]) -> int:
        """Append new raw records, flagging repeats with is_repeat. Returns the number written."""

        new_df = pd.DataFrame(list(records), columns=RAW_COLUMNS)
        if new_df.empty:
            return 0

        existing_df = self._load_existing()
        new_df = new_df.assign(is_repeat=False)
        combined = pd.concat(
            [frame for frame in (existing_df, new_df) if not frame.empty], ignore_index=True
        )
        is_new = combined.index >= len(existing_df)
        values = combined[VALUE_COLUMNS]
        previous = values.groupby(combined["location_name"]).shift()
        # NaN never equals NaN, so a reading with missing values is not a repeat.
        repeat = (values == previous).all(axis=1)
        combined.loc[is_new, "is_repeat"] = repeat[is_new]
        self._write_atomic(combined)

        repeats = int(combined.loc[is_new, "is_repeat"].sum())
        logger.info("Persisted %s new records; %s flagged as repeats", len(new_df), repeats)
        return len(new_df)

    def read_raw(self) -> pd.DataFrame:
        """Read all raw records. A missing or empty file reads as no rows."""

        if not self.raw_path.exists():
            return pd.DataFrame(columns=STORED_COLUMNS)
        try:
            frame = pd.read_csv(self.raw_path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=STORED_COLUMNS)
        if "is_repeat" not in frame.columns:  # files written before the flag existed
            frame["is_repeat"] = False
        return frame

    def read_raw_tail(self, data_rows: int) -> tuple[pd.DataFrame, bool]:
        """Read the header and the last `data_rows` rows, without reading the whole file.

        The flag is True when those rows are the whole file (nothing was left out).
        """

        if not self.raw_path.exists():
            return pd.DataFrame(columns=STORED_COLUMNS), True
        with self.raw_path.open("rb") as handle:
            header = handle.readline()
            if not header:
                return pd.DataFrame(columns=STORED_COLUMNS), True
            start = handle.tell()
            pos = os.fstat(handle.fileno()).st_size
            chunk = b""
            # Step backwards until there is one more newline than rows needed, so the
            # partial line at the cut is always among the extra ones.
            while pos > start and chunk.count(b"\n") <= data_rows:
                step = min(_TAIL_BLOCK_BYTES, pos - start)
                pos -= step
                handle.seek(pos)
                chunk = handle.read(step) + chunk
            whole = pos <= start
            lines = chunk.split(b"\n")
            if not whole:
                handle.seek(pos - 1)
                starts_on_line = handle.read(1) == b"\n"
                if not starts_on_line:  # the first piece is the end of a row cut in two
                    lines = lines[1:]
        rows = [line for line in lines if line.strip()]
        whole = whole and len(rows) <= data_rows
        body = rows[-data_rows:]
        frame = pd.read_csv(io.BytesIO(header + b"".join(line + b"\n" for line in body)))
        if "is_repeat" not in frame.columns:
            frame["is_repeat"] = False
        return frame, whole

    def _load_existing(self) -> pd.DataFrame:
        """Return stored rows. An unreadable or foreign-schema file is moved aside."""

        try:
            existing = self.read_raw()
        except (pd.errors.ParserError, UnicodeDecodeError):
            existing = None
        if existing is None or list(existing.columns) != STORED_COLUMNS:
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
            backup = self.raw_path.with_name(f"{self.raw_path.name}.corrupt-{stamp}")
            os.replace(self.raw_path, backup)
            logger.error("Raw data file was unreadable or malformed; moved it to %s", backup)
            return pd.DataFrame(columns=STORED_COLUMNS)
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
