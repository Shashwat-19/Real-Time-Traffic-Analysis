"""Scheduler for periodic TomTom traffic data collection."""

from __future__ import annotations

import logging
import time

import schedule

from src.config import get_settings
from src.data_fetcher import TomTomTrafficFetcher
from src.data_storage import TrafficDataRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def collect_once() -> int:
    """Fetch current traffic for configured locations and append to storage."""

    settings = get_settings()
    if not settings.tomtom_api_key:
        raise RuntimeError("TOMTOM_API_KEY is not set")

    fetcher = TomTomTrafficFetcher(settings.tomtom_api_key)
    repository = TrafficDataRepository(settings.raw_data_path)
    records = fetcher.fetch_many(settings.locations)
    written = repository.append_raw_records(records)
    logger.info("Collected %s records; wrote %s new rows", len(records), written)
    return written


def collect_safely() -> None:
    """Run one collection cycle; a failure is logged so the scheduler keeps running."""

    try:
        collect_once()
    except Exception:
        logger.exception("Collection cycle failed")


def run_scheduler() -> None:
    """Run collection every configured interval."""

    settings = get_settings()
    schedule.every(settings.fetch_interval_minutes).minutes.do(collect_safely)
    collect_safely()
    logger.info("Scheduler started with %s minute interval", settings.fetch_interval_minutes)
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    run_scheduler()
