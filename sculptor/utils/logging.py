"""Logging helpers."""

from __future__ import annotations

import json
import logging
import time
from typing import Any


class JsonFormatter(logging.Formatter):
    """Format log records as one-line JSON for easier ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_fields"):
            payload.update(record.extra_fields)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


class ThroughputTracker:
    """Simple throughput tracker for long-running jobs."""

    def __init__(self) -> None:
        self.start_time = time.perf_counter()
        self.items_processed = 0

    def update(self, increment: int) -> float:
        self.items_processed += increment
        elapsed = max(time.perf_counter() - self.start_time, 1.0e-12)
        return self.items_processed / elapsed
