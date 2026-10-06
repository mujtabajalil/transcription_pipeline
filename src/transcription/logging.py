"""Structured JSON logging with request/job context carried in contextvars.

    with log_context(job_id=str(job.id)):
        log.info("chunk transcribed", extra={"chunk": 3, "engine": "faster_whisper"})

Anything in ``log_context`` or ``extra`` becomes a top-level JSON field.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, TextIO

_context: ContextVar[dict[str, Any] | None] = ContextVar("tx_log_context", default=None)

_STANDARD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys() | {"message", "asctime"}
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(_context.get() or {})
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(
    level: str = "INFO", *, json_output: bool = True, stream: TextIO | None = None
) -> None:
    """Route all logging to ``stream`` (default stdout; CLIs whose stdout carries their
    output pass stderr)."""
    handler = logging.StreamHandler(sys.stdout if stream is None else stream)
    handler.setFormatter(
        JsonFormatter() if json_output else logging.Formatter("%(levelname)s %(name)s %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("botocore", "boto3", "urllib3", "s3transfer", "httpx", "faster_whisper"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    token = _context.set({**(_context.get() or {}), **fields})
    try:
        yield
    finally:
        _context.reset(token)
