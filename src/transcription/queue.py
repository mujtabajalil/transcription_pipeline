"""Job wake-up queue on Redis Streams with a consumer group.

Why streams: a delivered-but-unacked message stays in the group's pending list (PEL).
If a worker dies, its messages go idle and another worker takes them with XAUTOCLAIM
— the "visibility timeout" behaviour of SQS. Live workers keep long jobs from being
stolen by periodically re-claiming their own message (``touch``), which resets idle.

Messages carry only a job id. Postgres holds the truth (status, lease, attempts), so
duplicates are harmless: the DB claim decides who works.

Every method maps Redis connection failures and timeouts to
``DependencyUnavailableError`` so the worker loop can back off instead of crashing.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError, ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError
from redis.typing import EncodableT, FieldT

from transcription.errors import DependencyUnavailableError

log = logging.getLogger(__name__)

_CURSOR_DONE = "0-0"


@dataclass(frozen=True)
class QueueMessage:
    id: str
    job_id: uuid.UUID | None
    """None if the payload was malformed — dead-letter it."""
    raw: dict[str, str]


@contextmanager
def _redis_unavailable(op: str) -> Iterator[None]:
    try:
        yield
    except (RedisConnectionError, RedisTimeoutError) as exc:
        log.warning("redis unavailable", extra={"op": op, "error": str(exc)})
        raise DependencyUnavailableError("job queue unavailable") from exc


def _is_missing_group(exc: ResponseError) -> bool:
    """NOGROUP: the stream or group vanished (e.g. Redis restarted without persistence)."""
    return str(exc).startswith("NOGROUP")


def _to_message(message_id: str, fields: dict[str, str]) -> QueueMessage:
    try:
        job_id: uuid.UUID | None = uuid.UUID(fields["job_id"])
    except (KeyError, ValueError):
        job_id = None
    return QueueMessage(id=message_id, job_id=job_id, raw=dict(fields))


class JobQueue:
    """Requires a client built with ``decode_responses=True`` (see ``services``)."""

    def __init__(
        self,
        redis: Redis,
        *,
        stream: str,
        group: str,
        dlq_stream: str,
        maxlen: int = 100_000,
    ) -> None:
        self._redis = redis
        self._stream = stream
        self._group = group
        self._dlq_stream = dlq_stream
        self._maxlen = maxlen

    def ensure_group(self) -> None:
        """XGROUP CREATE <stream> <group> 0 MKSTREAM; BUSYGROUP is fine.

        Starting at id 0 means messages XADDed before the group existed (e.g. by the API
        right after a Redis restart) are still delivered."""
        with _redis_unavailable("ensure_group"):
            try:
                self._redis.xgroup_create(self._stream, self._group, id="0", mkstream=True)
            except ResponseError as exc:
                if not str(exc).startswith("BUSYGROUP"):
                    raise

    def enqueue(self, job_id: uuid.UUID) -> str:
        """XADD with approximate MAXLEN trimming. Returns the message id."""
        with _redis_unavailable("enqueue"):
            message_id = self._redis.xadd(
                self._stream, {"job_id": str(job_id)}, maxlen=self._maxlen, approximate=True
            )
        return str(message_id)

    def receive(self, consumer: str, *, block_ms: int) -> QueueMessage | None:
        """XREADGROUP > COUNT 1 BLOCK block_ms. None on timeout.

        A missing group is recreated (and None returned) so a Redis restart without
        persistence doesn't wedge the worker; the sweeper re-enqueues lost jobs."""
        with _redis_unavailable("receive"):
            try:
                response = self._redis.xreadgroup(
                    self._group, consumer, {self._stream: ">"}, count=1, block=block_ms
                )
            except ResponseError as exc:
                if not _is_missing_group(exc):
                    raise
                log.warning("consumer group missing, recreating", extra={"stream": self._stream})
                self.ensure_group()
                return None
        if not response:
            return None
        [[_stream, [(message_id, fields)]]] = cast("list[list[Any]]", response)
        return _to_message(message_id, fields)

    def reclaim(self, consumer: str, *, min_idle_ms: int, count: int = 10) -> list[QueueMessage]:
        """XAUTOCLAIM messages idle ≥ min_idle_ms to ``consumer``. Entries trimmed from
        the stream (returned as deleted ids) are acked and skipped."""
        messages: list[QueueMessage] = []
        cursor = _CURSOR_DONE
        with _redis_unavailable("reclaim"):
            while len(messages) < count:
                try:
                    cursor, claimed, deleted = self._redis.xautoclaim(
                        self._stream,
                        self._group,
                        consumer,
                        min_idle_ms,
                        start_id=cursor,
                        count=count - len(messages),
                    )
                except ResponseError as exc:
                    if not _is_missing_group(exc):
                        raise
                    self.ensure_group()
                    break
                if deleted:
                    self._redis.xack(self._stream, self._group, *deleted)
                    log.warning(
                        "skipped pending messages trimmed from the stream",
                        extra={"stream": self._stream, "message_ids": deleted},
                    )
                messages.extend(_to_message(mid, fields) for mid, fields in claimed)
                if cursor == _CURSOR_DONE:
                    break
        return messages

    def touch(self, consumer: str, message_id: str) -> None:
        """XCLAIM <id> to self with min-idle 0 + JUSTID: resets idle so the message is
        not reclaimed while we're still working on it."""
        with _redis_unavailable("touch"):
            self._redis.xclaim(self._stream, self._group, consumer, 0, [message_id], justid=True)

    def ack(self, message_id: str) -> None:
        with _redis_unavailable("ack"):
            self._redis.xack(self._stream, self._group, message_id)

    def dead_letter(self, message: QueueMessage, *, reason: str) -> None:
        """XADD to the DLQ stream (original id, job id, reason) then XACK. Pipelined.

        MULTI makes the pair atomic: a message is never both acked and missing from the
        DLQ. The DLQ shares ``maxlen`` so a poison-message flood can't exhaust memory."""
        job_id = str(message.job_id) if message.job_id else ""
        entry: dict[FieldT, EncodableT] = {
            "job_id": job_id,
            "message_id": message.id,
            "reason": reason,
            "raw": json.dumps(message.raw, sort_keys=True),
        }
        with _redis_unavailable("dead_letter"):
            pipe = self._redis.pipeline(transaction=True)
            pipe.xadd(self._dlq_stream, entry, maxlen=self._maxlen, approximate=True)
            pipe.xack(self._stream, self._group, message.id)
            pipe.execute()
        log.warning(
            "message dead-lettered",
            extra={"message_id": message.id, "job_id": job_id, "reason": reason},
        )

    def lag(self) -> int | None:
        """Entries not yet delivered to the group (XINFO GROUPS 'lag'); None if unknown."""
        with _redis_unavailable("lag"):
            try:
                groups = self._redis.xinfo_groups(self._stream)
            except ResponseError:  # stream does not exist
                return None
        for group in groups:
            if group["name"] == self._group:
                lag = group.get("lag")
                return None if lag is None else int(lag)
        return None

    def pending_count(self) -> int:
        """Delivered but unacked (XPENDING summary)."""
        with _redis_unavailable("pending_count"):
            try:
                summary = self._redis.xpending(self._stream, self._group)
            except ResponseError as exc:
                if not _is_missing_group(exc):
                    raise
                return 0
        return int(summary["pending"])

    def ping(self) -> None:
        """Raises DependencyUnavailableError."""
        try:
            self._redis.ping()
        except RedisError as exc:
            raise DependencyUnavailableError("job queue unavailable") from exc
