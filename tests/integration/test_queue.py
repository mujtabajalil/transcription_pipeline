"""JobQueue against a real Redis: delivery, visibility timeout, DLQ, failure mapping."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Iterator

import pytest
from redis import Redis
from redis.backoff import NoBackoff
from redis.retry import Retry

from transcription.errors import DependencyUnavailableError
from transcription.queue import JobQueue, QueueMessage

pytestmark = pytest.mark.integration

IDLE_MS = 50


@pytest.fixture
def queue(redis_client: Redis, redis_namespace: str) -> JobQueue:
    q = JobQueue(
        redis_client,
        stream=f"{redis_namespace}:jobs",
        group="workers",
        dlq_stream=f"{redis_namespace}:dlq",
    )
    q.ensure_group()
    return q


def _wait_idle() -> None:
    time.sleep(IDLE_MS * 2 / 1000)


def _receive(queue: JobQueue, consumer: str = "c1") -> QueueMessage:
    message = queue.receive(consumer, block_ms=100)
    assert message is not None
    return message


def _owners(redis_client: Redis, stream: str) -> dict[str, int]:
    summary = redis_client.xpending(stream, "workers")
    return {c["name"]: int(c["pending"]) for c in summary["consumers"]}


def test_ensure_group_is_idempotent(
    queue: JobQueue, redis_client: Redis, redis_namespace: str
) -> None:
    queue.ensure_group()
    groups = redis_client.xinfo_groups(f"{redis_namespace}:jobs")
    assert [g["name"] for g in groups] == ["workers"]


def test_enqueue_receive_ack(queue: JobQueue) -> None:
    job_id = uuid.uuid4()
    message_id = queue.enqueue(job_id)

    message = _receive(queue)

    assert message.id == message_id
    assert message.job_id == job_id
    assert message.raw == {"job_id": str(job_id)}
    assert queue.pending_count() == 1
    queue.ack(message.id)
    assert queue.pending_count() == 0
    assert queue.receive("c1", block_ms=10) is None


def test_receive_times_out_with_none(queue: JobQueue) -> None:
    started = time.monotonic()
    assert queue.receive("c1", block_ms=100) is None
    assert time.monotonic() - started >= 0.09


def test_concurrent_consumers_never_share_a_message(queue: JobQueue) -> None:
    job_ids = {queue.enqueue(uuid.uuid4()) for _ in range(40)}
    seen: dict[str, list[str]] = {"a": [], "b": [], "c": []}

    def consume(consumer: str) -> None:
        while (message := queue.receive(consumer, block_ms=50)) is not None:
            seen[consumer].append(message.id)

    threads = [threading.Thread(target=consume, args=(c,)) for c in seen]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    delivered = [mid for ids in seen.values() for mid in ids]
    assert sorted(delivered) == sorted(job_ids)  # every message exactly once


def test_reclaim_moves_idle_message_to_new_consumer(
    queue: JobQueue, redis_client: Redis, redis_namespace: str
) -> None:
    job_id = uuid.uuid4()
    queue.enqueue(job_id)
    original = _receive(queue, "dead-worker")

    assert queue.reclaim("rescuer", min_idle_ms=60_000) == []  # not idle long enough
    _wait_idle()
    [reclaimed] = queue.reclaim("rescuer", min_idle_ms=IDLE_MS)

    assert reclaimed.id == original.id
    assert reclaimed.job_id == job_id
    assert _owners(redis_client, f"{redis_namespace}:jobs") == {"rescuer": 1}


def test_touch_prevents_reclaim(queue: JobQueue) -> None:
    queue.enqueue(uuid.uuid4())
    message = _receive(queue, "busy-worker")
    _wait_idle()

    queue.touch("busy-worker", message.id)

    assert queue.reclaim("thief", min_idle_ms=IDLE_MS) == []


def test_touch_on_acked_message_is_a_noop(queue: JobQueue) -> None:
    queue.enqueue(uuid.uuid4())
    message = _receive(queue)
    queue.ack(message.id)

    queue.touch("c1", message.id)

    assert queue.pending_count() == 0


def test_reclaim_follows_cursor_up_to_count(queue: JobQueue) -> None:
    for _ in range(5):
        queue.enqueue(uuid.uuid4())
    for _ in range(5):
        _receive(queue, "dead-worker")
    _wait_idle()

    first = queue.reclaim("rescuer", min_idle_ms=IDLE_MS, count=2)
    rest = queue.reclaim("rescuer", min_idle_ms=IDLE_MS, count=10)

    assert len(first) == 2
    assert len(rest) == 3
    assert {m.id for m in first}.isdisjoint(m.id for m in rest)


def test_reclaim_acks_and_skips_trimmed_entries(
    queue: JobQueue, redis_client: Redis, redis_namespace: str
) -> None:
    stream = f"{redis_namespace}:jobs"
    for _ in range(3):
        queue.enqueue(uuid.uuid4())
    received = [_receive(queue, "dead-worker") for _ in range(3)]
    redis_client.xtrim(stream, maxlen=1, approximate=False)  # drops the first two entries
    _wait_idle()

    reclaimed = queue.reclaim("rescuer", min_idle_ms=IDLE_MS)

    assert [m.id for m in reclaimed] == [received[2].id]
    assert queue.pending_count() == 1
    assert _owners(redis_client, stream) == {"rescuer": 1}


def test_enqueue_trims_approximately_to_maxlen(redis_client: Redis, redis_namespace: str) -> None:
    stream = f"{redis_namespace}:small"
    q = JobQueue(redis_client, stream=stream, group="workers", dlq_stream="unused", maxlen=10)
    for _ in range(500):
        q.enqueue(uuid.uuid4())
    # Approximate trimming removes whole radix-tree nodes (100 entries by default).
    assert redis_client.xlen(stream) < 500


def test_dead_letter_writes_dlq_and_clears_pel(
    queue: JobQueue, redis_client: Redis, redis_namespace: str
) -> None:
    job_id = uuid.uuid4()
    queue.enqueue(job_id)
    message = _receive(queue)

    queue.dead_letter(message, reason="max_attempts_exceeded")

    assert queue.pending_count() == 0
    [(_, entry)] = redis_client.xrange(f"{redis_namespace}:dlq")
    assert entry == {
        "job_id": str(job_id),
        "message_id": message.id,
        "reason": "max_attempts_exceeded",
        "raw": f'{{"job_id": "{job_id}"}}',
    }


@pytest.mark.parametrize("fields", [{"job_id": "not-a-uuid"}, {"other": "x"}])
def test_malformed_payload_yields_none_job_id_and_can_be_dead_lettered(
    queue: JobQueue,
    redis_client: Redis,
    redis_namespace: str,
    fields: dict[str, str],
) -> None:
    redis_client.xadd(f"{redis_namespace}:jobs", fields)

    message = _receive(queue)
    assert message.job_id is None
    assert message.raw == fields

    queue.dead_letter(message, reason="malformed")
    [(_, entry)] = redis_client.xrange(f"{redis_namespace}:dlq")
    assert entry["job_id"] == ""
    assert queue.pending_count() == 0


def test_lag_and_pending_count(queue: JobQueue) -> None:
    assert queue.lag() == 0
    assert queue.pending_count() == 0
    for _ in range(3):
        queue.enqueue(uuid.uuid4())
    assert queue.lag() == 3

    message = _receive(queue)
    assert queue.lag() == 2
    assert queue.pending_count() == 1

    queue.ack(message.id)
    assert queue.pending_count() == 0


def test_lag_is_none_without_stream_or_group(redis_client: Redis, redis_namespace: str) -> None:
    q = JobQueue(
        redis_client, stream=f"{redis_namespace}:jobs", group="workers", dlq_stream="unused"
    )
    assert q.lag() is None  # no stream
    redis_client.xadd(f"{redis_namespace}:jobs", {"job_id": str(uuid.uuid4())})
    assert q.lag() is None  # stream, but not our group
    assert q.pending_count() == 0


def test_receive_recreates_group_after_redis_data_loss(
    queue: JobQueue, redis_client: Redis, redis_namespace: str
) -> None:
    redis_client.delete(f"{redis_namespace}:jobs")  # e.g. Redis restarted without AOF

    assert queue.receive("c1", block_ms=10) is None
    assert queue.reclaim("c1", min_idle_ms=0) == []
    job_id = uuid.uuid4()
    queue.enqueue(job_id)
    assert _receive(queue).job_id == job_id


def test_ping_ok(queue: JobQueue) -> None:
    queue.ping()


OPERATIONS: dict[str, Callable[[JobQueue], object]] = {
    "ensure_group": lambda q: q.ensure_group(),
    "enqueue": lambda q: q.enqueue(uuid.uuid4()),
    "receive": lambda q: q.receive("c1", block_ms=10),
    "reclaim": lambda q: q.reclaim("c1", min_idle_ms=0),
    "touch": lambda q: q.touch("c1", "1-0"),
    "ack": lambda q: q.ack("1-0"),
    "dead_letter": lambda q: q.dead_letter(QueueMessage("1-0", None, {}), reason="x"),
    "lag": lambda q: q.lag(),
    "pending_count": lambda q: q.pending_count(),
    "ping": lambda q: q.ping(),
}


@pytest.fixture
def unreachable_redis() -> Iterator[Redis]:
    """Nothing listens on port 1; no retries so each call fails fast."""
    client = Redis(port=1, socket_connect_timeout=0.2, retry=Retry(NoBackoff(), 0))
    yield client
    client.close()


@pytest.mark.parametrize("operation", OPERATIONS.values(), ids=OPERATIONS.keys())
def test_unreachable_redis_raises_dependency_unavailable(
    unreachable_redis: Redis, operation: Callable[[JobQueue], object]
) -> None:
    q = JobQueue(unreachable_redis, stream="s", group="g", dlq_stream="d")
    with pytest.raises(DependencyUnavailableError):
        operation(q)


def test_block_longer_than_socket_timeout_raises_dependency_unavailable(
    queue: JobQueue, redis_namespace: str
) -> None:
    """A read timeout maps like a dead Redis, so callers must keep block_ms below the
    client's socket_timeout (services uses 10 s)."""
    impatient = Redis(socket_timeout=0.05, retry=Retry(NoBackoff(), 0), decode_responses=True)
    q = JobQueue(impatient, stream=f"{redis_namespace}:jobs", group="workers", dlq_stream="unused")
    with pytest.raises(DependencyUnavailableError):
        q.receive("c1", block_ms=500)
    impatient.close()
