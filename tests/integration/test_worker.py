"""Worker against real Postgres + Redis and moto S3: the DESIGN.md outcome table, crash
resume, lease loss, graceful shutdown, heartbeat, sweeper and webhook outbox.

Leases are expired by rewriting timestamps in SQL. The only sleeps wait out the 1-2 s
visibility timeout, which Redis measures on its own clock.
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from prometheus_client import REGISTRY
from redis import Redis
from redis.backoff import NoBackoff
from redis.retry import Retry
from sqlalchemy import Engine, text

from tests.fakes import FakeEngine
from transcription.config import Settings
from transcription.db.repo import ApiKeyRecord, ClaimOutcome, JobRecord, Repository
from transcription.db.session import make_engine, make_session_factory
from transcription.domain import JobStatus, TranscriptionOptions
from transcription.queue import JobQueue
from transcription.ratelimit import RateLimiter
from transcription.services import Services
from transcription.storage import ObjectStore
from transcription.webhooks import SIGNATURE_HEADER, WebhookSender, verify
from transcription.worker import Worker
from transcription.worker.background import SWEEPER_LOCK_KEY, dispatch_webhooks, sweep

pytestmark = pytest.mark.integration

VISIBILITY_S = 1
MAX_ATTEMPTS = 2


class SimulatedKill(BaseException):
    """Stands in for SIGKILL: nothing in the worker may catch it."""


@pytest.fixture(scope="module")
def db(database_url: str) -> Iterator[Engine]:
    engine = make_engine(database_url, pool_size=10)
    yield engine
    engine.dispose()


@pytest.fixture
def make_services(
    db: Engine,
    database_url: str,
    redis_client: Redis,
    redis_namespace: str,
    s3_client: Any,
    settings: Settings,
) -> Callable[..., Services]:
    sql(db, "TRUNCATE job_chunks, jobs, api_keys CASCADE")
    store = ObjectStore(
        bucket=settings.s3_bucket, region="us-east-1", client=s3_client, presign_client=s3_client
    )
    store.ensure_bucket(retention_days=1)
    repo = Repository(make_session_factory(db))

    def make(**overrides: Any) -> Services:
        config = settings.model_copy(
            update={
                "database_url": database_url,
                "queue_stream": f"{redis_namespace}:jobs",
                "dlq_stream": f"{redis_namespace}:dlq",
                "visibility_timeout_s": VISIBILITY_S,
                "heartbeat_interval_s": 1,
                "max_job_attempts": MAX_ATTEMPTS,
                "chunk_max_retries": 0,
                **overrides,
            }
        )
        queue = JobQueue(
            redis_client,
            stream=config.queue_stream,
            group=config.queue_group,
            dlq_stream=config.dlq_stream,
        )
        queue.ensure_group()
        return Services(
            settings=config,
            engine=db,
            repo=repo,
            redis=redis_client,
            queue=queue,
            store=store,
            rate_limiter=RateLimiter(redis_client),
        )

    return make


@pytest.fixture
def services(make_services: Callable[..., Services]) -> Services:
    return make_services()


@pytest.fixture
def key(services: Services) -> ApiKeyRecord:
    return services.repo.create_api_key("tenant")[0]


@pytest.fixture
def gaps(samples_dir: Path) -> Path:
    """Speech / 20 s silence / speech / 3 s silence / speech: three chunks."""
    return samples_dir / "gaps.mp3"


def sql(engine: Engine, statement: str, **params: Any) -> Any:
    with engine.begin() as conn:
        result = conn.execute(text(statement), params)
        return result.all() if result.returns_rows else None


def submit(
    services: Services,
    key: ApiKeyRecord,
    audio: Path,
    *,
    webhook_url: str | None = None,
    enqueue: bool = True,
) -> JobRecord:
    audio_key = services.store.key_for(key.id, uuid.uuid4())
    services.store.upload_file(audio_key, audio)
    job, _ = services.repo.create_job(
        api_key_id=key.id,
        audio_key=audio_key,
        audio_bytes=None,
        audio_sha256=None,
        options=TranscriptionOptions(),
        webhook_url=webhook_url,
        idempotency_key=None,
        request_fingerprint=uuid.uuid4().hex,
    )
    if enqueue:
        services.queue.enqueue(job.id)
    return job


def reload(services: Services, job: JobRecord) -> JobRecord:
    current = services.repo.get_job(job.id)
    assert current is not None
    return current


def texts(job: JobRecord) -> list[str]:
    assert job.result is not None
    return [segment.text for segment in job.result.segments]


def expire_lease(db: Engine, job: JobRecord) -> None:
    sql(
        db,
        "UPDATE jobs SET lease_expires_at = now() - interval '1 second' WHERE id = :id",
        id=job.id,
    )


def wait_out_visibility(seconds: float = VISIBILITY_S) -> None:
    time.sleep(seconds + 0.2)


def wait_for(condition: Callable[[], bool], timeout_s: float = 10) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.05)


def metric(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def heartbeats_alive() -> bool:
    return any(thread.name.startswith("heartbeat-") for thread in threading.enumerate())


# --- outcome table ---------------------------------------------------------------------


def test_happy_path_transcribes_and_acks(services: Services, key: ApiKeyRecord, gaps: Path) -> None:
    job = submit(services, key, gaps)
    worker = Worker(services, FakeEngine(), worker_id="w1")
    succeeded = metric("tx_jobs_finished_total", status="succeeded", error_code="")

    assert worker.run_once(block_ms=100)

    done = reload(services, job)
    assert done.status is JobStatus.SUCCEEDED
    assert texts(done) == ["chunk 0", "chunk 1", "chunk 2"]
    assert done.chunks_done == done.chunks_total == 3
    assert done.attempts == 1
    assert done.worker_id == "w1"
    assert done.audio_sha256 == hashlib.sha256(gaps.read_bytes()).hexdigest()
    assert done.audio_bytes == gaps.stat().st_size
    assert done.audio_info is not None and done.audio_info.container == "mp3"
    assert services.queue.pending_count() == 0
    assert metric("tx_jobs_finished_total", status="succeeded", error_code="") == succeeded + 1
    assert not heartbeats_alive()
    assert not worker.run_once(block_ms=50)


def test_unsupported_media_fails_without_retry(
    services: Services, key: ApiKeyRecord, samples_dir: Path
) -> None:
    job = submit(services, key, samples_dir / "text.wav")
    engine = FakeEngine()
    worker = Worker(services, engine, worker_id="w1")

    assert worker.run_once(block_ms=100)

    failed = reload(services, job)
    assert failed.status is JobStatus.FAILED
    assert (failed.error_code, failed.error_status) == ("unsupported_media_type", 415)
    assert failed.attempts == 1
    assert engine.calls == []
    assert services.queue.pending_count() == 0
    wait_out_visibility()
    assert not worker.run_once(block_ms=50)


def test_engine_errors_are_redelivered_until_dead_lettered(
    services: Services, key: ApiKeyRecord, gaps: Path, redis_client: Redis
) -> None:
    job = submit(services, key, gaps)
    worker = Worker(services, FakeEngine(fail_times=10**6), worker_id="w1")

    assert worker.run_once(block_ms=100)
    released = reload(services, job)
    assert (released.status, released.attempts, released.worker_id) == (JobStatus.QUEUED, 1, None)
    assert released.error_message == "boom"
    assert services.queue.pending_count() == 1  # not acked
    assert not worker.run_once(block_ms=50)  # not idle long enough to be redelivered yet

    wait_out_visibility()
    assert worker.run_once(block_ms=50)
    assert reload(services, job).attempts == MAX_ATTEMPTS
    assert reload(services, job).status is JobStatus.QUEUED

    wait_out_visibility()
    dead_lettered = metric("tx_jobs_dead_lettered_total")
    assert worker.run_once(block_ms=50)

    failed = reload(services, job)
    assert failed.status is JobStatus.FAILED
    assert (failed.error_code, failed.error_status) == ("max_attempts_exceeded", 500)
    assert failed.error_message == "boom"
    assert failed.attempts == MAX_ATTEMPTS
    assert services.queue.pending_count() == 0
    [(_, entry)] = redis_client.xrange(services.settings.dlq_stream)
    assert (entry["job_id"], entry["reason"]) == (str(job.id), "max_attempts_exceeded")
    assert metric("tx_jobs_dead_lettered_total") == dead_lettered + 1


def test_malformed_message_is_dead_lettered(services: Services, redis_client: Redis) -> None:
    redis_client.xadd(services.settings.queue_stream, {"job_id": "not-a-uuid"})

    assert Worker(services, FakeEngine(), worker_id="w1").run_once(block_ms=100)

    assert services.queue.pending_count() == 0
    [(_, entry)] = redis_client.xrange(services.settings.dlq_stream)
    assert entry["reason"] == "malformed message"


def test_duplicate_messages_process_the_job_once(
    services: Services, key: ApiKeyRecord, gaps: Path
) -> None:
    job = submit(services, key, gaps)
    services.queue.enqueue(job.id)
    working, proceed = threading.Event(), threading.Event()

    def hold_first_chunk(n: int) -> None:
        if n == 0:
            working.set()
            proceed.wait(10)

    first = Worker(services, FakeEngine(on_call=hold_first_chunk), worker_id="w1")
    runner = threading.Thread(target=first.run_once, kwargs={"block_ms": 100})
    runner.start()
    assert working.wait(10)

    second_engine = FakeEngine()
    second = Worker(services, second_engine, worker_id="w2")
    assert second.run_once(block_ms=100)  # BUSY: the duplicate is acked
    assert services.queue.pending_count() == 1
    proceed.set()
    runner.join(10)

    assert reload(services, job).status is JobStatus.SUCCEEDED
    services.queue.enqueue(job.id)
    assert second.run_once(block_ms=100)  # TERMINAL: acked too
    done = reload(services, job)
    assert (done.status, done.attempts) == (JobStatus.SUCCEEDED, 1)
    assert second_engine.calls == []
    assert services.queue.pending_count() == 0


# --- crash, lease loss, shutdown, heartbeat ----------------------------------------------


def test_killed_worker_is_resumed_from_checkpoint(
    services: Services,
    key: ApiKeyRecord,
    gaps: Path,
    db: Engine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    job = submit(services, key, gaps)

    def kill_on_second_chunk(n: int) -> None:
        if n == 1:
            raise SimulatedKill

    with pytest.raises(SimulatedKill):
        Worker(
            services, FakeEngine(prefix="a", on_call=kill_on_second_chunk), worker_id="w1"
        ).run_once(block_ms=100)
    crashed = reload(services, job)
    assert (crashed.status, crashed.worker_id, crashed.chunks_done) == (
        JobStatus.PROCESSING,
        "w1",
        1,
    )
    assert list(scratch.iterdir()) == []  # temp dir removed even on BaseException
    assert not heartbeats_alive()

    expire_lease(db, job)
    wait_out_visibility()
    resumed = metric("tx_jobs_resumed_total")
    survivor = FakeEngine(prefix="b")
    assert Worker(services, survivor, worker_id="w2").run_once(block_ms=50)

    done = reload(services, job)
    assert (done.status, done.attempts, done.worker_id) == (JobStatus.SUCCEEDED, 2, "w2")
    assert texts(done) == ["a 0", "b 0", "b 1"]
    assert len(survivor.calls) == 2
    assert metric("tx_jobs_resumed_total") == resumed + 1
    assert services.queue.pending_count() == 0


def test_worker_that_lost_its_lease_stops_without_ack_or_writes(
    services: Services, key: ApiKeyRecord, gaps: Path, db: Engine
) -> None:
    job = submit(services, key, gaps)

    def steal_job(n: int) -> None:
        if n == 1:
            expire_lease(db, job)
            claim = services.repo.claim(job.id, worker_id="thief", lease_s=60, max_attempts=5)
            assert claim.outcome is ClaimOutcome.CLAIMED

    engine = FakeEngine(on_call=steal_job)
    assert Worker(services, engine, worker_id="w1").run_once(block_ms=100)

    stolen = reload(services, job)
    assert (stolen.status, stolen.worker_id, stolen.attempts) == (JobStatus.PROCESSING, "thief", 2)
    assert stolen.chunks_done == 1  # the chunk finished after the theft was not saved
    assert stolen.result is None
    assert len(engine.calls) == 2  # stopped at the first fenced write
    assert services.queue.pending_count() == 1  # message left for the new owner


def test_shutdown_hands_job_back_for_immediate_resume(
    services: Services, key: ApiKeyRecord, gaps: Path
) -> None:
    job = submit(services, key, gaps)

    def shut_down_during_first_chunk(n: int) -> None:
        if n == 0:
            worker.request_shutdown()

    worker = Worker(
        services, FakeEngine(prefix="a", on_call=shut_down_during_first_chunk), worker_id="w1"
    )
    assert worker.run_once(block_ms=100)

    handed_back = reload(services, job)
    assert (handed_back.status, handed_back.worker_id) == (JobStatus.QUEUED, None)
    assert (handed_back.attempts, handed_back.chunks_done) == (1, 1)
    assert services.queue.pending_count() == 0
    assert services.queue.lag() == 1  # a fresh message, no visibility timeout to wait out
    assert not worker.run_once(block_ms=50)  # takes nothing once shutting down

    successor = FakeEngine(prefix="b")
    assert Worker(services, successor, worker_id="w2").run_once(block_ms=100)
    done = reload(services, job)
    assert done.status is JobStatus.SUCCEEDED
    assert texts(done) == ["a 0", "b 0", "b 1"]
    assert len(successor.calls) == 2


def test_heartbeat_keeps_a_slow_job_from_being_reclaimed(
    make_services: Callable[..., Services], gaps: Path, db: Engine
) -> None:
    services = make_services(visibility_timeout_s=2, heartbeat_interval_s=1)
    key = services.repo.create_api_key("tenant")[0]
    job = submit(services, key, gaps)
    services.queue.enqueue(job.id)  # duplicate, for the second worker to receive
    in_slow_chunk = threading.Event()

    def slow_first_chunk(n: int) -> None:
        if n == 0:
            in_slow_chunk.set()
            time.sleep(4)

    first_engine = FakeEngine(on_call=slow_first_chunk)
    first = Worker(services, first_engine, worker_id="w1")
    runner = threading.Thread(target=first.run_once, kwargs={"block_ms": 100})
    runner.start()
    assert in_slow_chunk.wait(10)
    wait_out_visibility(2.5)  # well past the lease taken at claim time

    [(lease_live,)] = sql(db, "SELECT lease_expires_at > now() FROM jobs WHERE id = :id", id=job.id)
    assert lease_live
    second_engine = FakeEngine()
    second = Worker(services, second_engine, worker_id="w2")
    assert second.run_once(block_ms=100)  # the duplicate: job BUSY, acked
    assert not second.run_once(block_ms=100)  # the original keeps getting touched
    runner.join(15)

    done = reload(services, job)
    assert (done.status, done.attempts, done.worker_id) == (JobStatus.SUCCEEDED, 1, "w1")
    assert len(first_engine.calls) == 3
    assert second_engine.calls == []
    assert services.queue.pending_count() == 0


# --- loop ------------------------------------------------------------------------------


def test_run_forever_processes_until_shutdown(
    services: Services, key: ApiKeyRecord, gaps: Path
) -> None:
    job = submit(services, key, gaps)
    worker = Worker(services, FakeEngine(), worker_id="w1")
    runner = threading.Thread(target=worker.run_forever, args=(threading.Event(),))
    runner.start()

    wait_for(lambda: reload(services, job).status is JobStatus.SUCCEEDED)
    worker.request_shutdown()
    runner.join(10)

    assert not runner.is_alive()


def test_run_forever_backs_off_while_redis_is_down(
    services: Services, caplog: pytest.LogCaptureFixture
) -> None:
    dead = Redis(port=1, retry=Retry(NoBackoff(), 0), decode_responses=True)
    queue = JobQueue(dead, stream="s", group="g", dlq_stream="d")
    worker = Worker(replace(services, redis=dead, queue=queue), FakeEngine(), worker_id="w1")
    stop = threading.Event()
    runner = threading.Thread(target=worker.run_forever, args=(stop,))

    def backoffs() -> list[float]:
        return [
            record.__dict__["retry_in_s"]
            for record in caplog.records
            if record.getMessage() == "dependency unavailable, backing off"
        ]

    with caplog.at_level(logging.WARNING, logger="transcription.worker.processor"):
        runner.start()
        wait_for(lambda: len(backoffs()) == 2)
        stop.set()  # interrupts the 2 s backoff
        runner.join(5)

    assert not runner.is_alive()
    assert backoffs() == [1.0, 2.0]


# --- background threads ----------------------------------------------------------------


@dataclass
class Receiver:
    url: str
    statuses: list[int] = field(default_factory=list)
    """Scripted response codes, consumed in order; 200 once exhausted."""
    requests: list[tuple[str, bytes]] = field(default_factory=list)
    """(signature header, body) per request."""


@pytest.fixture
def receiver() -> Iterator[Receiver]:
    """Local webhook endpoint (allowed via webhook_allow_private_targets)."""
    received = Receiver(url="")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.requests.append((self.headers[SIGNATURE_HEADER], body))
            self.send_response(received.statuses.pop(0) if received.statuses else 200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    received.url = f"http://127.0.0.1:{server.server_port}/hook"
    yield received
    server.shutdown()
    server.server_close()


@pytest.fixture
def sender() -> Iterator[WebhookSender]:
    webhook_sender = WebhookSender(timeout_s=5, allow_private=True)
    yield webhook_sender
    webhook_sender.close()


def test_webhook_is_retried_after_a_500_then_delivered(
    services: Services,
    key: ApiKeyRecord,
    gaps: Path,
    db: Engine,
    receiver: Receiver,
    sender: WebhookSender,
) -> None:
    receiver.statuses.append(500)
    job = submit(services, key, gaps, webhook_url=receiver.url)
    assert Worker(services, FakeEngine(), worker_id="w1").run_once(block_ms=100)

    assert dispatch_webhooks(services.repo, sender, max_attempts=5) == 1
    [(status, attempts, error, delay)] = sql(
        db,
        "SELECT webhook_status, webhook_attempts, webhook_last_error, webhook_next_at - now()"
        " FROM jobs WHERE id = :id",
        id=job.id,
    )
    assert (status, attempts, error) == ("pending", 1, "HTTP 500")
    assert 4 < delay.total_seconds() <= 5
    assert dispatch_webhooks(services.repo, sender, max_attempts=5) == 0  # not due yet

    sql(db, "UPDATE jobs SET webhook_next_at = now() WHERE id = :id", id=job.id)
    assert dispatch_webhooks(services.repo, sender, max_attempts=5) == 1

    delivered = reload(services, job)
    assert (delivered.webhook_status, delivered.webhook_attempts) == ("delivered", 2)
    assert len(receiver.requests) == 2
    assert all(verify(key.webhook_secret, body, signature) for signature, body in receiver.requests)
    first, retry = (json.loads(body) for _, body in receiver.requests)
    assert first == retry  # same event id, so the receiver can drop duplicates
    assert first["type"] == "transcription.succeeded"
    assert first["data"]["id"] == str(job.id)


def test_webhook_is_abandoned_after_max_attempts(
    services: Services, key: ApiKeyRecord, gaps: Path, receiver: Receiver, sender: WebhookSender
) -> None:
    receiver.statuses.append(503)
    job = submit(services, key, gaps, webhook_url=receiver.url)
    assert Worker(services, FakeEngine(), worker_id="w1").run_once(block_ms=100)

    assert dispatch_webhooks(services.repo, sender, max_attempts=1) == 1

    assert reload(services, job).webhook_status == "failed"


def test_background_threads_deliver_webhooks_until_stopped(
    services: Services,
    key: ApiKeyRecord,
    gaps: Path,
    receiver: Receiver,
    sender: WebhookSender,
    redis_client: Redis,
) -> None:
    job = submit(services, key, gaps, webhook_url=receiver.url)
    worker = Worker(services, FakeEngine(), worker_id="w1", webhook_sender=sender)
    assert worker.run_once(block_ms=100)
    stop = threading.Event()

    threads = worker.start_background(stop)
    try:
        wait_for(lambda: reload(services, job).webhook_status == "delivered")
    finally:
        stop.set()
        for thread in threads:
            thread.join(5)
        redis_client.delete(SWEEPER_LOCK_KEY)

    assert {thread.name for thread in threads} == {"sweeper", "webhook-dispatcher"}
    assert not any(thread.is_alive() for thread in threads)
    assert len(receiver.requests) == 1


def test_sweeper_requeues_a_job_whose_message_was_lost(
    services: Services, key: ApiKeyRecord, gaps: Path, db: Engine, redis_namespace: str
) -> None:
    lost = submit(services, key, gaps, enqueue=False)
    sql(db, "UPDATE jobs SET updated_at = now() - interval '1 hour' WHERE id = :id", id=lost.id)
    services.queue.enqueue(uuid.uuid4())  # undelivered backlog: the job may just be waiting
    worker = Worker(services, FakeEngine(), worker_id="w1")

    assert sweep(services, lock_key=f"{redis_namespace}:lock:a") == []
    assert sweep(services, lock_key=f"{redis_namespace}:lock:a") is None  # swept recently

    assert worker.run_once(block_ms=100)  # drains the backlog (job MISSING, acked)
    assert sweep(services, lock_key=f"{redis_namespace}:lock:b") == [lost.id]
    assert worker.run_once(block_ms=100)
    assert reload(services, lost).status is JobStatus.SUCCEEDED
