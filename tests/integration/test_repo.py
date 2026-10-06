"""Repository against a real Postgres: lease fencing, idempotency races, outbox and
sweeper semantics. Time travel is done by rewriting timestamps in SQL, never by sleeping."""

from __future__ import annotations

import hashlib
import socket
import threading
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import sessionmaker

from transcription.db.repo import ClaimOutcome, JobRecord, Repository
from transcription.db.session import make_engine, make_session_factory
from transcription.domain import (
    AudioInfo,
    ChunkPlan,
    ChunkResult,
    JobStatus,
    PlannedChunk,
    Segment,
    Transcript,
    TranscriptionOptions,
    TranscriptStats,
)
from transcription.errors import DependencyUnavailableError, LeaseLostError

pytestmark = pytest.mark.integration

LEASE = 60
MAX = 3


@pytest.fixture(scope="module")
def engine(database_url: str) -> Iterator[Engine]:
    eng = make_engine(database_url, pool_size=10)
    yield eng
    eng.dispose()


@pytest.fixture
def repo(engine: Engine) -> Repository:
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE job_chunks, jobs, api_keys CASCADE"))
    return Repository(make_session_factory(engine))


@pytest.fixture
def key_id(repo: Repository) -> uuid.UUID:
    return repo.create_api_key("tenant")[0].id


def sql(engine: Engine, statement: str, **params: Any) -> Any:
    with engine.begin() as conn:
        result = conn.execute(text(statement), params)
        return result.all() if result.returns_rows else None


def new_job(
    repo: Repository,
    key_id: uuid.UUID,
    *,
    idempotency_key: str | None = None,
    fingerprint: str = "fp",
    webhook_url: str | None = None,
) -> JobRecord:
    job, created = repo.create_job(
        api_key_id=key_id,
        audio_key=f"audio/{key_id}/{uuid.uuid4()}",
        audio_bytes=123,
        audio_sha256="ab" * 32,
        options=TranscriptionOptions(language="en", prompt="Gettysburg"),
        webhook_url=webhook_url,
        idempotency_key=idempotency_key,
        request_fingerprint=fingerprint,
    )
    assert created
    return job


def claimed(repo: Repository, key_id: uuid.UUID, worker: str = "w1", **kw: Any) -> JobRecord:
    job = new_job(repo, key_id, **kw)
    result = repo.claim(job.id, worker_id=worker, lease_s=LEASE, max_attempts=MAX)
    assert result.outcome is ClaimOutcome.CLAIMED
    assert result.job is not None
    return result.job


def expire_lease(engine: Engine, job_id: uuid.UUID, seconds_ago: int = 1) -> None:
    sql(
        engine,
        "UPDATE jobs SET lease_expires_at = now() - make_interval(secs => :s) WHERE id = :id",
        s=seconds_ago,
        id=job_id,
    )


def race[T](n: int, fn: Callable[[], T]) -> list[T]:
    """Run ``fn`` on n threads released together by a barrier."""
    barrier = threading.Barrier(n)

    def run() -> T:
        barrier.wait()
        return fn()

    with ThreadPoolExecutor(n) as pool:
        return [f.result() for f in [pool.submit(run) for _ in range(n)]]


def chunk(index: int, text_: str = "four score") -> ChunkResult:
    return ChunkResult(
        index=index,
        start_s=index * 30.0,
        end_s=index * 30.0 + 29.5,
        engine="fake",
        language="en",
        segments=[Segment(start=index * 30.0, end=index * 30.0 + 2, text=text_, avg_logprob=-0.2)],
        dropped={"repetition": 1},
    )


PLAN = ChunkPlan(
    chunks=[
        PlannedChunk(index=0, start=0, end=480_000),
        PlannedChunk(index=1, channel=None, start=480_000, end=640_000, forced_cut=True),
    ],
    audio_samples=640_000,
    speech_samples=600_000,
    forced_cuts=1,
    channels=[None],
)

TRANSCRIPT = Transcript(
    language="en",
    language_probability=0.98,
    duration_s=40.0,
    text="four score",
    segments=[Segment(start=0.0, end=2.0, text="four score")],
    stats=TranscriptStats(
        audio_seconds=40, speech_seconds=37.5, transcribed_seconds=37.5, chunks=2, forced_cuts=1
    ),
    audio=AudioInfo(container="mp3", codec="mp3", channels=1, sample_rate=22050),
)


# --- API keys ----------------------------------------------------------------------------


def test_ping(repo: Repository) -> None:
    repo.ping()


def test_unreachable_database_is_dependency_unavailable() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    engine = make_engine(f"postgresql+psycopg://tx:tx@127.0.0.1:{port}/tx")
    try:
        with pytest.raises(DependencyUnavailableError) as raised:
            Repository(make_session_factory(engine)).ping()
    finally:
        engine.dispose()

    assert str(port) not in raised.value.message  # rendered to API clients


def test_lock_timeout_is_dependency_unavailable_without_leaking_sql(
    repo: Repository, engine: Engine, database_url: str, key_id: uuid.UUID
) -> None:
    job = claimed(repo, key_id, worker="w-secret-param")
    impatient = create_engine(database_url, connect_args={"options": "-c lock_timeout=50"})
    try:
        with engine.begin() as conn:
            conn.execute(text("SELECT 1 FROM jobs WHERE id = :id FOR UPDATE"), {"id": job.id})
            with pytest.raises(DependencyUnavailableError) as raised:
                Repository(make_session_factory(impatient)).heartbeat(
                    job.id, worker_id="w-secret-param", lease_s=LEASE
                )
    finally:
        impatient.dispose()

    assert raised.value.retryable
    assert "w-secret-param" not in raised.value.message
    assert "UPDATE" not in raised.value.message


def test_api_key_lifecycle(repo: Repository, engine: Engine) -> None:
    record, raw = repo.create_api_key("acme", rate_limit_per_minute=5)

    assert raw.startswith("tx_") and len(raw) == 3 + 43
    assert record.key_prefix == raw[:10]
    assert record.webhook_secret.startswith("whsec_")
    assert record.rate_limit_per_minute == 5
    [(stored_hash,)] = sql(engine, "SELECT key_hash FROM api_keys WHERE id = :id", id=record.id)
    assert stored_hash == hashlib.sha256(raw.encode()).hexdigest()

    assert repo.authenticate(raw) == record
    assert repo.authenticate(raw + "x") is None

    assert repo.revoke_api_key(record.id)
    [(revoked_at,)] = sql(engine, "SELECT revoked_at FROM api_keys WHERE id = :id", id=record.id)
    assert repo.authenticate(raw) is None
    assert not repo.revoke_api_key(record.id)  # no longer active
    assert sql(engine, "SELECT revoked_at FROM api_keys WHERE id = :id", id=record.id) == [
        (revoked_at,)
    ]
    assert not repo.revoke_api_key(uuid.uuid4())


def test_generated_keys_and_secrets_are_unique(repo: Repository) -> None:
    (a, raw_a), (b, raw_b) = repo.create_api_key("a"), repo.create_api_key("b")
    assert raw_a != raw_b
    assert a.webhook_secret != b.webhook_secret


def test_bootstrap_key_is_idempotent(repo: Repository) -> None:
    first, raw = repo.create_api_key("boot", raw_key="tx_bootstrap-dev-key")
    again, raw_again = repo.create_api_key("other name", raw_key="tx_bootstrap-dev-key")

    assert raw == raw_again == "tx_bootstrap-dev-key"
    assert again == first
    assert repo.authenticate("tx_bootstrap-dev-key") == first


# --- jobs: API side ----------------------------------------------------------------------


def test_create_job_round_trips_options(repo: Repository, key_id: uuid.UUID) -> None:
    job = new_job(repo, key_id, webhook_url="https://hooks.example/tx")

    assert job.status is JobStatus.QUEUED
    assert job.options == TranscriptionOptions(language="en", prompt="Gettysburg")
    assert job.attempts == 0 and job.chunks_done == 0
    assert job.webhook_status is None
    assert repo.get_job(job.id) == job


def test_idempotent_replay_returns_existing(repo: Repository, key_id: uuid.UUID) -> None:
    job = new_job(repo, key_id, idempotency_key="k1", fingerprint="fp-a")

    replay, created = repo.create_job(
        api_key_id=key_id,
        audio_key="audio/other",
        audio_bytes=None,
        audio_sha256=None,
        options=TranscriptionOptions(),
        webhook_url=None,
        idempotency_key="k1",
        request_fingerprint="fp-b",
    )

    assert not created
    assert replay == job  # caller compares fingerprints
    other_key = repo.create_api_key("other")[0].id
    assert new_job(repo, other_key, idempotency_key="k1").id != job.id


def test_concurrent_create_with_same_idempotency_key(repo: Repository, key_id: uuid.UUID) -> None:
    def create() -> tuple[JobRecord, bool]:
        return repo.create_job(
            api_key_id=key_id,
            audio_key="audio/x",
            audio_bytes=1,
            audio_sha256=None,
            options=TranscriptionOptions(),
            webhook_url=None,
            idempotency_key="same",
            request_fingerprint="fp",
        )

    results = race(8, create)

    assert sum(created for _, created in results) == 1
    assert len({job.id for job, _ in results}) == 1
    assert repo.count_active() == 1


def test_find_by_fingerprint_ignores_failed(
    repo: Repository, engine: Engine, key_id: uuid.UUID
) -> None:
    failed = new_job(repo, key_id, fingerprint="fp-x")
    repo.fail(failed.id, worker_id=None, code="undecodable_audio", http_status=422, message="x")
    assert repo.find_by_fingerprint(key_id, "fp-x") is None

    older = new_job(repo, key_id, fingerprint="fp-x")
    newer = new_job(repo, key_id, fingerprint="fp-x")
    sql(
        engine, "UPDATE jobs SET created_at = now() - interval '1 hour' WHERE id = :id", id=older.id
    )

    found = repo.find_by_fingerprint(key_id, "fp-x")
    assert found is not None and found.id == newer.id
    assert repo.find_by_fingerprint(repo.create_api_key("b")[0].id, "fp-x") is None


def test_get_job_checks_ownership(repo: Repository, key_id: uuid.UUID) -> None:
    job = new_job(repo, key_id)
    stranger = repo.create_api_key("stranger")[0].id

    assert repo.get_job(job.id, api_key_id=key_id) == job
    assert repo.get_job(job.id, api_key_id=stranger) is None
    assert repo.get_job(uuid.uuid4()) is None


def test_list_jobs_order_cursor_and_filter(
    repo: Repository, engine: Engine, key_id: uuid.UUID
) -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    jobs = [new_job(repo, key_id) for _ in range(5)]
    for i, job in enumerate(jobs):
        sql(
            engine,
            "UPDATE jobs SET created_at = :t WHERE id = :id",
            t=base + timedelta(minutes=i),
            id=job.id,
        )
    new_job(repo, repo.create_api_key("other")[0].id)  # never listed
    repo.fail(jobs[3].id, worker_id=None, code="c", http_status=422, message="m")
    newest_first = [j.id for j in reversed(jobs)]

    assert [j.id for j in repo.list_jobs(key_id)] == newest_first
    page1 = repo.list_jobs(key_id, limit=2)
    assert [j.id for j in page1] == newest_first[:2]
    page2 = repo.list_jobs(key_id, limit=2, before=page1[-1].created_at)
    assert [j.id for j in page2] == newest_first[2:4]
    assert [j.id for j in repo.list_jobs(key_id, status=JobStatus.FAILED)] == [jobs[3].id]
    assert len(repo.list_jobs(key_id, status=JobStatus.QUEUED)) == 4


def test_count_active(repo: Repository, key_id: uuid.UUID) -> None:
    other = repo.create_api_key("other")[0].id
    new_job(repo, key_id)
    claimed(repo, other)
    done = claimed(repo, key_id)
    repo.complete(done.id, worker_id="w1", transcript=TRANSCRIPT)

    assert repo.count_active() == 2


def test_delete_cascades_to_chunks(repo: Repository, engine: Engine, key_id: uuid.UUID) -> None:
    job = claimed(repo, key_id)
    repo.save_chunk(job.id, worker_id="w1", result=chunk(0))

    assert repo.delete_job(job.id, repo.create_api_key("x")[0].id) is None
    deleted = repo.delete_job(job.id, key_id)

    assert deleted is not None and deleted.audio_key == job.audio_key
    assert repo.get_job(job.id) is None
    assert sql(engine, "SELECT count(*) FROM job_chunks") == [(0,)]
    assert repo.delete_job(job.id, key_id) is None


def test_mark_audio_deleted_keeps_first_timestamp(repo: Repository, key_id: uuid.UUID) -> None:
    job = new_job(repo, key_id)
    repo.mark_audio_deleted(job.id)
    first = _get(repo, job.id).audio_deleted_at
    repo.mark_audio_deleted(job.id)

    assert first is not None
    assert _get(repo, job.id).audio_deleted_at == first


# --- claim -------------------------------------------------------------------------------


def test_claim_queued(repo: Repository, key_id: uuid.UUID) -> None:
    job = new_job(repo, key_id)
    result = repo.claim(job.id, worker_id="w1", lease_s=LEASE, max_attempts=MAX)

    assert result.outcome is ClaimOutcome.CLAIMED
    got = result.job
    assert got is not None
    assert got.status is JobStatus.PROCESSING and got.worker_id == "w1" and got.attempts == 1
    assert got.started_at is not None and got.lease_expires_at is not None
    lease = (got.lease_expires_at - got.updated_at).total_seconds()
    assert lease == pytest.approx(LEASE, abs=1)


def test_claim_with_live_lease_is_busy(repo: Repository, key_id: uuid.UUID) -> None:
    job = claimed(repo, key_id, "w1")
    result = repo.claim(job.id, worker_id="w2", lease_s=LEASE, max_attempts=MAX)

    assert result.outcome is ClaimOutcome.BUSY
    assert _get(repo, job.id).worker_id == "w1"


def test_claim_after_lease_expiry_takes_over(
    repo: Repository, engine: Engine, key_id: uuid.UUID
) -> None:
    job = claimed(repo, key_id, "w1")
    sql(engine, "UPDATE jobs SET error_message = 'engine down' WHERE id = :id", id=job.id)
    expire_lease(engine, job.id)

    result = repo.claim(job.id, worker_id="w2", lease_s=LEASE, max_attempts=MAX)

    assert result.outcome is ClaimOutcome.CLAIMED
    assert result.job is not None
    assert result.job.worker_id == "w2" and result.job.attempts == 2
    assert result.job.started_at == job.started_at  # first start is kept
    assert result.job.error_message is None


@pytest.mark.parametrize("terminal", ["succeeded", "failed"])
def test_claim_terminal(repo: Repository, key_id: uuid.UUID, terminal: str) -> None:
    job = claimed(repo, key_id)
    if terminal == "succeeded":
        repo.complete(job.id, worker_id="w1", transcript=TRANSCRIPT)
    else:
        repo.fail(job.id, worker_id="w1", code="c", http_status=422, message="m")

    result = repo.claim(job.id, worker_id="w2", lease_s=LEASE, max_attempts=MAX)

    assert result.outcome is ClaimOutcome.TERMINAL
    assert result.job is not None and result.job.status == terminal


@pytest.mark.parametrize("state", ["queued", "processing_expired"])
def test_claim_exhausted(repo: Repository, engine: Engine, key_id: uuid.UUID, state: str) -> None:
    job = claimed(repo, key_id)
    sql(engine, "UPDATE jobs SET attempts = :n WHERE id = :id", n=MAX, id=job.id)
    if state == "queued":
        repo.release(job.id, worker_id="w1", error_message="boom")
    else:
        expire_lease(engine, job.id)

    result = repo.claim(job.id, worker_id="w2", lease_s=LEASE, max_attempts=MAX)

    assert result.outcome is ClaimOutcome.EXHAUSTED
    assert _get(repo, job.id).attempts == MAX


def test_claim_exhausted_but_live_lease_is_busy(
    repo: Repository, engine: Engine, key_id: uuid.UUID
) -> None:
    job = claimed(repo, key_id)
    sql(engine, "UPDATE jobs SET attempts = :n WHERE id = :id", n=MAX, id=job.id)

    result = repo.claim(job.id, worker_id="w2", lease_s=LEASE, max_attempts=MAX)

    assert result.outcome is ClaimOutcome.BUSY


def test_claim_missing(repo: Repository) -> None:
    result = repo.claim(uuid.uuid4(), worker_id="w1", lease_s=LEASE, max_attempts=MAX)
    assert result.outcome is ClaimOutcome.MISSING and result.job is None


def test_concurrent_claims_have_one_winner(repo: Repository, key_id: uuid.UUID) -> None:
    job = new_job(repo, key_id)
    counter = iter(range(100))

    outcomes = race(
        8,
        lambda: (
            repo.claim(
                job.id, worker_id=f"w{next(counter)}", lease_s=LEASE, max_attempts=MAX
            ).outcome
        ),
    )

    assert sorted(outcomes) == sorted([ClaimOutcome.CLAIMED] + [ClaimOutcome.BUSY] * 7)
    assert _get(repo, job.id).attempts == 1


# --- lease fencing -----------------------------------------------------------------------


def test_heartbeat_extends_lease(repo: Repository, key_id: uuid.UUID) -> None:
    job = claimed(repo, key_id)
    assert job.lease_expires_at is not None

    assert repo.heartbeat(job.id, worker_id="w1", lease_s=10 * LEASE)

    extended = _get(repo, job.id).lease_expires_at
    assert extended is not None and extended - job.lease_expires_at > timedelta(seconds=LEASE)
    assert not repo.heartbeat(job.id, worker_id="intruder", lease_s=LEASE)


FENCED_WRITES: dict[str, Callable[[Repository, uuid.UUID], object]] = {
    "record_audio": lambda r, j: r.record_audio(
        j, worker_id="w1", info=TRANSCRIPT.audio, audio_sha256=None, audio_bytes=None
    ),
    "save_plan": lambda r, j: r.save_plan(j, worker_id="w1", plan=PLAN),
    "save_chunk": lambda r, j: r.save_chunk(j, worker_id="w1", result=chunk(0)),
    "save_language": lambda r, j: r.save_language(
        j, worker_id="w1", language="en", probability=0.9
    ),
    "complete": lambda r, j: r.complete(j, worker_id="w1", transcript=TRANSCRIPT),
    "fail": lambda r, j: r.fail(j, worker_id="w1", code="c", http_status=422, message="m"),
}


@pytest.mark.parametrize("write", FENCED_WRITES.values(), ids=FENCED_WRITES.keys())
def test_fenced_writes_after_takeover_raise_lease_lost(
    repo: Repository,
    engine: Engine,
    key_id: uuid.UUID,
    write: Callable[[Repository, uuid.UUID], object],
) -> None:
    job = claimed(repo, key_id, "w1")
    expire_lease(engine, job.id)
    assert repo.claim(job.id, worker_id="w2", lease_s=LEASE, max_attempts=MAX).outcome is (
        ClaimOutcome.CLAIMED
    )
    before = _get(repo, job.id)

    with pytest.raises(LeaseLostError):
        write(repo, job.id)

    assert _get(repo, job.id) == before
    assert not repo.heartbeat(job.id, worker_id="w1", lease_s=LEASE)
    assert not repo.release(job.id, worker_id="w1", error_message="x")


def test_save_chunk_upserts_and_counts(repo: Repository, key_id: uuid.UUID) -> None:
    job = claimed(repo, key_id)

    assert repo.save_chunk(job.id, worker_id="w1", result=chunk(0, "first try")) == 1
    assert repo.save_chunk(job.id, worker_id="w1", result=chunk(0, "second try")) == 1
    assert repo.save_chunk(job.id, worker_id="w1", result=chunk(3)) == 2

    chunks = repo.load_chunks(job.id)
    assert chunks == {0: chunk(0, "second try"), 3: chunk(3)}
    assert _get(repo, job.id).chunks_done == 2


def test_fenced_save_chunk_rolls_back_the_chunk(
    repo: Repository, engine: Engine, key_id: uuid.UUID
) -> None:
    job = claimed(repo, key_id, "w1")
    repo.save_chunk(job.id, worker_id="w1", result=chunk(0))
    expire_lease(engine, job.id)
    repo.claim(job.id, worker_id="w2", lease_s=LEASE, max_attempts=MAX)

    with pytest.raises(LeaseLostError):
        repo.save_chunk(job.id, worker_id="w1", result=chunk(1))

    assert set(repo.load_chunks(job.id)) == {0}
    assert repo.save_chunk(job.id, worker_id="w2", result=chunk(1)) == 2


def test_save_chunk_for_deleted_job_is_lease_lost(repo: Repository, key_id: uuid.UUID) -> None:
    job = claimed(repo, key_id)
    repo.delete_job(job.id, key_id)

    with pytest.raises(LeaseLostError):
        repo.save_chunk(job.id, worker_id="w1", result=chunk(0))


def test_checkpoints_round_trip_as_domain_types(repo: Repository, key_id: uuid.UUID) -> None:
    job = claimed(repo, key_id)
    info = AudioInfo(container="wav", codec="pcm_mulaw", channels=1, sample_rate=8000)

    repo.record_audio(job.id, worker_id="w1", info=info, audio_sha256=None, audio_bytes=None)
    repo.save_plan(job.id, worker_id="w1", plan=PLAN)
    repo.save_language(job.id, worker_id="w1", language="en", probability=0.97)

    got = _get(repo, job.id)
    assert got.audio_info == info
    assert got.audio_sha256 == job.audio_sha256 and got.audio_bytes == job.audio_bytes  # kept
    assert got.plan == PLAN and got.plan.chunks[1].forced_cut
    assert got.chunks_total == 2 and got.duration_s == pytest.approx(40.0)
    assert (got.language, got.language_probability) == ("en", 0.97)

    repo.record_audio(job.id, worker_id="w1", info=info, audio_sha256="cd" * 32, audio_bytes=9)
    got = _get(repo, job.id)
    assert (got.audio_sha256, got.audio_bytes) == ("cd" * 32, 9)


# --- terminal transitions and the outbox -------------------------------------------------


def test_complete_with_webhook_queues_outbox(repo: Repository, key_id: uuid.UUID) -> None:
    job = claimed(repo, key_id, webhook_url="https://hooks.example/tx")

    done = repo.complete(job.id, worker_id="w1", transcript=TRANSCRIPT)

    assert done.status is JobStatus.SUCCEEDED
    assert done.result == TRANSCRIPT
    assert (done.language, done.duration_s) == ("en", 40.0)
    assert done.finished_at is not None and done.lease_expires_at is None
    assert done.webhook_status == "pending"
    [task] = repo.claim_webhooks(limit=10, lease_s=LEASE)
    assert task.job.id == job.id


def test_complete_without_webhook_has_no_outbox(repo: Repository, key_id: uuid.UUID) -> None:
    job = claimed(repo, key_id)

    assert repo.complete(job.id, worker_id="w1", transcript=TRANSCRIPT).webhook_status is None
    assert repo.claim_webhooks(limit=10, lease_s=LEASE) == []


def test_fail_records_error_and_outbox(repo: Repository, key_id: uuid.UUID) -> None:
    hooked = claimed(repo, key_id, webhook_url="https://hooks.example/tx")
    plain = claimed(repo, key_id)

    failed = repo.fail(
        hooked.id,
        worker_id="w1",
        code="undecodable_audio",
        http_status=422,
        message="audio could not be decoded",
    )
    other = repo.fail(plain.id, worker_id="w1", code="c", http_status=422, message="m")

    assert failed is not None and other is not None
    assert failed.status is JobStatus.FAILED
    assert (failed.error_code, failed.error_status) == ("undecodable_audio", 422)
    assert failed.finished_at is not None and failed.lease_expires_at is None
    assert (failed.webhook_status, other.webhook_status) == ("pending", None)


def test_unfenced_fail_never_overwrites_terminal(repo: Repository, key_id: uuid.UUID) -> None:
    job = claimed(repo, key_id)
    done = repo.complete(job.id, worker_id="w1", transcript=TRANSCRIPT)

    assert (
        repo.fail(
            job.id, worker_id=None, code="max_attempts_exceeded", http_status=500, message="dlq"
        )
        is None
    )
    assert _get(repo, job.id) == done
    assert repo.fail(uuid.uuid4(), worker_id=None, code="c", http_status=500, message="m") is None


def test_unfenced_fail_of_exhausted_job(
    repo: Repository, engine: Engine, key_id: uuid.UUID
) -> None:
    job = claimed(repo, key_id)
    expire_lease(engine, job.id)

    failed = repo.fail(
        job.id, worker_id=None, code="max_attempts_exceeded", http_status=500, message="dlq"
    )

    assert failed is not None and failed.status is JobStatus.FAILED


def test_release_requeues_and_keeps_error(repo: Repository, key_id: uuid.UUID) -> None:
    job = claimed(repo, key_id)

    assert repo.release(job.id, worker_id="w1", error_message="engine unavailable")

    got = _get(repo, job.id)
    assert got.status is JobStatus.QUEUED and got.attempts == 1
    assert got.worker_id is None and got.lease_expires_at is None
    assert got.error_message == "engine unavailable"
    assert not repo.release(job.id, worker_id="w1", error_message="again")
    retried = repo.claim(job.id, worker_id="w2", lease_s=LEASE, max_attempts=MAX).job
    assert retried is not None and retried.attempts == 2 and retried.error_message is None


# --- sweeper -----------------------------------------------------------------------------


def test_requeue_orphans(repo: Repository, engine: Engine, key_id: uuid.UUID) -> None:
    stale_queued = new_job(repo, key_id)
    fresh_queued = new_job(repo, key_id)
    dead_worker = claimed(repo, key_id)
    just_expired = claimed(repo, key_id)
    finished = claimed(repo, key_id)
    repo.complete(finished.id, worker_id="w1", transcript=TRANSCRIPT)
    sql(engine, "UPDATE jobs SET updated_at = now() - interval '10 minutes'")
    sql(engine, "UPDATE jobs SET updated_at = now() WHERE id = :id", id=fresh_queued.id)
    expire_lease(engine, dead_worker.id, seconds_ago=300)
    expire_lease(engine, just_expired.id, seconds_ago=5)

    swept = repo.requeue_orphans(queued_older_than_s=60, lease_grace_s=60)

    assert set(swept) == {stale_queued.id, dead_worker.id}
    assert repo.requeue_orphans(queued_older_than_s=60, lease_grace_s=60) == []
    assert _get(repo, dead_worker.id).status is JobStatus.PROCESSING  # sweeping only re-wakes


def test_requeue_orphans_respects_limit(
    repo: Repository, engine: Engine, key_id: uuid.UUID
) -> None:
    for _ in range(5):
        new_job(repo, key_id)
    sql(engine, "UPDATE jobs SET updated_at = now() - interval '1 hour'")

    first = repo.requeue_orphans(queued_older_than_s=60, lease_grace_s=60, limit=3)
    second = repo.requeue_orphans(queued_older_than_s=60, lease_grace_s=60, limit=3)

    assert len(first) == 3 and len(second) == 2 and not set(first) & set(second)


def test_concurrent_sweepers_get_disjoint_sets(
    repo: Repository, engine: Engine, key_id: uuid.UUID
) -> None:
    jobs = {new_job(repo, key_id).id for _ in range(40)}
    sql(engine, "UPDATE jobs SET updated_at = now() - interval '1 hour'")

    a, b = race(2, lambda: repo.requeue_orphans(queued_older_than_s=60, lease_grace_s=60, limit=25))

    assert not set(a) & set(b)
    assert set(a) | set(b) == jobs


def test_sweeper_skips_rows_locked_by_another_transaction(
    repo: Repository, engine: Engine, key_id: uuid.UUID
) -> None:
    locked, free = new_job(repo, key_id), new_job(repo, key_id)
    sql(engine, "UPDATE jobs SET updated_at = now() - interval '1 hour'")

    with engine.begin() as conn:
        conn.execute(text("SELECT 1 FROM jobs WHERE id = :id FOR UPDATE"), {"id": locked.id})
        swept = repo.requeue_orphans(queued_older_than_s=60, lease_grace_s=60)

    assert swept == [free.id]


def test_records_survive_an_expire_on_commit_session_factory(
    repo: Repository, engine: Engine, key_id: uuid.UUID
) -> None:
    """Callers may hand in a default sessionmaker, which expires rows on commit:
    every record must be built before the session closes."""
    strict = Repository(sessionmaker(engine))
    job = claimed(strict, key_id, webhook_url="https://hooks.example/tx")

    busy = strict.claim(job.id, worker_id="w2", lease_s=LEASE, max_attempts=MAX)
    assert busy.outcome is ClaimOutcome.BUSY and busy.job is not None
    strict.save_language(job.id, worker_id="w1", language="en", probability=0.9)
    failed = strict.fail(job.id, worker_id="w1", code="c", http_status=422, message="m")
    assert failed is not None and failed.language == "en"
    [task] = strict.claim_webhooks(limit=10, lease_s=LEASE)
    assert task.job.id == job.id and task.attempt == 1


# --- webhook outbox ----------------------------------------------------------------------


def webhook_state(engine: Engine, job_id: uuid.UUID) -> tuple[Any, ...]:
    [row] = sql(
        engine,
        "SELECT webhook_status, webhook_attempts, webhook_last_error, "
        "webhook_next_at > now() FROM jobs WHERE id = :id",
        id=job_id,
    )
    return tuple(row)


def test_webhook_outbox_state_machine(repo: Repository, engine: Engine) -> None:
    key, _ = repo.create_api_key("hooks")
    job = claimed(repo, key.id, webhook_url="https://hooks.example/tx")
    repo.complete(job.id, worker_id="w1", transcript=TRANSCRIPT)

    [task] = repo.claim_webhooks(limit=10, lease_s=LEASE)
    assert (task.job.id, task.secret, task.attempt) == (job.id, key.webhook_secret, 1)
    assert task.job.result == TRANSCRIPT
    assert webhook_state(engine, job.id) == ("sending", 1, None, True)
    assert repo.claim_webhooks(limit=10, lease_s=LEASE) == []  # in flight

    repo.finish_webhook(job.id, delivered=False, error="HTTP 503", retry_in_s=3600)
    assert webhook_state(engine, job.id) == ("pending", 1, "HTTP 503", True)
    assert repo.claim_webhooks(limit=10, lease_s=LEASE) == []  # not due yet

    sql(engine, "UPDATE jobs SET webhook_next_at = now() - interval '1 second'")
    [retry] = repo.claim_webhooks(limit=10, lease_s=LEASE)
    assert retry.attempt == 2

    repo.finish_webhook(job.id, delivered=True, error=None, retry_in_s=None)
    assert webhook_state(engine, job.id) == ("delivered", 2, None, None)
    repo.finish_webhook(job.id, delivered=False, error="late duplicate", retry_in_s=0)
    assert webhook_state(engine, job.id)[0] == "delivered"
    assert repo.claim_webhooks(limit=10, lease_s=LEASE) == []


def test_webhook_lease_expiry_reclaims_then_gives_up(repo: Repository, engine: Engine) -> None:
    key, _ = repo.create_api_key("hooks")
    job = claimed(repo, key.id, webhook_url="https://hooks.example/tx")
    repo.fail(job.id, worker_id="w1", code="c", http_status=422, message="m")
    repo.claim_webhooks(limit=10, lease_s=LEASE)

    sql(engine, "UPDATE jobs SET webhook_next_at = now() - interval '1 second'")  # dispatcher died
    [again] = repo.claim_webhooks(limit=10, lease_s=LEASE)
    assert again.attempt == 2 and again.job.status is JobStatus.FAILED

    repo.finish_webhook(job.id, delivered=False, error="timeout", retry_in_s=None)
    assert webhook_state(engine, job.id) == ("failed", 2, "timeout", None)
    sql(engine, "UPDATE jobs SET webhook_next_at = now() - interval '1 hour'")
    assert repo.claim_webhooks(limit=10, lease_s=LEASE) == []


def test_claim_webhooks_takes_most_overdue_first(repo: Repository, engine: Engine) -> None:
    key, _ = repo.create_api_key("hooks")
    jobs = []
    for _ in range(3):
        job = claimed(repo, key.id, webhook_url="https://hooks.example/tx")
        repo.complete(job.id, worker_id="w1", transcript=TRANSCRIPT)
        jobs.append(job.id)
    for minutes, job_id in zip((1, 3, 2), jobs, strict=True):
        sql(
            engine,
            "UPDATE jobs SET webhook_next_at = now() - make_interval(mins => :m) WHERE id = :id",
            m=minutes,
            id=job_id,
        )

    first = repo.claim_webhooks(limit=2, lease_s=LEASE)

    assert {t.job.id for t in first} == {jobs[1], jobs[2]}
    assert [t.job.id for t in repo.claim_webhooks(limit=2, lease_s=LEASE)] == [jobs[0]]


def test_concurrent_dispatchers_claim_disjoint_webhooks(repo: Repository) -> None:
    key, _ = repo.create_api_key("hooks")
    jobs = set()
    for _ in range(20):
        job = claimed(repo, key.id, webhook_url="https://hooks.example/tx")
        repo.complete(job.id, worker_id="w1", transcript=TRANSCRIPT)
        jobs.add(job.id)

    a, b = race(2, lambda: repo.claim_webhooks(limit=15, lease_s=LEASE))

    assert not {t.job.id for t in a} & {t.job.id for t in b}
    assert {t.job.id for t in a + b} == jobs
    assert {t.attempt for t in a + b} == {1}


# --- helpers that need the repo ----------------------------------------------------------


def _get(repo: Repository, job_id: uuid.UUID) -> JobRecord:
    job = repo.get_job(job_id)
    assert job is not None
    return job
