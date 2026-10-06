"""HTTP API against real Postgres and Redis plus moto S3, through FastAPI's TestClient.

Each test gets its own Redis namespace (stream, rate-limit buckets) and empty tables;
time-dependent job states are produced with the repository's worker-side methods."""

from __future__ import annotations

import tempfile
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from redis import Redis
from redis.backoff import NoBackoff
from redis.retry import Retry
from sqlalchemy import Engine, text

from transcription.api.app import create_app
from transcription.config import Settings
from transcription.db.repo import JobRecord, Repository
from transcription.db.session import make_engine, make_session_factory
from transcription.domain import (
    AudioInfo,
    ChunkPlan,
    ChunkResult,
    PlannedChunk,
    Segment,
    Transcript,
    TranscriptionOptions,
    TranscriptStats,
)
from transcription.queue import JobQueue
from transcription.ratelimit import RateLimiter
from transcription.services import Services
from transcription.storage import ObjectStore

pytestmark = pytest.mark.integration

WORKER = "worker-1"
PROBLEM = "application/problem+json"
Auth = dict[str, str]


# --- fixtures ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def engine(database_url: str) -> Iterator[Engine]:
    eng = make_engine(database_url, pool_size=5)
    yield eng
    eng.dispose()


@pytest.fixture
def repo(engine: Engine) -> Repository:
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE job_chunks, jobs, api_keys CASCADE"))
    return Repository(make_session_factory(engine))


@pytest.fixture
def make_services(
    engine: Engine,
    repo: Repository,
    redis_client: Redis,
    redis_namespace: str,
    s3_client: Any,
    settings: Settings,
) -> Callable[..., Services]:
    def make(redis: Redis | None = None, **overrides: Any) -> Services:
        config = settings.model_copy(
            update={
                "queue_stream": f"{redis_namespace}:jobs",
                "dlq_stream": f"{redis_namespace}:dlq",
                **overrides,
            }
        )
        client = redis if redis is not None else redis_client
        return Services(
            settings=config,
            engine=engine,
            repo=repo,
            redis=client,
            queue=JobQueue(
                client,
                stream=config.queue_stream,
                group=config.queue_group,
                dlq_stream=config.dlq_stream,
            ),
            store=ObjectStore(bucket=config.s3_bucket, region=config.s3_region, client=s3_client),
            rate_limiter=RateLimiter(client, prefix=f"{redis_namespace}:rl"),
        )

    return make


@pytest.fixture
def open_client(make_services: Callable[..., Services]) -> Iterator[Callable[..., TestClient]]:
    """open_client(**settings_overrides) -> a started TestClient (lifespan run)."""
    with ExitStack() as stack:

        def open_(redis: Redis | None = None, **overrides: Any) -> TestClient:
            app = create_app(make_services(redis, **overrides))
            return stack.enter_context(TestClient(app, raise_server_exceptions=False))

        yield open_


@pytest.fixture
def client(open_client: Callable[..., TestClient]) -> TestClient:
    return open_client()


@pytest.fixture
def make_key(repo: Repository) -> Callable[..., Auth]:
    def make(rate_limit: int | None = None) -> Auth:
        _, raw = repo.create_api_key(
            f"tenant-{uuid.uuid4().hex[:6]}", rate_limit_per_minute=rate_limit
        )
        return {"Authorization": f"Bearer {raw}"}

    return make


@pytest.fixture
def auth(make_key: Callable[..., Auth]) -> Auth:
    return make_key()


@pytest.fixture
def spool_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Where direct uploads are spooled, so tests can assert nothing is left behind."""
    spool = tmp_path / "spool"
    spool.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(spool))
    return spool


@pytest.fixture
def hello(samples_dir: Path) -> bytes:
    return (samples_dir / "hello.mp3").read_bytes()


# --- helpers -----------------------------------------------------------------------------
def post_audio(
    client: TestClient,
    auth: Auth,
    body: Any,
    *,
    params: dict[str, Any] | None = None,
    **headers: str,
) -> Any:
    return client.post(
        "/v1/transcriptions",
        content=body,
        params=params,
        headers={"Content-Type": "audio/mpeg", **auth, **headers},
    )


def assert_problem(response: Any, status: int, code: str) -> dict[str, Any]:
    assert response.status_code == status, response.text
    assert response.headers["content-type"] == PROBLEM
    body: dict[str, Any] = response.json()
    assert body["status"] == status
    assert body["code"] == code
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert {"type", "title", "detail"} <= body.keys()
    return body


def s3_keys(s3_client: Any, bucket: str = "tx-test") -> list[str]:
    listing = s3_client.list_objects_v2(Bucket=bucket)
    return [obj["Key"] for obj in listing.get("Contents", [])]


def stream_length(redis_client: Redis, namespace: str) -> int:
    return int(redis_client.xlen(f"{namespace}:jobs"))


def key_id(repo: Repository, auth: Auth) -> uuid.UUID:
    record = repo.authenticate(auth["Authorization"].removeprefix("Bearer "))
    assert record is not None
    return record.id


def insert_job(repo: Repository, owner: uuid.UUID, *, fingerprint: str | None = None) -> JobRecord:
    job, _ = repo.create_job(
        api_key_id=owner,
        audio_key=f"audio/{owner}/{uuid.uuid4()}",
        audio_bytes=100,
        audio_sha256=None,
        options=TranscriptionOptions(),
        webhook_url=None,
        idempotency_key=None,
        request_fingerprint=fingerprint or uuid.uuid4().hex,
    )
    return job


def succeed(repo: Repository, job_id: uuid.UUID) -> None:
    """Drive a queued job to succeeded the way a worker would."""
    claim = repo.claim(job_id, worker_id=WORKER, lease_s=60, max_attempts=3)
    assert claim.job is not None
    repo.record_audio(
        job_id,
        worker_id=WORKER,
        info=AudioInfo(container="mp3", codec="mp3", channels=1, sample_rate=24_000),
        audio_sha256=None,
        audio_bytes=None,
    )
    repo.save_plan(
        job_id,
        worker_id=WORKER,
        plan=ChunkPlan(
            chunks=[PlannedChunk(index=0, start=0, end=32_000)],
            audio_samples=32_000,
            speech_samples=32_000,
            forced_cuts=0,
            channels=[None],
        ),
    )
    segments = [
        Segment(start=0.0, end=1.0, text="Hello", avg_logprob=-0.2),
        Segment(start=1.0, end=2.0, text="world.", avg_logprob=-0.3),
    ]
    repo.save_chunk(
        job_id,
        worker_id=WORKER,
        result=ChunkResult(index=0, start_s=0, end_s=2, engine="fake", segments=segments),
    )
    repo.complete(
        job_id,
        worker_id=WORKER,
        transcript=Transcript(
            language="en",
            language_probability=0.97,
            duration_s=2.0,
            text="Hello world.",
            segments=segments,
            stats=TranscriptStats(
                audio_seconds=2, speech_seconds=2, transcribed_seconds=2, chunks=1, forced_cuts=0
            ),
            audio=AudioInfo(container="mp3", codec="mp3"),
        ),
    )


# --- auth ------------------------------------------------------------------------------
def test_missing_invalid_and_revoked_keys_get_401(
    client: TestClient, repo: Repository, auth: Auth
) -> None:
    revoked = key_id(repo, auth)
    repo.revoke_api_key(revoked)
    for headers in ({}, {"Authorization": "Bearer tx_nope"}, {"Authorization": "Basic Zm9v"}, auth):
        response = client.get("/v1/transcriptions", headers=headers)
        assert_problem(response, 401, "unauthorized")
        assert response.headers["WWW-Authenticate"] == "Bearer"


# --- direct uploads --------------------------------------------------------------------
def test_direct_upload_creates_and_enqueues_a_job(
    client: TestClient,
    repo: Repository,
    auth: Auth,
    hello: bytes,
    s3_client: Any,
    redis_client: Redis,
    redis_namespace: str,
    spool_dir: Path,
) -> None:
    response = post_audio(client, auth, hello, params={"language": "EN", "prompt": "Gettysburg"})

    assert response.status_code == 202, response.text
    body = response.json()
    assert response.headers["Location"] == f"/v1/transcriptions/{body['id']}"
    assert body["status"] == "queued"
    assert body["options"]["language"] == "en"
    assert body["result"] is None
    job = repo.get_job(uuid.UUID(body["id"]))
    assert job is not None
    assert job.audio_bytes == len(hello)
    assert job.audio_key.startswith(f"audio/{key_id(repo, auth)}/")
    assert s3_keys(s3_client) == [job.audio_key]
    assert stream_length(redis_client, redis_namespace) == 1
    assert list(spool_dir.iterdir()) == []


@pytest.mark.parametrize(
    ("sample", "status", "code"),
    [
        ("evil.m3u8", 415, "unsupported_media_type"),
        ("evil.mp3", 415, "unsupported_media_type"),
        ("evil_concat.wav", 415, "unsupported_media_type"),
        ("text.wav", 415, "unsupported_media_type"),
        ("video_only.mp4", 422, "no_audio_stream"),
    ],
)
def test_bad_files_are_rejected_before_anything_is_stored(
    client: TestClient,
    repo: Repository,
    auth: Auth,
    samples_dir: Path,
    s3_client: Any,
    spool_dir: Path,
    sample: str,
    status: int,
    code: str,
) -> None:
    response = post_audio(client, auth, (samples_dir / sample).read_bytes())

    assert_problem(response, status, code)
    assert repo.list_jobs(key_id(repo, auth)) == []
    assert s3_keys(s3_client) == []
    assert list(spool_dir.iterdir()) == []


def test_empty_body_is_400(client: TestClient, auth: Auth) -> None:
    assert_problem(post_audio(client, auth, b""), 400, "bad_request")


def test_declared_size_over_limit_is_413(
    open_client: Callable[..., TestClient], auth: Auth, hello: bytes, spool_dir: Path
) -> None:
    client = open_client(max_direct_upload_bytes=len(hello) - 1)
    assert_problem(post_audio(client, auth, hello), 413, "payload_too_large")
    assert list(spool_dir.iterdir()) == []


def test_chunked_body_over_limit_is_413(
    open_client: Callable[..., TestClient], auth: Auth, spool_dir: Path
) -> None:
    client = open_client(max_direct_upload_bytes=1000)

    def body() -> Iterator[bytes]:
        for _ in range(4):
            yield b"\0" * 600

    response = post_audio(client, auth, body())

    assert_problem(response, 413, "payload_too_large")
    assert list(spool_dir.iterdir()) == []


def test_enqueue_failure_still_accepts_the_job(
    open_client: Callable[..., TestClient], repo: Repository, auth: Auth, hello: bytes
) -> None:
    """Redis down: the committed row is the truth and the sweeper re-enqueues it."""
    dead = Redis(port=1, socket_connect_timeout=0.2, retry=Retry(NoBackoff(), 0))
    client = open_client(redis=dead)

    response = post_audio(client, auth, hello)

    assert response.status_code == 202, response.text
    assert repo.get_job(uuid.UUID(response.json()["id"])) is not None
    dead.close()


# --- presigned uploads -----------------------------------------------------------------
def test_presigned_upload_flow(
    client: TestClient,
    repo: Repository,
    make_key: Callable[..., Auth],
    hello: bytes,
    s3_client: Any,
    redis_client: Redis,
    redis_namespace: str,
) -> None:
    auth = make_key()
    upload = client.post(
        "/v1/uploads", json={"size_bytes": len(hello), "content_type": "audio/mpeg"}, headers=auth
    )
    assert upload.status_code == 201, upload.text
    presigned = upload.json()
    assert presigned["max_bytes"] == len(hello)
    assert {"url", "fields", "expires_at"} <= presigned.keys()
    object_key = presigned["fields"]["key"]
    assert object_key == f"audio/{key_id(repo, auth)}/{presigned['upload_id']}"
    s3_client.put_object(Bucket="tx-test", Key=object_key, Body=hello)

    other = client.post(
        "/v1/transcriptions", json={"upload_id": presigned["upload_id"]}, headers=make_key()
    )
    assert_problem(other, 404, "upload_not_found")
    unknown = client.post("/v1/transcriptions", json={"upload_id": str(uuid.uuid4())}, headers=auth)
    assert_problem(unknown, 404, "upload_not_found")

    response = client.post(
        "/v1/transcriptions",
        json={"upload_id": presigned["upload_id"], "language": "en", "word_timestamps": True},
        headers=auth,
    )
    assert response.status_code == 202, response.text
    job = repo.get_job(uuid.UUID(response.json()["id"]))
    assert job is not None
    assert job.audio_key == object_key
    assert job.audio_bytes == len(hello)
    assert job.options == TranscriptionOptions(language="en", word_timestamps=True)
    assert stream_length(redis_client, redis_namespace) == 1

    again = client.post(
        "/v1/transcriptions",
        json={"upload_id": presigned["upload_id"], "language": "en", "word_timestamps": True},
        headers=auth,
    )
    assert again.status_code == 200
    assert again.headers["X-Deduplicated"] == "true"
    assert s3_keys(s3_client) == [object_key]  # the shared object is not a leftover


def test_upload_declared_over_limit_is_413(
    open_client: Callable[..., TestClient], auth: Auth
) -> None:
    client = open_client(max_presigned_upload_bytes=1000)
    response = client.post("/v1/uploads", json={"size_bytes": 1001}, headers=auth)
    assert_problem(response, 413, "payload_too_large")
    invalid = client.post("/v1/uploads", json={"size_bytes": 0}, headers=auth)
    assert_problem(invalid, 422, "validation_error")


def test_uploaded_object_over_limit_is_413(
    open_client: Callable[..., TestClient], repo: Repository, auth: Auth, s3_client: Any
) -> None:
    client = open_client(max_presigned_upload_bytes=10)
    upload_id = uuid.uuid4()
    s3_client.put_object(
        Bucket="tx-test", Key=f"audio/{key_id(repo, auth)}/{upload_id}", Body=b"x" * 11
    )
    response = client.post("/v1/transcriptions", json={"upload_id": str(upload_id)}, headers=auth)
    assert_problem(response, 413, "payload_too_large")


# --- admission control -----------------------------------------------------------------
def test_idempotency_key_replays_and_rejects_reuse(
    client: TestClient,
    auth: Auth,
    hello: bytes,
    s3_client: Any,
    redis_client: Redis,
    redis_namespace: str,
) -> None:
    first = post_audio(client, auth, hello, **{"Idempotency-Key": "req-1"})
    assert first.status_code == 202

    replay = post_audio(client, auth, hello, **{"Idempotency-Key": "req-1"})
    assert replay.status_code == 200
    assert replay.headers["Idempotent-Replayed"] == "true"
    assert replay.json()["id"] == first.json()["id"]

    reused = post_audio(
        client, auth, hello, params={"language": "fr"}, **{"Idempotency-Key": "req-1"}
    )
    assert_problem(reused, 409, "idempotency_key_reused")

    too_long = post_audio(client, auth, hello, **{"Idempotency-Key": "k" * 256})
    assert_problem(too_long, 422, "validation_error")
    assert len(s3_keys(s3_client)) == 1  # the conflicting upload was cleaned up
    assert stream_length(redis_client, redis_namespace) == 1


def test_concurrent_retries_with_one_idempotency_key_make_one_job(
    client: TestClient,
    repo: Repository,
    auth: Auth,
    hello: bytes,
    s3_client: Any,
    redis_client: Redis,
    redis_namespace: str,
) -> None:
    """Racing requests all pass the early replay check; the unique constraint picks one
    winner and the losers replay it and discard their uploaded copies."""
    with ThreadPoolExecutor(max_workers=6) as pool:
        responses = list(
            pool.map(
                lambda _: post_audio(client, auth, hello, **{"Idempotency-Key": "race"}), range(6)
            )
        )

    assert sorted(r.status_code for r in responses) == [200] * 5 + [202]
    assert len({r.json()["id"] for r in responses}) == 1
    assert len(repo.list_jobs(key_id(repo, auth))) == 1
    assert len(s3_keys(s3_client)) == 1
    assert stream_length(redis_client, redis_namespace) == 1


def test_identical_audio_and_options_are_deduplicated(
    client: TestClient, make_key: Callable[..., Auth], hello: bytes, s3_client: Any
) -> None:
    auth = make_key()
    first = post_audio(client, auth, hello)
    duplicate = post_audio(client, auth, hello)
    other_options = post_audio(client, auth, hello, params={"language": "en"})
    other_key = post_audio(client, make_key(), hello)

    assert duplicate.status_code == 200
    assert duplicate.headers["X-Deduplicated"] == "true"
    assert duplicate.json()["id"] == first.json()["id"]
    assert other_options.status_code == 202
    assert other_key.status_code == 202
    assert len(s3_keys(s3_client)) == 3


def test_full_queue_is_429_with_retry_after(
    open_client: Callable[..., TestClient], auth: Auth, hello: bytes, samples_dir: Path
) -> None:
    client = open_client(max_pending_jobs=1)
    assert post_audio(client, auth, hello).status_code == 202

    response = post_audio(client, auth, (samples_dir / "gaps.mp3").read_bytes())

    assert_problem(response, 429, "queue_full")
    assert response.headers["Retry-After"] == "30"


def test_rate_limit_per_key_with_separate_read_bucket(
    client: TestClient, make_key: Callable[..., Auth]
) -> None:
    auth = make_key(rate_limit=2)
    for remaining in (1, 0):
        ok = client.post("/v1/uploads", json={"size_bytes": 10}, headers=auth)
        assert ok.status_code == 201
        assert ok.headers["RateLimit-Limit"] == "2"
        assert ok.headers["RateLimit-Remaining"] == str(remaining)

    limited = client.post("/v1/uploads", json={"size_bytes": 10}, headers=auth)

    assert_problem(limited, 429, "rate_limited")
    assert 1 <= int(limited.headers["Retry-After"]) <= 60
    assert limited.headers["RateLimit-Remaining"] == "0"
    assert limited.headers["RateLimit-Reset"] == limited.headers["Retry-After"]
    reads = client.get("/v1/transcriptions", headers=auth)
    assert reads.status_code == 200
    assert reads.headers["RateLimit-Limit"] == "20"
    other = client.post("/v1/uploads", json={"size_bytes": 10}, headers=make_key(rate_limit=2))
    assert other.status_code == 201


@pytest.mark.parametrize(
    "params",
    [{"language": "english"}, {"prompt": "x" * 801}, {"langauge": "en"}],
    ids=["language", "prompt", "unknown-param"],
)
def test_invalid_options_are_422(
    client: TestClient, auth: Auth, hello: bytes, params: dict[str, str]
) -> None:
    body = assert_problem(post_audio(client, auth, hello, params=params), 422, "validation_error")
    assert body["errors"][0]["loc"][0] == "query"


def test_malformed_json_requests_are_rejected(client: TestClient, auth: Auth) -> None:
    bad_language = client.post(
        "/v1/transcriptions",
        json={"upload_id": str(uuid.uuid4()), "language": "e"},
        headers=auth,
    )
    body = assert_problem(bad_language, 422, "validation_error")
    assert body["errors"][0]["loc"] == ["body", "language"]
    not_json = client.post(
        "/v1/transcriptions",
        content=b"{",
        headers={**auth, "Content-Type": "application/json"},
    )
    assert_problem(not_json, 422, "validation_error")
    options_in_query = client.post(
        "/v1/transcriptions",
        params={"language": "fr"},
        json={"upload_id": str(uuid.uuid4())},
        headers=auth,
    )
    assert_problem(options_in_query, 400, "bad_request")


@pytest.mark.parametrize(
    "url", ["https://127.0.0.1/hook", "https://10.1.2.3/hook", "http://8.8.8.8/hook"]
)
def test_private_or_plain_http_webhook_is_400(
    open_client: Callable[..., TestClient], auth: Auth, hello: bytes, url: str
) -> None:
    client = open_client(webhook_allow_private_targets=False)
    response = post_audio(client, auth, hello, params={"webhook_url": url})
    assert_problem(response, 400, "bad_request")


# --- reads -----------------------------------------------------------------------------
def test_get_queued_job(client: TestClient, auth: Auth, hello: bytes) -> None:
    created = post_audio(
        client, auth, hello, params={"webhook_url": "https://hooks.example.com/tx"}
    ).json()

    response = client.get(f"/v1/transcriptions/{created['id']}", headers=auth)

    assert response.status_code == 200
    body = response.json()
    assert body == created
    assert body["status"] == "queued"
    assert body["progress"] is None
    assert body["audio"] is None
    assert body["error"] is None
    assert body["result"] is None
    assert body["webhook"] == {
        "url": "https://hooks.example.com/tx",
        "status": None,
        "attempts": 0,
    }
    assert body["links"] == {"self": f"/v1/transcriptions/{created['id']}", "subtitles": None}


def test_get_succeeded_job(client: TestClient, repo: Repository, auth: Auth) -> None:
    job = insert_job(repo, key_id(repo, auth))
    succeed(repo, job.id)

    body = client.get(f"/v1/transcriptions/{job.id}", headers=auth).json()

    assert body["status"] == "succeeded"
    assert body["progress"] == {"chunks_done": 1, "chunks_total": 1, "percent": 100.0}
    assert body["audio"] == {"container": "mp3", "codec": "mp3", "channels": 1, "duration_s": 2.0}
    assert body["language"] == "en"
    assert body["language_probability"] == 0.97
    assert body["started_at"] is not None
    assert body["finished_at"] is not None
    assert body["result"]["text"] == "Hello world."
    assert [s["text"] for s in body["result"]["segments"]] == ["Hello", "world."]
    assert body["result"]["stats"]["chunks"] == 1
    assert body["links"]["subtitles"] == f"/v1/transcriptions/{job.id}/subtitles"

    lean = client.get(
        f"/v1/transcriptions/{job.id}", params={"include_segments": "false"}, headers=auth
    ).json()
    assert lean["result"]["segments"] is None
    assert lean["result"]["text"] == "Hello world."


def test_get_failed_job_shows_error(client: TestClient, repo: Repository, auth: Auth) -> None:
    job = insert_job(repo, key_id(repo, auth))
    repo.fail(job.id, worker_id=None, code="no_audio_stream", http_status=422, message="no audio")

    body = client.get(f"/v1/transcriptions/{job.id}", headers=auth).json()

    assert body["status"] == "failed"
    assert body["error"] == {"code": "no_audio_stream", "status": 422, "message": "no audio"}
    assert body["result"] is None


def test_list_paginates_newest_first(client: TestClient, repo: Repository, auth: Auth) -> None:
    owner = key_id(repo, auth)
    jobs = [insert_job(repo, owner) for _ in range(3)]
    succeed(repo, jobs[0].id)

    first = client.get("/v1/transcriptions", params={"limit": 2}, headers=auth).json()
    assert [item["id"] for item in first["data"]] == [str(jobs[2].id), str(jobs[1].id)]
    assert "result" not in first["data"][0]
    assert first["next_cursor"] is not None

    second = client.get(
        "/v1/transcriptions", params={"limit": 2, "before": first["next_cursor"]}, headers=auth
    ).json()
    assert [item["id"] for item in second["data"]] == [str(jobs[0].id)]
    assert second["next_cursor"] is None

    succeeded = client.get(
        "/v1/transcriptions", params={"status": "succeeded"}, headers=auth
    ).json()
    assert [item["id"] for item in succeeded["data"]] == [str(jobs[0].id)]
    for limit in (0, 101):
        response = client.get("/v1/transcriptions", params={"limit": limit}, headers=auth)
        assert_problem(response, 422, "validation_error")


def test_subtitles_are_409_until_succeeded(
    client: TestClient, repo: Repository, auth: Auth
) -> None:
    job = insert_job(repo, key_id(repo, auth))
    path = f"/v1/transcriptions/{job.id}/subtitles"
    assert_problem(client.get(path, headers=auth), 409, "not_ready")
    succeed(repo, job.id)

    srt = client.get(path, headers=auth)
    vtt = client.get(path, params={"format": "vtt"}, headers=auth)

    assert srt.status_code == 200
    assert srt.headers["content-type"] == "application/x-subrip"
    assert srt.headers["content-disposition"] == f'attachment; filename="transcript-{job.id}.srt"'
    assert srt.text.startswith("1\n00:00:00,000 --> 00:00:01,000\nHello\n")
    assert srt.headers["RateLimit-Limit"] == "600"
    assert vtt.status_code == 200
    assert vtt.headers["content-type"].startswith("text/vtt")
    assert vtt.headers["content-disposition"] == f'attachment; filename="transcript-{job.id}.vtt"'
    assert vtt.text.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello\n")
    bad = client.get(path, params={"format": "txt"}, headers=auth)
    assert_problem(bad, 422, "validation_error")


# --- delete & ownership ----------------------------------------------------------------
def test_delete_removes_row_and_audio(
    client: TestClient, repo: Repository, auth: Auth, hello: bytes, s3_client: Any
) -> None:
    job_id = post_audio(client, auth, hello).json()["id"]
    assert len(s3_keys(s3_client)) == 1

    response = client.delete(f"/v1/transcriptions/{job_id}", headers=auth)

    assert response.status_code == 204
    assert response.content == b""
    assert repo.get_job(uuid.UUID(job_id)) is None
    assert s3_keys(s3_client) == []
    assert_problem(client.get(f"/v1/transcriptions/{job_id}", headers=auth), 404, "not_found")
    assert_problem(client.delete(f"/v1/transcriptions/{job_id}", headers=auth), 404, "not_found")


def test_delete_of_a_processing_job_fences_out_the_worker(
    client: TestClient, repo: Repository, auth: Auth
) -> None:
    job = insert_job(repo, key_id(repo, auth))
    repo.claim(job.id, worker_id=WORKER, lease_s=60, max_attempts=3)

    assert client.delete(f"/v1/transcriptions/{job.id}", headers=auth).status_code == 204
    assert repo.heartbeat(job.id, worker_id=WORKER, lease_s=60) is False


def test_other_keys_jobs_are_404(
    client: TestClient, repo: Repository, make_key: Callable[..., Auth]
) -> None:
    owner = make_key()
    job = insert_job(repo, key_id(repo, owner))
    intruder = make_key()

    for method in ("GET", "DELETE"):
        response = client.request(method, f"/v1/transcriptions/{job.id}", headers=intruder)
        assert_problem(response, 404, "not_found")
    subtitles = client.get(f"/v1/transcriptions/{job.id}/subtitles", headers=intruder)
    assert_problem(subtitles, 404, "not_found")
    assert client.get("/v1/transcriptions", headers=intruder).json()["data"] == []
    assert repo.get_job(job.id) is not None


# --- request context, health, metrics --------------------------------------------------
def test_request_id_is_echoed_or_generated(client: TestClient, auth: Auth) -> None:
    echoed = client.get("/healthz", headers={"X-Request-ID": "trace-abc_1.2"})
    assert echoed.headers["X-Request-ID"] == "trace-abc_1.2"

    for bad in ("has space", "x" * 129, ""):
        generated = client.get("/healthz", headers={"X-Request-ID": bad})
        assert uuid.UUID(generated.headers["X-Request-ID"])

    problem = client.get(
        f"/v1/transcriptions/{uuid.uuid4()}", headers={**auth, "X-Request-ID": "r-1"}
    )
    assert assert_problem(problem, 404, "not_found")["request_id"] == "r-1"
    unknown_route = client.get("/nope")
    assert_problem(unknown_route, 404, "not_found")


def test_unhandled_error_is_an_opaque_500(
    client: TestClient, auth: Auth, hello: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    services: Services = client.app.state.services  # type: ignore[attr-defined]

    def explode() -> int:
        raise RuntimeError("password=hunter2 at db-internal:5432")

    monkeypatch.setattr(services.repo, "count_active", explode)
    response = post_audio(client, auth, hello)

    body = assert_problem(response, 500, "internal_error")
    assert "hunter2" not in response.text
    assert body["detail"] == "internal server error"


def test_healthz(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_reports_every_dependency(
    client: TestClient, open_client: Callable[..., TestClient]
) -> None:
    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json() == {"status": "ok", "checks": {"postgres": "ok", "redis": "ok", "s3": "ok"}}

    dead = Redis(port=1, socket_connect_timeout=0.2, retry=Retry(NoBackoff(), 0))
    unready = open_client(redis=dead).get("/readyz")

    assert unready.status_code == 503
    assert unready.json() == {
        "status": "unavailable",
        "checks": {"postgres": "ok", "redis": "error", "s3": "ok"},
    }
    dead.close()


def test_metrics_label_requests_by_route_template(client: TestClient, auth: Auth) -> None:
    client.get(f"/v1/transcriptions/{uuid.uuid4()}", headers=auth)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "tx_http_requests_total" in response.text
    assert 'route="/v1/transcriptions/{job_id}",status="404"' in response.text


def test_openapi_documents_security_and_problems(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert schema["components"]["securitySchemes"]["HTTPBearer"]["scheme"] == "bearer"
    create = schema["paths"]["/v1/transcriptions"]["post"]
    assert {"application/json", "audio/*"} <= create["requestBody"]["content"].keys()
    assert "application/problem+json" in create["responses"]["429"]["content"]
