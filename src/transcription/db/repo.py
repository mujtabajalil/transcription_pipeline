"""Data access. Every public method is one transaction and returns detached pydantic
records, never ORM objects, so callers can't trip over lazy loads or stale sessions.

Lease fencing: every worker-side write is conditioned on
``status='processing' AND worker_id=:worker_id``. If that matches zero rows the worker
has lost the job to someone else and gets ``LeaseLostError`` — it must stop writing.

All time comparisons use the database clock (``now()``), never the app clock, so lease
decisions agree across hosts whose clocks drift.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from psycopg import errors as pg_errors
from pydantic import BaseModel, ConfigDict
from sqlalchemy import ColumnElement, Update, case, delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from sqlalchemy.orm import Session, sessionmaker

from transcription.db.models import ApiKey, Job, JobChunk
from transcription.domain import (
    SAMPLE_RATE,
    AudioInfo,
    ChunkPlan,
    ChunkResult,
    JobStatus,
    Transcript,
    TranscriptionOptions,
)
from transcription.errors import ConflictError, DependencyUnavailableError, LeaseLostError

log = logging.getLogger(__name__)

_ACTIVE = (JobStatus.QUEUED, JobStatus.PROCESSING)
_TERMINAL = (JobStatus.SUCCEEDED, JobStatus.FAILED)


class ApiKeyRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    key_prefix: str
    webhook_secret: str
    rate_limit_per_minute: int | None
    created_at: datetime
    revoked_at: datetime | None


class JobRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    api_key_id: uuid.UUID
    status: JobStatus
    idempotency_key: str | None
    request_fingerprint: str
    audio_key: str
    audio_bytes: int | None
    audio_sha256: str | None
    options: TranscriptionOptions
    webhook_url: str | None
    audio_info: AudioInfo | None
    duration_s: float | None
    language: str | None
    language_probability: float | None
    plan: ChunkPlan | None
    chunks_total: int | None
    chunks_done: int
    attempts: int
    worker_id: str | None
    lease_expires_at: datetime | None
    error_code: str | None
    error_status: int | None
    error_message: str | None
    result: Transcript | None
    webhook_status: str | None
    webhook_attempts: int
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    audio_deleted_at: datetime | None


class ClaimOutcome(StrEnum):
    CLAIMED = "claimed"
    """Lease acquired; attempts incremented."""
    BUSY = "busy"
    """Another worker holds a live lease (duplicate message). Ack and move on."""
    TERMINAL = "terminal"
    """Already succeeded/failed. Ack and move on."""
    EXHAUSTED = "exhausted"
    """attempts >= max_attempts and no live lease. Caller dead-letters + fails it."""
    MISSING = "missing"
    """No such job (deleted). Ack and move on."""


class ClaimResult(BaseModel):
    outcome: ClaimOutcome
    job: JobRecord | None = None


class WebhookTask(BaseModel):
    job: JobRecord
    secret: str
    attempt: int
    """1-based attempt number this delivery represents."""


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def _now_plus(seconds: float) -> ColumnElement[datetime]:
    return func.now() + timedelta(seconds=seconds)


def _fenced(job_id: uuid.UUID, worker_id: str) -> Update:
    return update(Job).where(
        Job.id == job_id, Job.status == JobStatus.PROCESSING, Job.worker_id == worker_id
    )


def _outbox_values() -> dict[str, Any]:
    """Queue the webhook in the same UPDATE that makes the job terminal, so a crash
    can't leave a finished job whose webhook is never sent."""
    has_hook = Job.webhook_url.is_not(None)
    return {
        "webhook_status": case((has_hook, "pending")),
        "webhook_next_at": case((has_hook, func.now())),
    }


def _job(row: Job) -> JobRecord:
    return JobRecord.model_validate(row)


def _lease_lost(job_id: uuid.UUID, worker_id: str) -> LeaseLostError:
    log.warning("job lease lost", extra={"job_id": str(job_id), "worker_id": worker_id})
    return LeaseLostError(f"job {job_id} is no longer leased to {worker_id}")


class Repository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._sf = session_factory

    @contextmanager
    def _tx(self) -> Iterator[Session]:
        """One transaction; connection-level failures surface as the retryable
        DependencyUnavailableError instead of a driver exception.

        Records must be built inside the block: the caller's sessionmaker may expire
        instances on commit, after which a detached ORM row can't be read."""
        try:
            with self._sf.begin() as session:
                yield session
        except (OperationalError, PoolTimeoutError) as exc:
            # str(exc) carries the SQL and its bound parameters (key hashes, webhook
            # secrets); only the driver's own message goes to the log, and the raised
            # error, which the API renders to clients, stays generic.
            log.warning("database unavailable", extra={"error": str(getattr(exc, "orig", exc))})
            raise DependencyUnavailableError("database unavailable") from exc

    def ping(self) -> None:
        """SELECT 1; raises on failure."""
        with self._tx() as session:
            session.execute(select(1))

    # --- API keys ----------------------------------------------------------------------
    def create_api_key(
        self,
        name: str,
        *,
        raw_key: str | None = None,
        rate_limit_per_minute: int | None = None,
    ) -> tuple[ApiKeyRecord, str]:
        """Create a key and return (record, raw_key). The raw key is shown once and only
        its sha256 is stored. Generated keys look like ``tx_<43 url-safe chars>``. If
        ``raw_key`` is given and already exists, return the existing record (bootstrap
        is idempotent). Generates a random webhook_secret."""
        raw = raw_key if raw_key is not None else f"tx_{secrets.token_urlsafe(32)}"
        key_hash = _hash_key(raw)
        insert = (
            pg_insert(ApiKey)
            .values(
                name=name,
                key_prefix=raw[:10],
                key_hash=key_hash,
                webhook_secret=f"whsec_{secrets.token_urlsafe(32)}",
                rate_limit_per_minute=rate_limit_per_minute,
            )
            .on_conflict_do_nothing(index_elements=[ApiKey.key_hash])
            .returning(ApiKey)
        )
        with self._tx() as session:
            key = session.scalars(insert).one_or_none()
            if key is None:
                key = session.scalars(select(ApiKey).where(ApiKey.key_hash == key_hash)).one()
            return ApiKeyRecord.model_validate(key), raw

    def authenticate(self, raw_key: str) -> ApiKeyRecord | None:
        """Lookup by sha256(raw_key); None if unknown or revoked."""
        query = select(ApiKey).where(
            ApiKey.key_hash == _hash_key(raw_key), ApiKey.revoked_at.is_(None)
        )
        with self._tx() as session:
            key = session.scalars(query).one_or_none()
            return ApiKeyRecord.model_validate(key) if key else None

    def revoke_api_key(self, key_id: uuid.UUID) -> bool:
        """True if this call revoked an active key; False if the key is unknown or was
        already revoked (``tx-admin revoke-key`` reports that as "no active key"). The
        first revocation time is kept."""
        stmt = (
            update(ApiKey)
            .where(ApiKey.id == key_id, ApiKey.revoked_at.is_(None))
            .values(revoked_at=func.now())
            .returning(ApiKey.id)
        )
        with self._tx() as session:
            return session.scalars(stmt).one_or_none() is not None

    # --- jobs: API side ----------------------------------------------------------------
    def create_job(
        self,
        *,
        api_key_id: uuid.UUID,
        audio_key: str,
        audio_bytes: int | None,
        audio_sha256: str | None,
        options: TranscriptionOptions,
        webhook_url: str | None,
        idempotency_key: str | None,
        request_fingerprint: str,
    ) -> tuple[JobRecord, bool]:
        """Insert a queued job. Returns (job, created). If (api_key_id, idempotency_key)
        already exists, returns (existing_job, False) without inserting — the caller
        compares fingerprints and raises IdempotencyMismatchError on mismatch. Must be
        race-safe (INSERT ... ON CONFLICT DO NOTHING, then SELECT)."""
        insert = pg_insert(Job).values(
            api_key_id=api_key_id,
            audio_key=audio_key,
            audio_bytes=audio_bytes,
            audio_sha256=audio_sha256,
            options=options.model_dump(mode="json"),
            webhook_url=webhook_url,
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
        )
        with self._tx() as session:
            if idempotency_key is None:
                return _job(session.scalars(insert.returning(Job)).one()), True
            created = session.scalars(
                insert.on_conflict_do_nothing(constraint="uq_jobs_idempotency").returning(Job)
            ).one_or_none()
            if created is not None:
                return _job(created), True
            # READ COMMITTED: this statement sees the row the conflicting insert committed.
            existing = session.scalars(
                select(Job).where(
                    Job.api_key_id == api_key_id, Job.idempotency_key == idempotency_key
                )
            ).one_or_none()
            if existing is None:
                raise ConflictError("the job for this Idempotency-Key was just deleted; retry")
            return _job(existing), False

    def find_by_fingerprint(
        self, api_key_id: uuid.UUID, request_fingerprint: str
    ) -> JobRecord | None:
        """Most recent job of this key with this fingerprint that has not failed."""
        query = (
            select(Job)
            .where(
                Job.api_key_id == api_key_id,
                Job.request_fingerprint == request_fingerprint,
                Job.status != JobStatus.FAILED,
            )
            .order_by(Job.created_at.desc())
            .limit(1)
        )
        with self._tx() as session:
            job = session.scalars(query).first()
            return _job(job) if job else None

    def get_job(
        self, job_id: uuid.UUID, *, api_key_id: uuid.UUID | None = None
    ) -> JobRecord | None:
        """api_key_id given → only return the job if that key owns it."""
        query = select(Job).where(Job.id == job_id)
        if api_key_id is not None:
            query = query.where(Job.api_key_id == api_key_id)
        with self._tx() as session:
            job = session.scalars(query).one_or_none()
            return _job(job) if job else None

    def list_jobs(
        self,
        api_key_id: uuid.UUID,
        *,
        limit: int = 20,
        before: datetime | None = None,
        status: JobStatus | None = None,
    ) -> list[JobRecord]:
        """Newest first. ``before`` is an exclusive created_at cursor."""
        query = (
            select(Job)
            .where(Job.api_key_id == api_key_id)
            .order_by(Job.created_at.desc(), Job.id.desc())
            .limit(limit)
        )
        if before is not None:
            query = query.where(Job.created_at < before)
        if status is not None:
            query = query.where(Job.status == status)
        with self._tx() as session:
            return [_job(job) for job in session.scalars(query)]

    def count_active(self) -> int:
        """queued + processing, all keys (global backpressure)."""
        query = select(func.count()).select_from(Job).where(Job.status.in_(_ACTIVE))
        with self._tx() as session:
            return session.scalars(query).one()

    def delete_job(self, job_id: uuid.UUID, api_key_id: uuid.UUID) -> JobRecord | None:
        """Delete the job (chunks cascade) and return what was deleted so the caller can
        remove the audio object. None if not found/not owned."""
        stmt = delete(Job).where(Job.id == job_id, Job.api_key_id == api_key_id).returning(Job)
        with self._tx() as session:
            job = session.scalars(stmt).one_or_none()
            return _job(job) if job else None

    # --- jobs: worker side -------------------------------------------------------------
    def claim(
        self, job_id: uuid.UUID, *, worker_id: str, lease_s: int, max_attempts: int
    ) -> ClaimResult:
        """Atomically take the job if it is queued, or processing with an expired lease,
        and attempts < max_attempts: set processing, worker_id, lease_expires_at =
        now()+lease_s, attempts += 1, started_at = coalesce(started_at, now()), clear
        error fields. Otherwise classify why not (see ClaimOutcome)."""
        take = (
            update(Job)
            .where(
                Job.id == job_id,
                Job.attempts < max_attempts,
                or_(
                    Job.status == JobStatus.QUEUED,
                    (Job.status == JobStatus.PROCESSING) & (Job.lease_expires_at < func.now()),
                ),
            )
            .values(
                status=JobStatus.PROCESSING,
                worker_id=worker_id,
                lease_expires_at=_now_plus(lease_s),
                attempts=Job.attempts + 1,
                started_at=func.coalesce(Job.started_at, func.now()),
                error_code=None,
                error_status=None,
                error_message=None,
                updated_at=func.now(),
            )
            .returning(Job)
        )
        with self._tx() as session:
            claimed = session.scalars(take).one_or_none()
            if claimed is not None:
                return ClaimResult(outcome=ClaimOutcome.CLAIMED, job=_job(claimed))
            live = (Job.status == JobStatus.PROCESSING) & (Job.lease_expires_at >= func.now())
            row = session.execute(select(Job, live).where(Job.id == job_id)).one_or_none()
            if row is None:
                return ClaimResult(outcome=ClaimOutcome.MISSING)
            job, lease_live = row
            return ClaimResult(outcome=_classify(job, lease_live, max_attempts), job=_job(job))

    def heartbeat(self, job_id: uuid.UUID, *, worker_id: str, lease_s: int) -> bool:
        """Extend the lease. False if we no longer own it."""
        stmt = (
            _fenced(job_id, worker_id)
            .values(lease_expires_at=_now_plus(lease_s), updated_at=func.now())
            .returning(Job.id)
        )
        with self._tx() as session:
            return session.scalars(stmt).one_or_none() is not None

    def record_audio(
        self,
        job_id: uuid.UUID,
        *,
        worker_id: str,
        info: AudioInfo,
        audio_sha256: str | None,
        audio_bytes: int | None,
    ) -> None:
        """Store probe result (+ hash/size computed on download). LeaseLostError if fenced."""
        values: dict[str, Any] = {"audio_info": info.model_dump(mode="json")}
        # None means "not computed here": keep what the API recorded for direct uploads.
        if audio_sha256 is not None:
            values["audio_sha256"] = audio_sha256
        if audio_bytes is not None:
            values["audio_bytes"] = audio_bytes
        self._write_fenced(job_id, worker_id, values)

    def save_plan(self, job_id: uuid.UUID, *, worker_id: str, plan: ChunkPlan) -> None:
        """Store plan, chunks_total=plan.total, duration_s. LeaseLostError if fenced."""
        self._write_fenced(
            job_id,
            worker_id,
            {
                "plan": plan.model_dump(mode="json"),
                "chunks_total": plan.total,
                "duration_s": plan.audio_samples / SAMPLE_RATE,
            },
        )

    def save_chunk(self, job_id: uuid.UUID, *, worker_id: str, result: ChunkResult) -> int:
        """Upsert job_chunks(job_id, result.index) and set jobs.chunks_done to the number
        of chunk rows, in one transaction that also checks the lease. Returns
        chunks_done. LeaseLostError if fenced."""
        payload = result.model_dump(mode="json")
        upsert = (
            pg_insert(JobChunk)
            .values(job_id=job_id, idx=result.index, result=payload)
            .on_conflict_do_update(
                index_elements=[JobChunk.job_id, JobChunk.idx], set_={"result": payload}
            )
        )
        chunk_rows = (
            select(func.count()).select_from(JobChunk).where(JobChunk.job_id == job_id)
        ).scalar_subquery()
        count = (
            _fenced(job_id, worker_id)
            .values(chunks_done=chunk_rows, updated_at=func.now())
            .returning(Job.chunks_done)
        )
        try:
            with self._tx() as session:
                session.execute(upsert)
                done = session.scalars(count).one_or_none()
                if done is None:
                    raise _lease_lost(job_id, worker_id)  # rolls the upsert back
                return done
        except IntegrityError as exc:
            if isinstance(exc.orig, pg_errors.ForeignKeyViolation):  # job was deleted
                raise _lease_lost(job_id, worker_id) from exc
            raise

    def load_chunks(self, job_id: uuid.UUID) -> dict[int, ChunkResult]:
        query = select(JobChunk.idx, JobChunk.result).where(JobChunk.job_id == job_id)
        with self._tx() as session:
            rows = session.execute(query).all()
        return {idx: ChunkResult.model_validate(result) for idx, result in rows}

    def save_language(
        self, job_id: uuid.UUID, *, worker_id: str, language: str, probability: float | None
    ) -> None:
        self._write_fenced(
            job_id, worker_id, {"language": language, "language_probability": probability}
        )

    def complete(self, job_id: uuid.UUID, *, worker_id: str, transcript: Transcript) -> JobRecord:
        """status=succeeded, result, language(+prob), duration_s, finished_at, lease
        cleared; webhook_status='pending' (webhook_next_at=now()) iff webhook_url is set
        — same transaction, so the webhook outbox can't miss it. LeaseLostError if fenced."""
        return self._write_fenced(
            job_id,
            worker_id,
            {
                "status": JobStatus.SUCCEEDED,
                "result": transcript.model_dump(mode="json"),
                "language": transcript.language,
                "language_probability": transcript.language_probability,
                "duration_s": transcript.duration_s,
                "finished_at": func.now(),
                "lease_expires_at": None,
                **_outbox_values(),
            },
        )

    def fail(
        self,
        job_id: uuid.UUID,
        *,
        worker_id: str | None,
        code: str,
        http_status: int,
        message: str,
    ) -> JobRecord | None:
        """status=failed with error fields, finished_at, lease cleared, webhook pending
        iff webhook_url. worker_id=None skips fencing (DLQ path) but still never
        overwrites a terminal job. Returns the updated job, or None if nothing changed.

        With a ``worker_id`` this is a fenced write like any other: LeaseLostError if
        the worker no longer owns the job."""
        stmt = (
            update(Job).where(Job.id == job_id, Job.status.in_(_ACTIVE))
            if worker_id is None
            else _fenced(job_id, worker_id)
        )
        stmt = stmt.values(
            status=JobStatus.FAILED,
            error_code=code,
            error_status=http_status,
            error_message=message,
            finished_at=func.now(),
            lease_expires_at=None,
            updated_at=func.now(),
            **_outbox_values(),
        ).returning(Job)
        with self._tx() as session:
            job = session.scalars(stmt).one_or_none()
            record = _job(job) if job else None
        if record is None and worker_id is not None:
            raise _lease_lost(job_id, worker_id)
        return record

    def release(self, job_id: uuid.UUID, *, worker_id: str, error_message: str) -> bool:
        """Transient failure: back to queued, lease cleared, error_message kept for
        visibility, attempts unchanged. False if fenced."""
        stmt = (
            _fenced(job_id, worker_id)
            .values(
                status=JobStatus.QUEUED,
                worker_id=None,
                lease_expires_at=None,
                error_message=error_message,
                updated_at=func.now(),
            )
            .returning(Job.id)
        )
        with self._tx() as session:
            return session.scalars(stmt).one_or_none() is not None

    def requeue_orphans(
        self, *, queued_older_than_s: int, lease_grace_s: int, limit: int = 100
    ) -> list[uuid.UUID]:
        """Find jobs whose wake-up message may be lost — queued and untouched for
        queued_older_than_s, or processing with lease expired more than lease_grace_s
        ago — bump their updated_at (so they aren't re-swept immediately) and return
        their ids. Uses FOR UPDATE SKIP LOCKED so concurrent sweepers don't collide."""
        grace_ago = _now_plus(-lease_grace_s)
        orphans = (
            select(Job.id)
            .where(
                or_(
                    (Job.status == JobStatus.QUEUED)
                    & (Job.updated_at < _now_plus(-queued_older_than_s)),
                    # updated_at keeps a just-swept processing job out of the next sweep.
                    (Job.status == JobStatus.PROCESSING)
                    & (Job.lease_expires_at < grace_ago)
                    & (Job.updated_at < grace_ago),
                )
            )
            .order_by(Job.updated_at)
            .limit(limit)
            .with_for_update(skip_locked=True)
            .cte("orphans")
        )
        stmt = (
            update(Job)
            .where(Job.id == orphans.c.id)
            .values(updated_at=func.now())
            .returning(Job.id)
            .execution_options(synchronize_session=False)
        )
        with self._tx() as session:
            ids = list(session.scalars(stmt))
        if ids:
            log.info("requeued orphaned jobs", extra={"count": len(ids)})
        return ids

    # --- webhook outbox ----------------------------------------------------------------
    def claim_webhooks(self, *, limit: int, lease_s: int) -> list[WebhookTask]:
        """Due deliveries: webhook_status='pending' and webhook_next_at <= now(), or
        'sending' whose lease (webhook_next_at) expired. Marks them 'sending',
        webhook_next_at=now()+lease_s, webhook_attempts += 1. SKIP LOCKED."""
        due = (
            select(Job.id)
            .where(
                Job.webhook_status.in_(("pending", "sending")), Job.webhook_next_at <= func.now()
            )
            .order_by(Job.webhook_next_at)
            .limit(limit)
            .with_for_update(skip_locked=True)
            .cte("due")
        )
        stmt = (
            update(Job)
            .where(Job.id == due.c.id, ApiKey.id == Job.api_key_id)
            .values(
                webhook_status="sending",
                webhook_next_at=_now_plus(lease_s),
                webhook_attempts=Job.webhook_attempts + 1,
                updated_at=func.now(),
            )
            .returning(Job, ApiKey.webhook_secret)
            .execution_options(synchronize_session=False)
        )
        with self._tx() as session:
            return [
                WebhookTask(job=_job(job), secret=secret, attempt=job.webhook_attempts)
                for job, secret in session.execute(stmt)
            ]

    def finish_webhook(
        self,
        job_id: uuid.UUID,
        *,
        delivered: bool,
        error: str | None,
        retry_in_s: float | None,
    ) -> None:
        """delivered → 'delivered'. Else retry_in_s set → 'pending' at now()+retry_in_s;
        retry_in_s None → 'failed' (gave up). Records webhook_last_error."""
        if delivered:
            status, next_at = "delivered", None
        elif retry_in_s is not None:
            status, next_at = "pending", _now_plus(retry_in_s)
        else:
            status, next_at = "failed", None
        # Only an in-flight delivery can finish: a late duplicate must not resurrect a
        # delivery that another dispatcher already settled.
        stmt = (
            update(Job)
            .where(Job.id == job_id, Job.webhook_status == "sending")
            .values(
                webhook_status=status,
                webhook_next_at=next_at,
                webhook_last_error=error,
                updated_at=func.now(),
            )
        )
        with self._tx() as session:
            session.execute(stmt)

    def mark_audio_deleted(self, job_id: uuid.UUID) -> None:
        stmt = (
            update(Job)
            .where(Job.id == job_id)
            .values(
                audio_deleted_at=func.coalesce(Job.audio_deleted_at, func.now()),
                updated_at=func.now(),
            )
        )
        with self._tx() as session:
            session.execute(stmt)

    def _write_fenced(self, job_id: uuid.UUID, worker_id: str, values: dict[str, Any]) -> JobRecord:
        stmt = _fenced(job_id, worker_id).values(updated_at=func.now(), **values).returning(Job)
        with self._tx() as session:
            job = session.scalars(stmt).one_or_none()
            record = _job(job) if job else None
        if record is None:
            raise _lease_lost(job_id, worker_id)
        return record


def _classify(job: Job, lease_live: bool, max_attempts: int) -> ClaimOutcome:
    """Why a claim matched nothing. Called only after the conditional UPDATE missed."""
    if job.status in _TERMINAL:
        return ClaimOutcome.TERMINAL
    if lease_live:
        return ClaimOutcome.BUSY
    if job.attempts >= max_attempts:
        return ClaimOutcome.EXHAUSTED
    # Claimable now, so the UPDATE raced a concurrent release; whoever released it
    # still holds the stream message that will be redelivered.
    return ClaimOutcome.BUSY
