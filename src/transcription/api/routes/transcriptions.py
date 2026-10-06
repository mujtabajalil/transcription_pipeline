"""Transcription jobs: create (direct upload, presigned upload or a multipart batch of
direct uploads), read, list, captions, delete.

Admission control on create runs in the order of docs/DESIGN.md: auth -> rate limit
(dependencies) -> idempotency replay -> backpressure -> size -> probe -> content dedupe
-> insert + enqueue. Cheap checks reject before expensive ones, and a client retrying
with its Idempotency-Key gets its job back even while the queue is full.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import IO, Annotated, Any, Literal

from botocore.exceptions import ClientError
from fastapi import APIRouter, Header, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from transcription.api.deps import AuthorizedKey, ServicesDep, charge
from transcription.api.problems import problem_of, problem_responses
from transcription.api.schemas import (
    TRANSCRIPTIONS_PATH,
    BatchItem,
    CreateTranscriptionRequest,
    TranscriptionBatch,
    TranscriptionJob,
    TranscriptionList,
    TranscriptionParams,
    TranscriptionSummary,
    job_path,
)
from transcription.audio.probe import probe
from transcription.db.repo import ApiKeyRecord, JobRecord
from transcription.domain import JobStatus, Transcript, TranscriptionOptions
from transcription.errors import (
    BadRequestError,
    DependencyUnavailableError,
    IdempotencyMismatchError,
    NotFoundError,
    NotReadyError,
    PayloadTooLargeError,
    QueueFullError,
    TranscriptionError,
    UnsupportedMediaError,
)
from transcription.formats import to_srt, to_vtt
from transcription.metrics import JOBS_CREATED, JOBS_PENDING, UPLOAD_BYTES
from transcription.services import Services
from transcription.storage import ObjectStore
from transcription.webhooks import validate_target

log = logging.getLogger(__name__)

router = APIRouter(prefix=TRANSCRIPTIONS_PATH, tags=["transcriptions"])

QUEUE_FULL_RETRY_AFTER_S = 30
_JSON_BODY_LIMIT = 64 * 1024
_BINARY = {"schema": {"type": "string", "format": "binary"}}
_CREATE_BODY: dict[str, Any] = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {"schema": CreateTranscriptionRequest.model_json_schema()},
            "audio/*": _BINARY,
            "video/*": _BINARY,
            "application/octet-stream": _BINARY,
        },
    }
}
_BATCH_BODY: dict[str, Any] = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {"file": {"type": "array", "items": _BINARY["schema"]}},
                }
            }
        },
    }
}
_FILE_READ_CHUNK = 1024 * 1024
_SUBTITLE_FORMATS: dict[str, tuple[Callable[[Transcript], str], str]] = {
    "srt": (to_srt, "application/x-subrip"),
    "vtt": (to_vtt, "text/vtt"),
}


class UploadNotFoundError(NotFoundError):
    code = "upload_not_found"
    default_message = "no upload with this upload_id exists for this API key"


class _Outcome(Enum):
    CREATED = "created"
    REPLAYED = "replayed"
    """Same Idempotency-Key, same request: the original job."""
    DEDUPLICATED = "deduplicated"
    """Same audio and options as an existing job of this key."""


# --- create ----------------------------------------------------------------------------
@router.post(
    "",
    status_code=202,
    response_model=TranscriptionJob,
    responses={
        200: {"model": TranscriptionJob, "description": "Idempotent replay or duplicate"},
        **problem_responses(400, 401, 404, 409, 413, 415, 422, 429, 503),
    },
    openapi_extra=_CREATE_BODY,
)
async def create_transcription(
    request: Request,
    response: Response,
    services: ServicesDep,
    api_key: AuthorizedKey,
    params: Annotated[TranscriptionParams, Query()],
    idempotency_key: Annotated[
        str | None,
        Header(
            min_length=1,
            max_length=255,
            description="Retrying with the same key returns the original job (200).",
        ),
    ] = None,
) -> TranscriptionJob:
    """Create a job from an upload (``application/json`` body with ``upload_id``) or
    from the audio itself as the raw body (any other Content-Type, options in the query
    string, at most ``max_direct_upload_bytes``).

    202 with ``Location`` for a new job. 200 with the existing job when the
    Idempotency-Key was already used for the same request (``Idempotent-Replayed:
    true``) or, without a key, when this key already submitted the same audio with the
    same options (``X-Deduplicated: true``)."""
    if _media_type(request) == "application/json":
        if params != TranscriptionParams():
            # Ignoring them silently would run the job with options the client didn't get.
            raise BadRequestError("with a JSON body, options go in the body, not the query")
        payload = await _read_json(request)
        admission = await _admission(services, api_key, payload, idempotency_key)
        job, outcome = await _admit_upload(admission, payload.upload_id)
    else:
        admission = await _admission(services, api_key, params, idempotency_key)
        body = _body_chunks(request, services.settings.max_direct_upload_bytes)
        job, outcome = await _admit_direct(admission, body)
    match outcome:
        case _Outcome.CREATED:
            response.headers["Location"] = job_path(job.id)
        case _Outcome.REPLAYED:
            response.status_code = 200
            response.headers["Idempotent-Replayed"] = "true"
        case _Outcome.DEDUPLICATED:
            response.status_code = 200
            response.headers["X-Deduplicated"] = "true"
    return TranscriptionJob.from_job(job)


@router.post(
    "/batch",
    status_code=207,
    response_model=TranscriptionBatch,
    responses=problem_responses(400, 401, 413, 415, 422, 429),
    openapi_extra=_BATCH_BODY,
)
async def create_transcription_batch(
    request: Request,
    services: ServicesDep,
    api_key: AuthorizedKey,
    params: Annotated[TranscriptionParams, Query()],
    idempotency_key: Annotated[
        str | None,
        Header(
            min_length=1,
            max_length=240,  # leaves room for the per-file suffix in a 255-char column
            description="Retrying the batch with the same key replays the files it created.",
        ),
    ] = None,
) -> TranscriptionBatch:
    """One job per file of a ``multipart/form-data`` body (repeat the ``file`` part).
    Options go in the query string and apply to every file. The whole body counts as
    one direct upload against ``max_direct_upload_bytes``; each file costs one request
    of the rate limit.

    Each file is admitted like a single direct upload, independently of the others: a
    corrupt file or a full queue rejects that file, not the batch. Always 207 once the
    body is accepted; ``data`` says what happened to each file, in upload order. With an
    Idempotency-Key, file ``i`` uses ``<key>:<i>``, so retrying a partly rejected batch
    replays the jobs it created and admits only the rest."""
    if _media_type(request) != "multipart/form-data":
        raise UnsupportedMediaError("send the files as multipart/form-data")
    settings = services.settings
    parser = MultiPartParser(
        request.headers,
        _body_chunks(request, settings.max_direct_upload_bytes),
        max_files=settings.max_batch_files,
        max_fields=0,  # options go in the query string, as for a single direct upload
    )
    try:
        form = await parser.parse()
    except MultiPartException as exc:
        raise BadRequestError(f"invalid multipart body: {exc.message}") from exc
    try:
        files = [value for _, value in form.multi_items() if isinstance(value, UploadFile)]
        if not files:
            raise BadRequestError("no files in the request")
        if len(files) > 1:  # the rate-limit dependency already charged the first
            await run_in_threadpool(charge, request, services, api_key, cost=len(files) - 1)
        admission = await _admission(services, api_key, params, idempotency_key)
        return TranscriptionBatch(
            data=[await _batch_item(request, admission, i, file) for i, file in enumerate(files)]
        )
    finally:
        await form.close()


async def _batch_item(
    request: Request, admission: _Admission, index: int, file: UploadFile
) -> BatchItem:
    if admission.idempotency_key is not None:
        admission = replace(admission, idempotency_key=f"{admission.idempotency_key}:{index}")
    try:
        job, outcome = await _admit_direct(admission, _file_chunks(file))
    except TranscriptionError as exc:
        log.info(
            "batch file rejected",
            extra={"index": index, "code": exc.code, "status": exc.http_status},
        )
        return BatchItem(
            filename=file.filename, status=exc.http_status, job=None, error=problem_of(request, exc)
        )
    return BatchItem(
        filename=file.filename,
        status=202 if outcome is _Outcome.CREATED else 200,
        job=TranscriptionJob.from_job(job),
        error=None,
    )


@dataclass(frozen=True)
class _Admission:
    """One create request on its way through admission control."""

    services: Services
    api_key: ApiKeyRecord
    options: TranscriptionOptions
    webhook_url: str | None
    idempotency_key: str | None

    async def replay(self, fingerprint: str) -> JobRecord | None:
        """The job this Idempotency-Key already created for the same request.

        The repository has no lookup by idempotency key, so this early check finds a
        live job with the same fingerprint carrying our key. A key reused for a
        different request, or whose job failed, is resolved by ``insert``, where the
        unique constraint is the source of truth."""
        if self.idempotency_key is None:
            return None
        job = await run_in_threadpool(
            self.services.repo.find_by_fingerprint, self.api_key.id, fingerprint
        )
        return job if job is not None and job.idempotency_key == self.idempotency_key else None

    async def check_capacity(self) -> None:
        active = await run_in_threadpool(self.services.repo.count_active)
        JOBS_PENDING.set(active)
        if active >= self.services.settings.max_pending_jobs:
            raise QueueFullError(retry_after_s=QUEUE_FULL_RETRY_AFTER_S)

    async def duplicate(self, fingerprint: str) -> JobRecord | None:
        """An existing job of this key for the same audio and options. Skipped with an
        Idempotency-Key: the client then asked for exactly-once semantics on its key."""
        if self.idempotency_key is not None or not self.services.settings.dedupe_by_content:
            return None
        return await run_in_threadpool(
            self.services.repo.find_by_fingerprint, self.api_key.id, fingerprint
        )

    async def insert(
        self,
        *,
        audio_key: str,
        audio_bytes: int,
        audio_sha256: str | None,
        fingerprint: str,
        source: Literal["direct", "upload"],
    ) -> tuple[JobRecord, _Outcome]:
        job, created = await run_in_threadpool(
            lambda: self.services.repo.create_job(
                api_key_id=self.api_key.id,
                audio_key=audio_key,
                audio_bytes=audio_bytes,
                audio_sha256=audio_sha256,
                options=self.options,
                webhook_url=self.webhook_url,
                idempotency_key=self.idempotency_key,
                request_fingerprint=fingerprint,
            )
        )
        if not created:
            if job.request_fingerprint != fingerprint:
                raise IdempotencyMismatchError()
            return job, _Outcome.REPLAYED
        try:
            await run_in_threadpool(self.services.queue.enqueue, job.id)
        except DependencyUnavailableError:
            # The committed row is the source of truth: the worker's sweeper re-enqueues
            # queued jobs whose message never arrived. Failing now would make the client
            # retry into a replay or a duplicate for no benefit.
            log.warning("job created but not enqueued", extra={"job_id": str(job.id)})
        JOBS_CREATED.labels(source=source).inc()
        log.info(
            "job created",
            extra={"job_id": str(job.id), "key_id": str(self.api_key.id), "source": source},
        )
        return job, _Outcome.CREATED


async def _admission(
    services: Services,
    api_key: ApiKeyRecord,
    params: TranscriptionParams,
    idempotency_key: str | None,
) -> _Admission:
    if params.webhook_url is not None:
        # Resolves DNS, hence the threadpool.
        await run_in_threadpool(
            validate_target,
            params.webhook_url,
            allow_private=services.settings.webhook_allow_private_targets,
        )
    return _Admission(
        services=services,
        api_key=api_key,
        options=params.options(),
        webhook_url=params.webhook_url,
        idempotency_key=idempotency_key,
    )


async def _admit_upload(admission: _Admission, upload_id: uuid.UUID) -> tuple[JobRecord, _Outcome]:
    """Presigned upload: the object is already in S3 under the caller's prefix, so
    another key's upload_id simply isn't found. The worker probes it."""
    services = admission.services
    key = services.store.key_for(admission.api_key.id, upload_id)
    info = await run_in_threadpool(services.store.head, key)
    if info is None:
        raise UploadNotFoundError()
    # The ETag pins the object's content (re-uploading to the same key is a new request);
    # it is why the HEAD has to come before the replay check.
    fingerprint = admission.options.fingerprint(f"s3:{key}:{info.etag or ''}")
    if (job := await admission.replay(fingerprint)) is not None:
        return job, _Outcome.REPLAYED
    await admission.check_capacity()
    if info.size > services.settings.max_presigned_upload_bytes:
        raise PayloadTooLargeError("uploaded object exceeds the upload size limit")
    if (job := await admission.duplicate(fingerprint)) is not None:
        return job, _Outcome.DEDUPLICATED
    return await admission.insert(
        audio_key=key,
        audio_bytes=info.size,
        audio_sha256=None,
        fingerprint=fingerprint,
        source="upload",
    )


async def _admit_direct(
    admission: _Admission, body: AsyncIterator[bytes]
) -> tuple[JobRecord, _Outcome]:
    """Direct upload (the raw body, or one file of a batch): spooled to a temp file
    (always removed) while hashing, probed so bad files fail now with 415/422, then
    stored in S3."""
    services = admission.services
    with tempfile.NamedTemporaryFile(prefix="tx-upload-") as spool:
        path = Path(spool.name)
        size, sha256 = await _spool(body, spool)
        if size == 0:
            raise BadRequestError("the file is empty")
        # The fingerprint covers the body's hash, so for direct uploads the replay check
        # can only run once the whole body has been received.
        fingerprint = admission.options.fingerprint(f"sha256:{sha256}")
        if (job := await admission.replay(fingerprint)) is not None:
            return job, _Outcome.REPLAYED
        await admission.check_capacity()
        await run_in_threadpool(probe, path, timeout_s=services.settings.ffprobe_timeout_s)
        # Before the S3 upload, so a duplicate costs no upload. Two identical uploads
        # racing past this check both become jobs: wasted compute, nothing incorrect.
        if (job := await admission.duplicate(fingerprint)) is not None:
            return job, _Outcome.DEDUPLICATED
        key = services.store.key_for(admission.api_key.id, uuid.uuid4())
        await run_in_threadpool(services.store.upload_file, key, path)
    created = False
    try:
        job, outcome = await admission.insert(
            audio_key=key,
            audio_bytes=size,
            audio_sha256=sha256,
            fingerprint=fingerprint,
            source="direct",
        )
        created = outcome is _Outcome.CREATED
    finally:
        if not created:  # replay, idempotency conflict or error: nothing references it
            await run_in_threadpool(_discard_audio, services.store, key)
    return job, outcome


async def _spool(body: AsyncIterator[bytes], sink: IO[bytes]) -> tuple[int, str]:
    """Write ``body`` to ``sink``; returns (size, sha256 hex)."""
    digest = hashlib.sha256()
    size = 0
    try:
        async for chunk in body:
            sink.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    finally:
        UPLOAD_BYTES.inc(size)
    sink.flush()
    return size, digest.hexdigest()


async def _body_chunks(request: Request, limit: int) -> AsyncGenerator[bytes]:
    """The request body, aborting with 413 once it exceeds ``limit`` bytes.

    A declared Content-Length over the limit is refused before reading anything; the
    running count catches bodies without one (chunked) or with one that lies."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise PayloadTooLargeError(f"body exceeds the {limit} byte limit")
    received = 0
    async for chunk in request.stream():
        received += len(chunk)
        if received > limit:
            raise PayloadTooLargeError(f"body exceeds the {limit} byte limit")
        yield chunk


async def _file_chunks(file: UploadFile) -> AsyncIterator[bytes]:
    # ponytail: copies each file out of Starlette's spool so ffprobe gets a path (2x disk
    # writes, bounded by max_direct_upload_bytes); parse straight into named temp files
    # if that ever shows up in a profile.
    while chunk := await file.read(_FILE_READ_CHUNK):
        yield chunk


async def _read_json(request: Request) -> CreateTranscriptionRequest:
    body = b"".join([chunk async for chunk in _body_chunks(request, _JSON_BODY_LIMIT)])
    try:
        return CreateTranscriptionRequest.model_validate_json(body)
    except ValidationError as exc:
        raise RequestValidationError(
            [{**error, "loc": ("body", *error["loc"])} for error in exc.errors()]
        ) from exc


def _media_type(request: Request) -> str:
    return request.headers.get("content-type", "").partition(";")[0].strip().lower()


def _discard_audio(store: ObjectStore, key: str) -> None:
    """Best effort: the bucket's lifecycle rule expires anything left behind."""
    try:
        store.delete(key)
    except (DependencyUnavailableError, ClientError):
        log.warning("could not delete audio object", extra={"audio_key": key}, exc_info=True)


# --- read ------------------------------------------------------------------------------
@router.get(
    "",
    response_model=TranscriptionList,
    responses=problem_responses(401, 422, 429),
)
def list_transcriptions(
    services: ServicesDep,
    api_key: AuthorizedKey,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    before: Annotated[
        datetime | None, Query(description="Cursor: next_cursor of the previous page.")
    ] = None,
    status: JobStatus | None = None,
) -> TranscriptionList:
    """The caller's jobs, newest first, without transcripts."""
    if before is not None and before.tzinfo is None:
        before = before.replace(tzinfo=UTC)  # never let the DB session's timezone decide
    # One extra row tells whether another page exists without a second query.
    jobs = services.repo.list_jobs(api_key.id, limit=limit + 1, before=before, status=status)
    page = jobs[:limit]
    return TranscriptionList(
        data=[TranscriptionSummary.from_job(job) for job in page],
        next_cursor=page[-1].created_at if len(jobs) > limit else None,
    )


@router.get(
    "/{job_id}",
    response_model=TranscriptionJob,
    responses=problem_responses(401, 404, 422, 429),
)
def get_transcription(
    job_id: uuid.UUID,
    services: ServicesDep,
    api_key: AuthorizedKey,
    include_segments: bool = True,
) -> TranscriptionJob:
    """Status, progress, error and, once succeeded, the transcript. Another key's job
    is a 404, not a 403, so ids can't be probed for existence."""
    job = _owned_job(services, api_key, job_id)
    return TranscriptionJob.from_job(job, include_segments=include_segments)


@router.get(
    "/{job_id}/subtitles",
    response_class=Response,
    responses={
        200: {
            "description": "Captions file (attachment)",
            "content": {
                media_type: {"schema": {"type": "string"}}
                for _, media_type in _SUBTITLE_FORMATS.values()
            },
        },
        **problem_responses(401, 404, 409, 422, 429),
    },
)
def get_subtitles(
    job_id: uuid.UUID,
    services: ServicesDep,
    api_key: AuthorizedKey,
    format_: Annotated[Literal["srt", "vtt"], Query(alias="format")] = "srt",
) -> Response:
    """SubRip or WebVTT captions; 409 ``not_ready`` until the job has succeeded."""
    job = _owned_job(services, api_key, job_id)
    if job.status is not JobStatus.SUCCEEDED or job.result is None:
        raise NotReadyError()
    render, media_type = _SUBTITLE_FORMATS[format_]
    return Response(
        render(job.result),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="transcript-{job.id}.{format_}"'},
    )


# --- delete ----------------------------------------------------------------------------
@router.delete(
    "/{job_id}",
    status_code=204,
    response_class=Response,
    responses=problem_responses(401, 404, 422, 429),
)
def delete_transcription(
    job_id: uuid.UUID, services: ServicesDep, api_key: AuthorizedKey
) -> Response:
    """Delete the job, its transcript and its audio now, whatever its status.

    Deleting a processing job is safe: every worker write is fenced on the job row, so
    once the row is gone the worker's next checkpoint, heartbeat or completion fails
    with LeaseLostError and it stops without writing; its redelivered message then
    claims MISSING and is acked. The audio object is removed best effort (the bucket's
    lifecycle rule expires it otherwise)."""
    job = services.repo.delete_job(job_id, api_key.id)
    if job is None:
        raise NotFoundError("transcription not found")
    _discard_audio(services.store, job.audio_key)
    log.info("job deleted", extra={"job_id": str(job.id), "status": job.status.value})
    return Response(status_code=204)


def _owned_job(services: Services, api_key: ApiKeyRecord, job_id: uuid.UUID) -> JobRecord:
    job = services.repo.get_job(job_id, api_key_id=api_key.id)
    if job is None:
        raise NotFoundError("transcription not found")
    return job
