"""Request and response bodies of the public API.

Responses are views built from ``JobRecord``s, never the records themselves, so
internal columns (lease, worker id, fingerprints, storage keys) can't leak by accident.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, Field

from transcription.db.repo import JobRecord
from transcription.domain import JobStatus, Segment, TranscriptionOptions, TranscriptStats
from transcription.storage import PresignedUpload

TRANSCRIPTIONS_PATH = "/v1/transcriptions"


# --- requests --------------------------------------------------------------------------
class TranscriptionParams(TranscriptionOptions):
    """Everything a create request carries besides the audio: transcription options
    plus the webhook target, which is not part of the request fingerprint."""

    webhook_url: str | None = Field(
        default=None,
        max_length=2048,
        description="https URL that receives a signed event when the job finishes.",
    )

    def options(self) -> TranscriptionOptions:
        fields = set(TranscriptionOptions.model_fields)
        return TranscriptionOptions.model_validate(self.model_dump(include=fields))


class CreateTranscriptionRequest(TranscriptionParams):
    upload_id: uuid.UUID = Field(description="From POST /v1/uploads, after the S3 upload.")


class UploadRequest(BaseModel):
    size_bytes: int = Field(gt=0, description="Exact size of the file to upload.")
    content_type: str | None = Field(
        default=None,
        max_length=255,
        description="Informational only: the format is identified from the bytes.",
    )


# --- responses -------------------------------------------------------------------------
class UploadResponse(PresignedUpload):
    """POST ``fields`` then the ``file`` part as multipart/form-data to ``url``. S3
    rejects anything larger than ``max_bytes`` (the declared size)."""

    upload_id: uuid.UUID


class Progress(BaseModel):
    chunks_done: int
    chunks_total: int
    percent: float


class AudioSummary(BaseModel):
    container: str
    codec: str | None
    channels: int
    duration_s: float | None = Field(description="Decoded duration; null until decoded.")


class JobError(BaseModel):
    code: str
    status: int | None
    message: str | None


class WebhookState(BaseModel):
    url: str
    status: str | None = Field(description="pending, sending, delivered or failed.")
    attempts: int


class Links(BaseModel):
    self_: str = Field(serialization_alias="self")
    subtitles: str | None = Field(description="Present once the job has succeeded.")


class TranscriptResult(BaseModel):
    text: str
    segments: list[Segment] | None = Field(
        description="Absolute-time segments; null when include_segments=false."
    )
    stats: TranscriptStats


class TranscriptionSummary(BaseModel):
    id: uuid.UUID
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    options: TranscriptionOptions
    progress: Progress | None = Field(description="Null until the chunk plan exists.")
    audio: AudioSummary | None = Field(description="Null until the worker probed the file.")
    language: str | None
    language_probability: float | None
    error: JobError | None
    webhook: WebhookState | None
    links: Links

    @classmethod
    def from_job(cls, job: JobRecord) -> Self:
        return cls(
            id=job.id,
            status=job.status,
            created_at=job.created_at,
            updated_at=job.updated_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            options=job.options,
            progress=_progress(job),
            audio=_audio(job),
            language=job.language,
            language_probability=job.language_probability,
            error=_error(job),
            webhook=_webhook(job),
            links=Links(
                self_=job_path(job.id),
                subtitles=f"{job_path(job.id)}/subtitles"
                if job.status is JobStatus.SUCCEEDED
                else None,
            ),
        )


class TranscriptionJob(TranscriptionSummary):
    result: TranscriptResult | None = Field(
        default=None, description="Only once the job has succeeded."
    )

    @classmethod
    def from_job(cls, job: JobRecord, *, include_segments: bool = True) -> Self:
        view = super().from_job(job)
        if job.status is JobStatus.SUCCEEDED and job.result is not None:
            view.result = TranscriptResult(
                text=job.result.text,
                segments=job.result.segments if include_segments else None,
                stats=job.result.stats,
            )
        return view


class TranscriptionList(BaseModel):
    data: list[TranscriptionSummary]
    next_cursor: datetime | None = Field(
        description="Pass as ``before`` to get the next page; null on the last page."
    )


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadinessResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    checks: dict[str, Literal["ok", "error"]]


def job_path(job_id: uuid.UUID) -> str:
    return f"{TRANSCRIPTIONS_PATH}/{job_id}"


def _progress(job: JobRecord) -> Progress | None:
    if job.chunks_total is None:
        return None
    # A plan with no chunks (no speech at all) has nothing left to do.
    percent = 100.0 if job.chunks_total == 0 else 100 * job.chunks_done / job.chunks_total
    return Progress(
        chunks_done=job.chunks_done, chunks_total=job.chunks_total, percent=round(percent, 1)
    )


def _audio(job: JobRecord) -> AudioSummary | None:
    if job.audio_info is None:
        return None
    return AudioSummary(
        container=job.audio_info.container,
        codec=job.audio_info.codec,
        channels=job.audio_info.channels,
        duration_s=job.duration_s,
    )


def _error(job: JobRecord) -> JobError | None:
    # A released job keeps error_message as a hint but has no code: it isn't failed.
    if job.error_code is None:
        return None
    return JobError(code=job.error_code, status=job.error_status, message=job.error_message)


def _webhook(job: JobRecord) -> WebhookState | None:
    if job.webhook_url is None:
        return None
    return WebhookState(
        url=job.webhook_url, status=job.webhook_status, attempts=job.webhook_attempts
    )
