"""The worker loop: take one message, own its job through a Postgres lease, run the
pipeline with per-chunk checkpoints, and settle the message per the outcome table in
DESIGN.md ("Job lifecycle and failure handling").

Postgres decides who works on a job; the stream only wakes workers up. So the ack is the
last step and happens only once the message is no longer needed: the job finished,
failed for good, was handed back with a fresh message, or belongs to nobody we could
help (busy, terminal, missing). Every other exit leaves the message pending, and
XAUTOCLAIM redelivers it after the visibility timeout: that delay is the retry backoff.
"""

from __future__ import annotations

import contextvars
import hashlib
import logging
import tempfile
import threading
import time
import uuid
from pathlib import Path
from types import TracebackType

from transcription.asr.base import ASREngine
from transcription.audio.probe import probe
from transcription.db.repo import ClaimOutcome, JobRecord
from transcription.domain import Transcript
from transcription.errors import (
    DependencyUnavailableError,
    LeaseLostError,
    MaxAttemptsExceededError,
    TranscriptionError,
)
from transcription.logging import log_context
from transcription.metrics import (
    AUDIO_SECONDS,
    JOB_PROCESSING_SECONDS,
    JOB_REDELIVERIES,
    JOBS_DEAD_LETTERED,
    JOBS_FINISHED,
    JOBS_RESUMED,
    REALTIME_FACTOR,
    SPEECH_RATIO,
)
from transcription.pipeline import ProgressCallback, transcribe_file
from transcription.queue import QueueMessage
from transcription.services import Services
from transcription.webhooks import WebhookSender
from transcription.worker.background import backoff_s, start_dispatcher, start_sweeper
from transcription.worker.checkpoint import DbCheckpoint

log = logging.getLogger(__name__)

_RECEIVE_BLOCK_MS = 2_000
_MAX_LOOP_BACKOFF_S = 30.0
_MAX_ERROR_MESSAGE = 500


class ShutdownRequested(TranscriptionError):
    """Raised at a chunk boundary after ``Worker.request_shutdown()``. The job goes back
    to the queue for another worker to resume from its checkpoint."""

    code = "worker_shutdown"
    http_status = 503
    retryable = True
    default_message = "interrupted by worker shutdown"


class Worker:
    """Processes one job at a time with one engine (one model per process)."""

    def __init__(
        self,
        services: Services,
        engine: ASREngine,
        *,
        worker_id: str,
        webhook_sender: WebhookSender | None = None,
    ) -> None:
        """``worker_id`` is both the lease owner in Postgres and the stream consumer
        name, so it must be unique per live process. ``webhook_sender`` feeds the
        dispatcher started by ``start_background``; without one, none is started."""
        self._services = services
        self._settings = services.settings
        self._repo = services.repo
        self._queue = services.queue
        self._engine = engine
        self._worker_id = worker_id
        self._webhook_sender = webhook_sender
        self._shutdown = threading.Event()

    def request_shutdown(self) -> None:
        """Stop taking messages. A running job stops once its current chunk is
        checkpointed and is handed back to the queue. Safe from signal handlers."""
        self._shutdown.set()

    def start_background(self, stop: threading.Event) -> list[threading.Thread]:
        """Start the sweeper and the webhook dispatcher as daemon threads that run
        until ``stop`` is set."""
        threads = [start_sweeper(self._services, stop)]
        if self._webhook_sender is not None:
            threads.append(
                start_dispatcher(
                    self._repo,
                    self._webhook_sender,
                    max_attempts=self._settings.webhook_max_attempts,
                    stop=stop,
                )
            )
        return threads

    def run_forever(self, stop: threading.Event) -> None:
        """Handle messages until ``stop`` is set or shutdown is requested. Never raises
        an Exception: an outage (or a bug outside a job) is logged and retried with
        exponential backoff, because a crash would reload the model for nothing."""
        failures = 0
        while not (stop.is_set() or self._shutdown.is_set()):
            try:
                self.run_once(block_ms=_RECEIVE_BLOCK_MS)
            except Exception as exc:
                failures += 1
                delay = backoff_s(failures, cap_s=_MAX_LOOP_BACKOFF_S)
                if isinstance(exc, DependencyUnavailableError):
                    log.warning(
                        "dependency unavailable, backing off",
                        extra={"error": exc.message, "retry_in_s": delay},
                    )
                else:
                    log.exception("worker loop failed, backing off", extra={"retry_in_s": delay})
                stop.wait(delay)
            else:
                failures = 0

    def run_once(self, *, block_ms: int) -> bool:
        """Handle at most one message: first one idle longer than the visibility
        timeout (its worker died or released the job), else a new one, waiting up to
        ``block_ms``. Returns whether a message was handled; always False once shutdown
        was requested."""
        if self._shutdown.is_set():
            return False
        message = self._next_message(block_ms)
        if message is None:
            return False
        self._handle(message)
        return True

    def _next_message(self, block_ms: int) -> QueueMessage | None:
        reclaimed = self._queue.reclaim(
            self._worker_id, min_idle_ms=self._settings.visibility_timeout_s * 1000, count=1
        )
        if reclaimed:
            JOB_REDELIVERIES.inc()
            log.info(
                "reclaimed idle message",
                extra={"message_id": reclaimed[0].id, "job_id": str(reclaimed[0].job_id)},
            )
            return reclaimed[0]
        return self._queue.receive(self._worker_id, block_ms=block_ms)

    def _handle(self, message: QueueMessage) -> None:
        if message.job_id is None:
            self._queue.dead_letter(message, reason="malformed message")
            return
        claim = self._repo.claim(
            message.job_id,
            worker_id=self._worker_id,
            lease_s=self._settings.visibility_timeout_s,
            max_attempts=self._settings.max_job_attempts,
        )
        job = claim.job
        if claim.outcome is ClaimOutcome.CLAIMED and job is not None:
            self._process(message, job)
        elif claim.outcome is ClaimOutcome.EXHAUSTED and job is not None:
            self._dead_letter(message, job)
        else:
            log.info(
                "message skipped",
                extra={"job_id": str(message.job_id), "claim": claim.outcome.value},
            )
            self._queue.ack(message.id)

    def _dead_letter(self, message: QueueMessage, job: JobRecord) -> None:
        code = MaxAttemptsExceededError.code
        failed = self._repo.fail(
            job.id,
            worker_id=None,
            code=code,
            http_status=MaxAttemptsExceededError.http_status,
            message=job.error_message or MaxAttemptsExceededError.default_message,
        )
        self._queue.dead_letter(message, reason=code)
        JOBS_DEAD_LETTERED.inc()
        if failed is not None:
            JOBS_FINISHED.labels(status="failed", error_code=code).inc()
        log.error(
            "job exhausted its attempts",
            extra={"job_id": str(job.id), "attempts": job.attempts, "error": job.error_message},
        )

    def _process(self, message: QueueMessage, job: JobRecord) -> None:
        started = time.monotonic()
        with (
            log_context(job_id=str(job.id), attempt=job.attempts),
            _Heartbeat(self._services, job.id, self._worker_id, message.id) as heartbeat,
        ):
            if job.plan is not None:
                JOBS_RESUMED.inc()
            log.info("job claimed", extra={"chunks_done": job.chunks_done})
            try:
                self._execute(job, heartbeat.lease_lost, started)
            except LeaseLostError:
                log.warning("job lease lost, leaving the job and its message to the new owner")
            except ShutdownRequested:
                self._hand_back(message, job)
            except Exception as exc:
                log.exception("job attempt failed, leaving it for redelivery")
                self._release(job, exc)
            else:
                self._queue.ack(message.id)

    def _execute(self, job: JobRecord, lease_lost: threading.Event, started: float) -> None:
        """Run the job to a terminal state in the DB, or raise."""
        try:
            transcript = self._transcribe(job, lease_lost)
        except TranscriptionError as exc:
            if exc.retryable:
                raise
            # Bad input (or audio that no longer exists): another attempt can't help.
            self._repo.fail(
                job.id,
                worker_id=self._worker_id,
                code=exc.code,
                http_status=exc.http_status,
                message=exc.message,
            )
            JOBS_FINISHED.labels(status="failed", error_code=exc.code).inc()
            log.warning("job failed", extra={"error_code": exc.code, "error": exc.message})
            return
        self._repo.complete(job.id, worker_id=self._worker_id, transcript=transcript)
        _observe_success(transcript, time.monotonic() - started)

    def _transcribe(self, job: JobRecord, lease_lost: threading.Event) -> Transcript:
        with tempfile.TemporaryDirectory(prefix="tx-job-") as tmp:
            workdir = Path(tmp)
            audio = workdir / "audio"
            size = self._services.store.download_file(job.audio_key, audio)
            with audio.open("rb") as file:
                sha256 = hashlib.file_digest(file, "sha256").hexdigest()
            info = probe(audio, timeout_s=self._settings.ffprobe_timeout_s)
            self._repo.record_audio(
                job.id, worker_id=self._worker_id, info=info, audio_sha256=sha256, audio_bytes=size
            )
            return transcribe_file(
                audio,
                self._engine,
                job.options,
                self._settings.pipeline_config(),
                checkpoint=DbCheckpoint(self._repo, job.id, self._worker_id),
                workdir=workdir,
                on_progress=self._cancellation_check(lease_lost),
            )

    def _cancellation_check(self, lease_lost: threading.Event) -> ProgressCallback:
        """Runs at every chunk boundary, i.e. right after a checkpoint, so stopping
        there loses no finished work."""

        def check(_done: int, _total: int) -> None:
            if lease_lost.is_set():
                raise LeaseLostError("lease lost (heartbeat)")
            if self._shutdown.is_set():
                raise ShutdownRequested

        return check

    def _hand_back(self, message: QueueMessage, job: JobRecord) -> None:
        """Graceful shutdown: requeue now rather than after the visibility timeout."""
        if not self._repo.release(
            job.id, worker_id=self._worker_id, error_message=ShutdownRequested.default_message
        ):
            log.warning("job lease lost before hand-back")
            return
        # Enqueue before ack: a crash in between leaves a harmless duplicate, not a job
        # that waits for the sweeper.
        self._queue.enqueue(job.id)
        self._queue.ack(message.id)
        log.info("job handed back for another worker to resume")

    def _release(self, job: JobRecord, exc: Exception) -> None:
        # error_message is shown to the client (and becomes the final error once the
        # attempts run out); arbitrary exception text may carry internals.
        message = exc.message if isinstance(exc, TranscriptionError) else "internal error"
        if not self._repo.release(
            job.id, worker_id=self._worker_id, error_message=message[:_MAX_ERROR_MESSAGE]
        ):
            log.warning("job lease lost before release")


def _observe_success(transcript: Transcript, elapsed_s: float) -> None:
    stats = transcript.stats
    JOBS_FINISHED.labels(status="succeeded", error_code="").inc()
    JOB_PROCESSING_SECONDS.observe(elapsed_s)
    AUDIO_SECONDS.inc(transcript.duration_s)
    if stats.realtime_factor is not None:
        REALTIME_FACTOR.observe(stats.realtime_factor)
    if stats.audio_seconds > 0:
        SPEECH_RATIO.observe(stats.speech_seconds / stats.audio_seconds)
    log.info(
        "job succeeded",
        extra={
            "audio_s": round(transcript.duration_s, 3),
            "segments": len(transcript.segments),
            "elapsed_s": round(elapsed_s, 3),
        },
    )


class _Heartbeat:
    """Background thread that keeps a running job's DB lease and stream message alive.

    Without it, any job longer than the visibility timeout would be reclaimed and run
    twice. It sets ``lease_lost`` (and stops) as soon as the DB says another worker owns
    the job; transient failures are logged and retried at the next beat.
    """

    def __init__(
        self, services: Services, job_id: uuid.UUID, worker_id: str, message_id: str
    ) -> None:
        self.lease_lost = threading.Event()
        self._services = services
        self._job_id = job_id
        self._worker_id = worker_id
        self._message_id = message_id
        self._stop = threading.Event()
        # Threads don't inherit contextvars; copying keeps job_id on heartbeat logs.
        context = contextvars.copy_context()
        self._thread = threading.Thread(
            target=context.run, args=(self._run,), name=f"heartbeat-{job_id}", daemon=True
        )

    def __enter__(self) -> _Heartbeat:
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        settings = self._services.settings
        while not self._stop.wait(settings.heartbeat_interval_s):
            try:
                if not self._services.repo.heartbeat(
                    self._job_id, worker_id=self._worker_id, lease_s=settings.visibility_timeout_s
                ):
                    self.lease_lost.set()
                    return
                self._services.queue.touch(self._worker_id, self._message_id)
            except Exception as exc:
                log.warning("heartbeat failed", extra={"error": str(exc)})
