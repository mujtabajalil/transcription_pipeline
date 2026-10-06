"""Error taxonomy.

Every failure carries a stable machine ``code``, the HTTP status it maps to, and whether
retrying can help. The worker uses ``retryable`` to decide between failing a job now
(bad input) and letting the queue redeliver it (transient infra/engine trouble); the API
renders any ``TranscriptionError`` as RFC 9457 problem+json.
"""

from __future__ import annotations


class TranscriptionError(Exception):
    code: str = "internal_error"
    http_status: int = 500
    retryable: bool = False
    default_message: str = "internal error"

    def __init__(self, message: str | None = None, *, retry_after_s: int | None = None) -> None:
        super().__init__(message or self.default_message)
        self.message = message or self.default_message
        self.retry_after_s = retry_after_s


# --- input: permanent, never retried ---------------------------------------------------
class InputError(TranscriptionError):
    http_status = 422
    code = "invalid_input"


class UnsupportedMediaError(InputError):
    code = "unsupported_media_type"
    http_status = 415
    default_message = "unsupported or corrupt file"


class NoAudioStreamError(InputError):
    code = "no_audio_stream"
    http_status = 422
    default_message = "file has no audio stream"


class UndecodableAudioError(InputError):
    code = "undecodable_audio"
    http_status = 422
    default_message = "audio could not be decoded"


class PayloadTooLargeError(InputError):
    code = "payload_too_large"
    http_status = 413
    default_message = "file too large"


class AudioTooLongError(InputError):
    code = "audio_too_long"
    http_status = 413
    default_message = "audio exceeds the maximum duration"


# --- ASR engines -------------------------------------------------------------------------
class EngineError(TranscriptionError):
    """Engine returned garbage or failed in a way that may succeed on retry."""

    code = "asr_engine_error"
    http_status = 502
    retryable = True
    default_message = "speech recognition failed"


class EngineUnavailableError(EngineError):
    """Backend unreachable, timing out, rate limiting or 5xx-ing. Triggers fallback."""

    code = "asr_engine_unavailable"
    http_status = 503
    default_message = "speech recognition backend unavailable"


# --- worker ----------------------------------------------------------------------------
class LeaseLostError(TranscriptionError):
    """Another worker took the job over (our lease expired). Stop without writing."""

    code = "lease_lost"
    retryable = True
    default_message = "job lease lost"


class MaxAttemptsExceededError(TranscriptionError):
    code = "max_attempts_exceeded"
    http_status = 500
    default_message = "job failed repeatedly and was moved to the dead-letter queue"


# --- API -------------------------------------------------------------------------------
class UnauthorizedError(TranscriptionError):
    code = "unauthorized"
    http_status = 401
    default_message = "missing or invalid API key"


class NotFoundError(TranscriptionError):
    code = "not_found"
    http_status = 404
    default_message = "not found"


class ConflictError(TranscriptionError):
    code = "conflict"
    http_status = 409
    default_message = "conflict"


class IdempotencyMismatchError(ConflictError):
    code = "idempotency_key_reused"
    default_message = "Idempotency-Key was already used with a different request"


class NotReadyError(ConflictError):
    code = "not_ready"
    default_message = "transcription has not succeeded yet"


class RateLimitedError(TranscriptionError):
    code = "rate_limited"
    http_status = 429
    default_message = "rate limit exceeded"


class QueueFullError(TranscriptionError):
    code = "queue_full"
    http_status = 429
    default_message = "too many pending jobs, retry later"


class BadRequestError(TranscriptionError):
    code = "bad_request"
    http_status = 400
    default_message = "bad request"


class DependencyUnavailableError(TranscriptionError):
    code = "dependency_unavailable"
    http_status = 503
    retryable = True
    default_message = "a backing service is unavailable"
