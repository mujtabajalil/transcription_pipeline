"""Primary/fallback engine routing behind a circuit breaker."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Literal

from transcription.asr.base import ASREngine, Audio
from transcription.domain import ChunkTranscript
from transcription.errors import EngineError, EngineUnavailableError
from transcription.metrics import BREAKER_OPEN, ENGINE_FALLBACKS

log = logging.getLogger(__name__)

BreakerState = Literal["closed", "open", "half_open"]


class CircuitBreaker:
    """Stops calling a backend after ``failure_threshold`` consecutive failures.

    While open, calls are refused so a dead backend costs no timeouts. After
    ``reset_s`` it turns half-open and admits exactly one trial call: success closes
    it, failure re-opens it for another ``reset_s``. A granted trial that never
    reports back (e.g. the caller crashed) is retried after another ``reset_s``, so the
    breaker can't wedge half-open.
    """

    def __init__(
        self,
        *,
        failure_threshold: int,
        reset_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self._failure_threshold = failure_threshold
        self._reset_s = reset_s
        self._clock = clock
        self._lock = threading.Lock()
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> BreakerState:
        with self._lock:
            return self._state()

    def allow(self) -> bool:
        """Whether the caller may try the backend now. Half-open grants one trial."""
        with self._lock:
            state = self._state()
            if state == "half_open":
                # Re-arm: concurrent callers see "open" until the trial reports back.
                self._opened_at = self._clock()
                return True
            return state == "closed"

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._opened_at is not None or self._failures >= self._failure_threshold:
                self._opened_at = self._clock()

    def _state(self) -> BreakerState:
        if self._opened_at is None:
            return "closed"
        if self._clock() - self._opened_at < self._reset_s:
            return "open"
        return "half_open"


class FallbackEngine:
    """Serves each chunk from ``primary`` unless it is unavailable, then from ``fallback``.

    Only ``EngineUnavailableError`` (outage, throttling, timeouts) fails over: a plain
    ``EngineError`` means the primary answered and rejected this chunk, which a second
    engine would not fix, so it propagates to the pipeline's retry logic. The returned
    transcript's ``engine`` names whichever engine actually served the chunk.
    """

    def __init__(
        self,
        primary: ASREngine,
        fallback: ASREngine,
        *,
        failure_threshold: int,
        reset_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._breaker = CircuitBreaker(
            failure_threshold=failure_threshold, reset_s=reset_s, clock=clock
        )
        self._publish_state()

    @property
    def name(self) -> str:
        return self._primary.name

    def transcribe(
        self,
        audio: Audio,
        *,
        language: str | None,
        prompt: str | None,
        word_timestamps: bool = False,
    ) -> ChunkTranscript:
        if self._breaker.allow():
            try:
                transcript = self._primary.transcribe(
                    audio, language=language, prompt=prompt, word_timestamps=word_timestamps
                )
            except EngineUnavailableError as exc:
                state = self._record(available=False)
                log.warning(
                    "primary ASR engine unavailable, serving chunk from fallback",
                    extra={
                        "primary": self._primary.name,
                        "fallback": self._fallback.name,
                        "error": str(exc),
                        "breaker": state,
                    },
                )
            except EngineError:
                # The primary answered, so it is available; this chunk is the problem.
                self._record(available=True)
                raise
            else:
                self._record(available=True)
                return transcript
        ENGINE_FALLBACKS.labels(primary=self._primary.name, fallback=self._fallback.name).inc()
        return self._fallback.transcribe(
            audio, language=language, prompt=prompt, word_timestamps=word_timestamps
        )

    def _record(self, *, available: bool) -> BreakerState:
        if available:
            self._breaker.record_success()
        else:
            self._breaker.record_failure()
        return self._publish_state()

    def _publish_state(self) -> BreakerState:
        state = self._breaker.state
        BREAKER_OPEN.labels(engine=self._primary.name).set(0 if state == "closed" else 1)
        return state
