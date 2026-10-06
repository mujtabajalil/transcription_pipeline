"""Test doubles shared across suites."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from transcription.asr.base import Audio
from transcription.domain import SAMPLE_RATE, ChunkTranscript, Segment
from transcription.errors import EngineError


class FakeEngine:
    """Deterministic ASREngine. Each call returns one segment spanning the chunk with
    text ``"<prefix> <n>"`` (n = 0-based call count), language ``language``.

    ``fail_times``: raise ``error`` on the first N calls (retry tests).
    ``on_call``: hook run before each call (e.g. to simulate a crash mid-job).
    """

    def __init__(
        self,
        *,
        prefix: str = "chunk",
        language: str = "en",
        fail_times: int = 0,
        error: Exception | None = None,
        on_call: Callable[[int], None] | None = None,
        name: str = "fake",
    ) -> None:
        self._name = name
        self.prefix = prefix
        self.language = language
        self.fail_times = fail_times
        self.error = error or EngineError("boom")
        self.on_call = on_call
        self.calls: list[dict[str, object]] = []

    @property
    def name(self) -> str:
        return self._name

    def transcribe(
        self,
        audio: Audio,
        *,
        language: str | None,
        prompt: str | None,
        word_timestamps: bool = False,
    ) -> ChunkTranscript:
        n = len(self.calls)
        self.calls.append(
            {"samples": len(audio), "language": language, "prompt": prompt, "dtype": audio.dtype}
        )
        if self.on_call:
            self.on_call(n)
        if n < self.fail_times:
            raise self.error
        assert audio.dtype == np.float32
        dur = len(audio) / SAMPLE_RATE
        return ChunkTranscript(
            segments=[
                Segment(
                    start=0.0,
                    end=dur,
                    text=f"{self.prefix} {n}",
                    avg_logprob=-0.1,
                    no_speech_prob=0.01,
                    compression_ratio=1.2,
                )
            ],
            language=language or self.language,
            language_probability=None if language else 0.99,
            engine=self._name,
        )
