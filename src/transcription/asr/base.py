"""The one seam every speech-to-text backend plugs into.

The pipeline owns long-audio handling (VAD, chunking, retries, checkpoints, timestamp
mapping); an engine only ever sees one chunk of at most ~30 s. That keeps engines
swappable (local Whisper, hosted API, fallback chain) without touching the pipeline.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

from transcription.domain import ChunkTranscript

Audio = npt.NDArray[np.float32]
"""16 kHz mono float32 in [-1, 1]."""


class ASREngine(Protocol):
    @property
    def name(self) -> str:
        """Stable identifier recorded per chunk, e.g. ``faster_whisper:small``."""
        ...

    def transcribe(
        self,
        audio: Audio,
        *,
        language: str | None,
        prompt: str | None,
        word_timestamps: bool = False,
    ) -> ChunkTranscript:
        """Transcribe one chunk. Segment times are relative to ``audio[0]``.

        ``language=None`` means detect; the result's ``language`` is then what was
        detected. Implementations must be safe to call repeatedly from one thread;
        the worker never calls one engine instance concurrently.

        Raises:
            EngineUnavailableError: backend unreachable/overloaded; caller may fall back.
            EngineError: any other engine failure worth retrying.
        """
        ...
