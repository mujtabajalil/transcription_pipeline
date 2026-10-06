"""Local Whisper via faster-whisper (CTranslate2)."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from transcription.asr.base import Audio
from transcription.asr.languages import normalize_language
from transcription.domain import ChunkTranscript, Segment, Word
from transcription.errors import EngineError

log = logging.getLogger(__name__)

_TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
"""Re-decode hotter only when a pass looks like a loop or gibberish (see thresholds)."""


class FasterWhisperEngine:
    """Whisper on CTranslate2, loaded once per worker process.

    Decoding is tuned for independent ~30 s chunks: no conditioning on previous text (a
    repetition loop can't leak into the next window), no VAD (the pipeline already ran
    it over the whole file), and temperature fallback driven by the same thresholds the
    post-processing guards use.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        compute_type: str = "int8",
        beam_size: int = 5,
        cpu_threads: int = 0,
        download_root: str | None = None,
        no_speech_threshold: float = 0.6,
        logprob_threshold: float = -1.0,
        compression_ratio_threshold: float = 2.4,
        model: Any = None,
    ) -> None:
        self._name = f"faster_whisper:{model_name}"
        self._beam_size = beam_size
        self._no_speech_threshold = no_speech_threshold
        self._logprob_threshold = logprob_threshold
        self._compression_ratio_threshold = compression_ratio_threshold
        if model is None:
            model = _load_model(
                model_name,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                download_root=download_root,
            )
        self._model = model

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
        if audio.ndim != 1 or audio.dtype != np.float32:
            raise ValueError(f"expected 1-D float32 audio, got {audio.ndim}-D {audio.dtype}")
        pinned = normalize_language(language)
        try:
            segments, info = self._model.transcribe(
                audio,
                language=pinned,
                initial_prompt=prompt,
                beam_size=self._beam_size,
                temperature=_TEMPERATURES,
                compression_ratio_threshold=self._compression_ratio_threshold,
                log_prob_threshold=self._logprob_threshold,
                no_speech_threshold=self._no_speech_threshold,
                condition_on_previous_text=False,
                vad_filter=False,
                word_timestamps=word_timestamps,
                without_timestamps=False,
            )
            # The segments are a lazy generator: decoding (and its failures) happens here.
            decoded = list(segments)
        except (RuntimeError, ValueError, MemoryError) as exc:
            raise EngineError(f"faster-whisper decoding failed: {exc}") from exc
        return ChunkTranscript(
            segments=[_to_segment(s) for s in decoded],
            language=info.language,
            language_probability=None if pinned else info.language_probability,
            engine=self._name,
        )


def _load_model(model_name: str, **options: Any) -> Any:
    # Imported here so `import transcription.asr` stays cheap for the API process.
    from faster_whisper import WhisperModel

    started = time.perf_counter()
    model = WhisperModel(model_name, **options)
    log.info(
        "whisper model loaded",
        extra={
            "model": model_name,
            "load_seconds": round(time.perf_counter() - started, 2),
            **options,
        },
    )
    return model


def _to_segment(segment: Any) -> Segment:
    words = None
    if segment.words is not None:
        words = [
            Word(start=w.start, end=w.end, text=w.word, probability=w.probability)
            for w in segment.words
        ]
    return Segment(
        start=segment.start,
        end=segment.end,
        text=segment.text,
        avg_logprob=segment.avg_logprob,
        no_speech_prob=segment.no_speech_prob,
        compression_ratio=segment.compression_ratio,
        words=words,
    )
