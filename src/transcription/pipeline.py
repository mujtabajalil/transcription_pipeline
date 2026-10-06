"""Audio file → Transcript. Pure orchestration: no DB, queue or S3 in here.

    probe → decode (per channel) → VAD → plan chunks → for each pending chunk:
    engine.transcribe (retried) → post-process → checkpoint → assemble

Infra-specific durability is injected through ``Checkpoint``: the worker passes a
Postgres-backed one so a crashed job resumes; the CLI passes an in-memory one.
"""

from __future__ import annotations

import logging
import random
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from transcription.asr.base import ASREngine, Audio
from transcription.asr.languages import normalize_language
from transcription.asr.postprocess import postprocess
from transcription.audio.chunking import plan_chunks
from transcription.audio.decode import decode_to_pcm, open_pcm, to_float32
from transcription.audio.probe import probe
from transcription.audio.vad import detect_speech
from transcription.domain import (
    SAMPLE_RATE,
    AudioInfo,
    ChunkPlan,
    ChunkResult,
    ChunkTranscript,
    PipelineConfig,
    PlannedChunk,
    Transcript,
    TranscriptionOptions,
    TranscriptStats,
)
from transcription.errors import TranscriptionError
from transcription.metrics import CHUNK_SECONDS, CHUNKS, FORCED_CUTS

log = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]
"""(chunks_done, chunks_total)."""

_Pcm = np.memmap[Any, np.dtype[np.int16]]


class Checkpoint(Protocol):
    """Durable per-job state that lets a restarted worker skip finished work."""

    def load_plan(self) -> ChunkPlan | None: ...

    def save_plan(self, plan: ChunkPlan) -> None: ...

    def load_results(self) -> dict[int, ChunkResult]:
        """Finished chunk results keyed by chunk index."""
        ...

    def save_result(self, result: ChunkResult) -> None:
        """Persist one finished chunk. Must be durable before returning."""
        ...

    def load_language(self) -> tuple[str, float | None] | None: ...

    def save_language(self, language: str, probability: float | None) -> None: ...


class InMemoryCheckpoint:
    """Checkpoint for the CLI and tests.

    Lives only as long as the object, so it saves work only when the same instance is
    passed to a repeated ``transcribe_file`` call (e.g. after a cancelled run).
    """

    def __init__(self) -> None:
        self.plan: ChunkPlan | None = None
        self.results: dict[int, ChunkResult] = {}
        self.language: tuple[str, float | None] | None = None

    def load_plan(self) -> ChunkPlan | None:
        return self.plan

    def save_plan(self, plan: ChunkPlan) -> None:
        self.plan = plan

    def load_results(self) -> dict[int, ChunkResult]:
        return dict(self.results)

    def save_result(self, result: ChunkResult) -> None:
        self.results[result.index] = result

    def load_language(self) -> tuple[str, float | None] | None:
        return self.language

    def save_language(self, language: str, probability: float | None) -> None:
        self.language = (language, probability)


def transcribe_file(
    path: str | Path,
    engine: ASREngine,
    options: TranscriptionOptions,
    config: PipelineConfig | None = None,
    *,
    checkpoint: Checkpoint | None = None,
    on_progress: ProgressCallback | None = None,
    workdir: Path | None = None,
) -> Transcript:
    """Transcribe an audio file of any supported container.

    Decoded PCM goes to ``workdir`` (left for the caller to clean up) or to a temporary
    directory removed on return. Chunks already in ``checkpoint`` are not transcribed
    again, provided its plan matches the decoded audio. ``on_progress`` is called once
    before any work and after every checkpointed chunk; an exception it raises cancels
    the run and propagates (finished chunks stay checkpointed).

    Raises InputError subclasses for bad input (never worth retrying) and lets
    EngineError propagate once per-chunk retries are exhausted.
    """
    config = config or PipelineConfig()
    checkpoint = checkpoint if checkpoint is not None else InMemoryCheckpoint()
    started = time.monotonic()
    info = probe(path, timeout_s=config.ffprobe_timeout_s)
    channels: list[int | None] = (
        list(range(info.channels)) if options.split_channels and info.channels > 1 else [None]
    )
    scratch = tempfile.TemporaryDirectory() if workdir is None else nullcontext(str(workdir))
    with scratch as directory:
        pcm = {ch: _decode(path, Path(directory), ch, config) for ch in channels}
        plan, results = _load_or_plan(checkpoint, pcm, config)
        resumed = len(results)
        language, probability = _known_language(options, checkpoint)
        if on_progress:
            on_progress(len(results), plan.total)
        for chunk in plan.chunks:
            if chunk.index in results:
                continue
            audio = to_float32(pcm[chunk.channel][chunk.start : chunk.end])
            raw = _transcribe_chunk(engine, audio, chunk, language, options, config)
            if language is None and (detected := normalize_language(raw.language)):
                language, probability = detected, raw.language_probability
                checkpoint.save_language(language, probability)
                log.info(
                    "language detected", extra={"language": language, "probability": probability}
                )
            results[chunk.index] = _checkpoint_chunk(raw, chunk, config, checkpoint)
            if on_progress:
                on_progress(len(results), plan.total)
    transcript = _assemble(info, plan, results, language, probability, started)
    log.info(
        "transcription finished",
        extra={
            "chunks": plan.total,
            "resumed_chunks": resumed,
            "language": language,
            "audio_s": round(transcript.duration_s, 3),
            "speech_s": round(transcript.stats.speech_seconds, 3),
            "processing_s": transcript.stats.processing_seconds,
            "realtime_factor": transcript.stats.realtime_factor,
        },
    )
    return transcript


def _decode(path: str | Path, directory: Path, channel: int | None, config: PipelineConfig) -> _Pcm:
    out = directory / f"decoded-{'mix' if channel is None else f'ch{channel}'}.pcm"
    decode_to_pcm(
        path,
        out,
        channel=channel,
        max_seconds=config.max_audio_seconds,
        timeout_s=config.ffmpeg_timeout_s,
    )
    return open_pcm(out)


def _load_or_plan(
    checkpoint: Checkpoint, pcm: dict[int | None, _Pcm], config: PipelineConfig
) -> tuple[ChunkPlan, dict[int, ChunkResult]]:
    """The checkpointed plan and its finished results, or a fresh plan with none.

    A plan is reused only if it describes this exact decode: chunk boundaries from a
    different decode (e.g. another ffmpeg version) would stitch results onto the wrong
    audio.
    """
    audio_samples = max(len(audio) for audio in pcm.values())
    channels = list(pcm)
    plan = checkpoint.load_plan()
    if plan is not None and (plan.audio_samples, plan.channels) == (audio_samples, channels):
        return plan, _matching_results(plan, checkpoint.load_results())
    if plan is not None:
        log.warning(
            "checkpointed chunk plan does not match the decoded audio, replanning",
            extra={
                "planned_samples": plan.audio_samples,
                "decoded_samples": audio_samples,
                "planned_channels": plan.channels,
                "decoded_channels": channels,
            },
        )
    plan = _plan(pcm, audio_samples, config)
    checkpoint.save_plan(plan)
    FORCED_CUTS.inc(plan.forced_cuts)
    return plan, {}


def _matching_results(plan: ChunkPlan, loaded: dict[int, ChunkResult]) -> dict[int, ChunkResult]:
    # A store may still hold rows from an earlier plan (replanned, then crashed); those
    # cover different audio and must be transcribed again.
    chunks = {chunk.index: chunk for chunk in plan.chunks}
    return {
        index: result
        for index, result in loaded.items()
        if (chunk := chunks.get(index)) is not None
        and (result.channel, result.start_s, result.end_s)
        == (chunk.channel, chunk.start_s, chunk.end_s)
    }


def _plan(pcm: dict[int | None, _Pcm], audio_samples: int, config: PipelineConfig) -> ChunkPlan:
    chunks: list[PlannedChunk] = []
    forced_cuts = speech_samples = 0
    for channel, audio in pcm.items():
        regions = detect_speech(
            audio,
            block_samples=config.vad_block_s * SAMPLE_RATE,
            threshold=config.vad_threshold,
            min_silence_ms=config.vad_min_silence_ms,
            speech_pad_ms=config.vad_speech_pad_ms,
        )
        planned, cuts = plan_chunks(
            regions,
            audio,
            max_chunk=int(config.chunk_max_s * SAMPLE_RATE),
            cut_window=int(config.forced_cut_window_s * SAMPLE_RATE),
            cut_frame=config.forced_cut_frame_ms * SAMPLE_RATE // 1000,
            max_gap=int(config.max_pack_gap_s * SAMPLE_RATE),
            channel=channel,
            start_index=len(chunks),
        )
        chunks += planned
        forced_cuts += cuts
        speech_samples += sum(end - start for start, end in regions)
    return ChunkPlan(
        chunks=chunks,
        audio_samples=audio_samples,
        speech_samples=speech_samples,
        forced_cuts=forced_cuts,
        channels=list(pcm),
    )


def _known_language(
    options: TranscriptionOptions, checkpoint: Checkpoint
) -> tuple[str | None, float | None]:
    if pinned := normalize_language(options.language):
        return pinned, None
    return checkpoint.load_language() or (None, None)


def _checkpoint_chunk(
    raw: ChunkTranscript, chunk: PlannedChunk, config: PipelineConfig, checkpoint: Checkpoint
) -> ChunkResult:
    result = postprocess(raw, chunk, config)
    checkpoint.save_result(result)
    log.debug(
        "chunk transcribed",
        extra={
            "chunk": chunk.index,
            "channel": chunk.channel,
            "start_s": chunk.start_s,
            "end_s": chunk.end_s,
            "engine": result.engine,
            "segments": len(result.segments),
            "dropped": result.dropped,
        },
    )
    return result


def _transcribe_chunk(
    engine: ASREngine,
    audio: Audio,
    chunk: PlannedChunk,
    language: str | None,
    options: TranscriptionOptions,
    config: PipelineConfig,
) -> ChunkTranscript:
    """``engine.transcribe`` with in-job retries of retryable failures.

    Retrying one chunk here is far cheaper than a job redelivery, which re-downloads and
    re-decodes everything. Full jitter keeps workers that hit the same outage from
    retrying in lockstep.
    """
    attempt = 0
    while True:
        started = time.monotonic()
        try:
            transcript = engine.transcribe(
                audio,
                language=language,
                prompt=options.prompt,
                word_timestamps=options.word_timestamps,
            )
        except Exception as exc:
            retryable = isinstance(exc, TranscriptionError) and exc.retryable
            if not retryable or attempt >= config.chunk_max_retries:
                CHUNKS.labels(engine=engine.name, outcome="error").inc()
                raise
            CHUNKS.labels(engine=engine.name, outcome="retry").inc()
            delay = random.uniform(0, config.chunk_retry_base_delay_s * 2**attempt)  # noqa: S311
            log.warning(
                "chunk transcription failed, retrying",
                extra={
                    "chunk": chunk.index,
                    "attempt": attempt + 1,
                    "delay_s": round(delay, 3),
                    "error": str(exc),
                    "error_code": exc.code if isinstance(exc, TranscriptionError) else None,
                },
            )
            if delay > 0:
                time.sleep(delay)
            attempt += 1
        else:
            CHUNK_SECONDS.labels(engine=transcript.engine).observe(time.monotonic() - started)
            CHUNKS.labels(engine=transcript.engine, outcome="ok").inc()
            return transcript


def _assemble(
    info: AudioInfo,
    plan: ChunkPlan,
    results: dict[int, ChunkResult],
    language: str | None,
    probability: float | None,
    started: float,
) -> Transcript:
    ordered = [results[chunk.index] for chunk in plan.chunks]
    segments = sorted(
        (segment for result in ordered for segment in result.segments),
        key=lambda segment: (segment.start, segment.channel or 0),
    )
    dropped: Counter[str] = Counter()
    for result in ordered:
        dropped.update(result.dropped)
    audio_seconds = plan.audio_samples / SAMPLE_RATE
    processing_seconds = time.monotonic() - started
    return Transcript(
        audio=info,
        language=language,
        language_probability=probability,
        duration_s=audio_seconds,
        text=" ".join(segment.text for segment in segments),
        segments=segments,
        stats=TranscriptStats(
            audio_seconds=audio_seconds,
            speech_seconds=plan.speech_samples / SAMPLE_RATE,
            transcribed_seconds=sum(chunk.end - chunk.start for chunk in plan.chunks) / SAMPLE_RATE,
            chunks=plan.total,
            forced_cuts=plan.forced_cuts,
            dropped_segments=dict(dropped),
            low_confidence_segments=sum(segment.low_confidence for segment in segments),
            engines=dict(Counter(result.engine for result in ordered)),
            processing_seconds=round(processing_seconds, 3),
            realtime_factor=round(audio_seconds / processing_seconds, 2)
            if processing_seconds > 0
            else None,
        ),
    )
