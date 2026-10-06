"""Domain types shared by the pipeline, worker, API and storage layers.

Pydantic models so the same objects validate API input, persist as JSONB checkpoints,
and render as API output without hand-written (de)serialisers.

Timestamp convention: ``Segment`` times are seconds. Inside a ``ChunkTranscript`` they
are relative to the chunk start (what the engine saw); everywhere else they are
absolute positions on the original recording.
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

SAMPLE_RATE = 16_000
"""Whisper's native input rate. Everything after decode is 16 kHz mono int16."""


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED)


class PipelineConfig(BaseModel):
    """Audio-pipeline knobs. Built from Settings in services; defaults suit tests/CLI."""

    model_config = ConfigDict(frozen=True)

    max_audio_seconds: int = 4 * 3600
    ffprobe_timeout_s: float = 30
    ffmpeg_timeout_s: float = 900
    chunk_max_s: float = 30.0
    forced_cut_window_s: float = 5.0
    forced_cut_frame_ms: int = 100
    max_pack_gap_s: float = 2.0
    vad_block_s: int = 600
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 500
    vad_speech_pad_ms: int = 200
    chunk_max_retries: int = 3
    chunk_retry_base_delay_s: float = 1.0
    no_speech_threshold: float = 0.6
    logprob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4
    low_confidence_logprob: float = -0.7


_LANG = re.compile(r"^[a-z]{2,3}$")


class TranscriptionOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    language: str | None = Field(
        default=None, description="ISO 639-1/3 code. Pins the language instead of detecting it."
    )
    prompt: str | None = Field(
        default=None,
        max_length=800,
        description="Glossary of names/jargon, passed to Whisper as initial_prompt for "
        "consistent spelling across independently decoded chunks.",
    )
    split_channels: bool = Field(
        default=False,
        description="Transcribe each channel separately (call-centre recordings with one "
        "speaker per channel). Segments carry their channel index.",
    )
    word_timestamps: bool = False

    @field_validator("language")
    @classmethod
    def _lang(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip().lower()
        if not _LANG.match(v):
            raise ValueError("language must be an ISO 639-1/639-3 code like 'en' or 'yue'")
        return v

    def fingerprint(self, audio_identity: str) -> str:
        """Stable hash of (audio, options) used for idempotency and content dedupe."""
        payload = json.dumps(
            {"audio": audio_identity, "options": self.model_dump(mode="json")}, sort_keys=True
        )
        return hashlib.sha256(payload.encode()).hexdigest()


class AudioInfo(BaseModel):
    """What ffprobe found in the bytes (never the extension or Content-Type)."""

    container: str
    codec: str | None = None
    channels: int = 1
    sample_rate: int | None = None
    header_duration_s: float | None = Field(
        default=None, description="Informational only; real duration comes from decoded samples."
    )


class PlannedChunk(BaseModel):
    """A contiguous slice of one channel's 16 kHz timeline, in samples. At most one
    Whisper window long, starting and ending in a pause (or a forced cut)."""

    model_config = ConfigDict(frozen=True)

    index: int
    channel: int | None = None
    start: int
    end: int
    forced_cut: bool = False
    """True when ``end`` was a forced cut inside continuous speech."""

    @property
    def start_s(self) -> float:
        return self.start / SAMPLE_RATE

    @property
    def end_s(self) -> float:
        return self.end / SAMPLE_RATE

    @property
    def duration_s(self) -> float:
        return (self.end - self.start) / SAMPLE_RATE


class ChunkPlan(BaseModel):
    """Persisted on the job so a resumed worker reuses identical chunk boundaries."""

    chunks: list[PlannedChunk]
    audio_samples: int
    speech_samples: int
    forced_cuts: int
    channels: list[int | None]

    @property
    def total(self) -> int:
        return len(self.chunks)


class Word(BaseModel):
    start: float
    end: float
    text: str
    probability: float | None = None


class Segment(BaseModel):
    start: float
    end: float
    text: str
    channel: int | None = None
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    words: list[Word] | None = None
    low_confidence: bool = False


class ChunkTranscript(BaseModel):
    """Raw engine output for one chunk. Segment times are RELATIVE to the chunk."""

    segments: list[Segment]
    language: str | None = None
    language_probability: float | None = None
    engine: str


class ChunkResult(BaseModel):
    """Post-processed, absolute-time result for one chunk; the checkpoint unit."""

    index: int
    channel: int | None = None
    start_s: float
    end_s: float
    engine: str
    language: str | None = None
    segments: list[Segment]
    dropped: dict[str, int] = Field(default_factory=dict)
    """Segments removed by quality guards, by reason."""


class TranscriptStats(BaseModel):
    audio_seconds: float
    speech_seconds: float
    transcribed_seconds: float
    chunks: int
    forced_cuts: int
    dropped_segments: dict[str, int] = Field(default_factory=dict)
    low_confidence_segments: int = 0
    engines: dict[str, int] = Field(default_factory=dict)
    """Chunks transcribed per engine (shows fallback usage)."""
    processing_seconds: float | None = None
    realtime_factor: float | None = None


class Transcript(BaseModel):
    language: str | None
    language_probability: float | None = None
    duration_s: float
    text: str
    segments: list[Segment]
    stats: TranscriptStats
    audio: AudioInfo
