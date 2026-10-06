"""Engine-agnostic quality guards, applied to every chunk before it is checkpointed.

Whisper invents text on silence and music ("Thanks for watching!"), and a decoding loop
can repeat one phrase for the rest of a window. Temperature fallback catches some of
that inside the engine; these guards catch what is left, using the per-segment scores
the engine reports. A score an engine does not report (``None``) never triggers a guard,
so hosted engines without Whisper's scores pass through untouched.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter

from transcription.domain import (
    ChunkResult,
    ChunkTranscript,
    PipelineConfig,
    PlannedChunk,
    Segment,
    Word,
)
from transcription.metrics import LOW_CONFIDENCE_SEGMENTS, SEGMENTS_DROPPED

log = logging.getLogger(__name__)

_MAX_CONSECUTIVE_REPEATS = 2
"""Identical consecutive segments beyond this many are a decoding loop, not speech."""

_SUSPECT_NO_SPEECH_PROB = 0.3
"""Above this Whisper already doubts there was speech, so boilerplate is suspect even
when the decoder was confident about the words."""

_KNOWN_HALLUCINATIONS = re.compile(
    r"(?:thanks|thank you)(?: so much| very much)? for watching"
    r"|(?:please )?(?:like and )?subscribe(?: to (?:my|our|the) channel)?"
    r"|(?:subtitles|captions|transcribed|transcription|translated|translation) by .*"
)
"""Boilerplate Whisper learnt from YouTube subtitles, matched against normalised text."""


def postprocess(
    transcript: ChunkTranscript, chunk: PlannedChunk, config: PipelineConfig
) -> ChunkResult:
    """Turn one engine result into the absolute-time, guarded checkpoint for ``chunk``.

    Times are shifted onto the recording's timeline and clamped to the chunk, because
    chunks are contiguous slices and an engine can overshoot the audio it was given.
    Guards run in a fixed order and each dropped segment is counted under the first
    reason that matched: ``empty``, ``silence_hallucination``, ``repetition``,
    ``known_hallucination``.
    """
    dropped: Counter[str] = Counter()
    kept: list[Segment] = []
    last_text: str | None = None
    streak = 0
    absolute = sorted(
        (_to_absolute(segment, chunk, config) for segment in transcript.segments),
        key=lambda segment: segment.start,
    )
    for segment in absolute:
        text = _normalize(segment.text)
        prior_repeats = streak if text == last_text else 0
        reason = _drop_reason(segment, text, prior_repeats, config)
        if reason is not None:
            dropped[reason] += 1
            SEGMENTS_DROPPED.labels(reason=reason).inc()
            continue
        streak = prior_repeats + 1
        last_text = text
        if segment.low_confidence:
            LOW_CONFIDENCE_SEGMENTS.inc()
        kept.append(segment)

    if dropped:
        log.debug(
            "segments dropped by quality guards",
            extra={"chunk": chunk.index, "channel": chunk.channel, "dropped": dict(dropped)},
        )
    return ChunkResult(
        index=chunk.index,
        channel=chunk.channel,
        start_s=chunk.start_s,
        end_s=chunk.end_s,
        engine=transcript.engine,
        language=transcript.language,
        segments=kept,
        dropped=dict(dropped),
    )


def _drop_reason(
    segment: Segment, text: str, prior_repeats: int, config: PipelineConfig
) -> str | None:
    if not segment.text:
        return "empty"
    if _above(segment.no_speech_prob, config.no_speech_threshold) and _below(
        segment.avg_logprob, config.logprob_threshold
    ):
        return "silence_hallucination"
    if (
        _above(segment.compression_ratio, config.compression_ratio_threshold)
        or prior_repeats >= _MAX_CONSECUTIVE_REPEATS
    ):
        return "repetition"
    suspect = segment.low_confidence or _above(segment.no_speech_prob, _SUSPECT_NO_SPEECH_PROB)
    if suspect and _is_known_hallucination(text):
        return "known_hallucination"
    return None


def _is_known_hallucination(text: str) -> bool:
    # Empty after normalisation means the segment was only symbols, e.g. "♪ ♪".
    return not text or _KNOWN_HALLUCINATIONS.fullmatch(text) is not None


def _normalize(text: str) -> str:
    # By category, not [^\w\s]: \w excludes combining marks, so कि/का/के would all
    # become क and the repetition guard would drop distinct Hindi or Thai speech.
    kept = "".join(c for c in text.lower() if unicodedata.category(c)[0] not in "PS")
    return " ".join(kept.split())


def _above(value: float | None, limit: float) -> bool:
    return value is not None and value > limit


def _below(value: float | None, limit: float) -> bool:
    return value is not None and value < limit


def _to_absolute(segment: Segment, chunk: PlannedChunk, config: PipelineConfig) -> Segment:
    start = _clamp(segment.start, chunk)
    words = None if segment.words is None else [_word_to_absolute(w, chunk) for w in segment.words]
    return segment.model_copy(
        update={
            "start": start,
            "end": max(_clamp(segment.end, chunk), start),
            "text": segment.text.strip(),
            "channel": chunk.channel,
            "words": words,
            "low_confidence": _below(segment.avg_logprob, config.low_confidence_logprob),
        }
    )


def _word_to_absolute(word: Word, chunk: PlannedChunk) -> Word:
    start = _clamp(word.start, chunk)
    return word.model_copy(
        update={
            "start": start,
            "end": max(_clamp(word.end, chunk), start),
            "text": word.text.strip(),
        }
    )


def _clamp(relative_s: float, chunk: PlannedChunk) -> float:
    return min(max(chunk.start_s + relative_s, chunk.start_s), chunk.end_s)
