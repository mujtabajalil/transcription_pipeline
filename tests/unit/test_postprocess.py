from __future__ import annotations

from typing import Any

import pytest
from prometheus_client import REGISTRY

from transcription.asr.postprocess import postprocess
from transcription.domain import (
    SAMPLE_RATE,
    ChunkResult,
    ChunkTranscript,
    PipelineConfig,
    PlannedChunk,
    Segment,
    Word,
)

CONFIG = PipelineConfig()  # no_speech 0.6, logprob -1.0, compression 2.4, low conf -0.7
CHUNK = PlannedChunk(index=3, channel=1, start=10 * SAMPLE_RATE, end=20 * SAMPLE_RATE)


def seg(text: str = "hello world", start: float = 0.0, end: float = 1.0, **kw: Any) -> Segment:
    return Segment(start=start, end=end, text=text, **kw)


def run(*segments: Segment, chunk: PlannedChunk = CHUNK) -> ChunkResult:
    transcript = ChunkTranscript(segments=list(segments), language="de", engine="fake:x")
    return postprocess(transcript, chunk, CONFIG)


def texts(result: ChunkResult) -> list[str]:
    return [s.text for s in result.segments]


def metric(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


# --- timestamps, text, metadata ----------------------------------------------------------


def test_result_carries_chunk_and_transcript_metadata() -> None:
    result = run(seg())
    assert (result.index, result.channel, result.engine, result.language) == (3, 1, "fake:x", "de")
    assert (result.start_s, result.end_s) == (10.0, 20.0)
    assert result.dropped == {}


def test_shifts_segments_and_words_onto_the_recording_timeline() -> None:
    words = [Word(start=0.25, end=0.5, text=" hello"), Word(start=0.6, end=0.9, text=" world")]
    original = seg(" hello world ", start=0.25, end=0.9, words=words)

    (out,) = run(original).segments

    assert (out.start, out.end, out.text) == (10.25, 10.9, "hello world")
    assert [(w.start, w.end, w.text) for w in out.words or []] == [
        (10.25, 10.5, "hello"),
        (10.6, 10.9, "world"),
    ]
    assert original.start == 0.25 and original.words == words, "input must not be mutated"


def test_clamps_times_to_the_chunk() -> None:
    words = [Word(start=-0.5, end=0.2, text="a"), Word(start=9.0, end=12.0, text="b")]
    (out,) = run(seg(start=-1.0, end=11.0, words=words)).segments

    assert (out.start, out.end) == (10.0, 20.0)
    assert [(w.start, w.end) for w in out.words or []] == [(10.0, 10.2), (19.0, 20.0)]


def test_end_never_precedes_start() -> None:
    words = [Word(start=0.8, end=0.3, text="x")]
    first, beyond = run(seg("a", start=2.0, end=1.0, words=words), seg("b", 15.0, 16.0)).segments

    assert (first.start, first.end) == (12.0, 12.0)
    assert first.words is not None and (first.words[0].start, first.words[0].end) == (10.8, 10.8)
    assert (beyond.start, beyond.end) == (20.0, 20.0)


def test_segments_are_ordered_by_start() -> None:
    result = run(seg("late", 5.0, 6.0), seg("early", 1.0, 2.0))
    assert texts(result) == ["early", "late"]


def test_channel_comes_from_the_chunk() -> None:
    assert run(seg(channel=7)).segments[0].channel == 1
    mono = PlannedChunk(index=0, start=0, end=SAMPLE_RATE)
    assert run(seg(channel=7), chunk=mono).segments[0].channel is None


# --- guards ------------------------------------------------------------------------------


def test_empty_text_is_dropped() -> None:
    result = run(seg(""), seg("   \n"), seg("kept"))
    assert texts(result) == ["kept"]
    assert result.dropped == {"empty": 2}


@pytest.mark.parametrize(
    ("no_speech_prob", "avg_logprob", "dropped"),
    [
        (0.61, -1.01, True),
        (0.6, -2.0, False),  # threshold is exclusive
        (0.9, -1.0, False),  # threshold is exclusive
        (0.9, -0.2, False),  # confident text over "silence" is real speech
        (0.1, -2.0, False),  # unsure decoding of clear speech is low confidence, not silence
        (None, -2.0, False),
        (0.9, None, False),
    ],
)
def test_silence_hallucination(
    no_speech_prob: float | None, avg_logprob: float | None, dropped: bool
) -> None:
    result = run(seg(no_speech_prob=no_speech_prob, avg_logprob=avg_logprob))
    assert result.dropped == ({"silence_hallucination": 1} if dropped else {})
    assert len(result.segments) == (0 if dropped else 1)


@pytest.mark.parametrize(
    ("compression_ratio", "dropped"), [(2.41, True), (2.4, False), (1.2, False), (None, False)]
)
def test_compression_ratio_repetition(compression_ratio: float | None, dropped: bool) -> None:
    result = run(seg(compression_ratio=compression_ratio))
    assert result.dropped == ({"repetition": 1} if dropped else {})


def test_repeated_text_keeps_the_first_two() -> None:
    result = run(
        *(seg(t, i, i + 0.5) for i, t in enumerate(["Go.", "go", "GO!", "go", "stop", "go"]))
    )
    assert texts(result) == ["Go.", "go", "stop", "go"]
    assert result.dropped == {"repetition": 2}


@pytest.mark.parametrize("words", [["कि", "का", "के"], ["ไม่", "ไม้", "ไม่"]], ids=["hindi", "thai"])
def test_words_differing_only_in_vowel_marks_are_not_repetition(words: list[str]) -> None:
    result = run(*(seg(t, i, i + 0.5) for i, t in enumerate(words)))
    assert texts(result) == words


def test_repetition_counts_against_the_previous_kept_segment() -> None:
    result = run(seg("go", 0, 1), seg("go", 1, 2), seg("", 2, 3), seg("go", 3, 4))
    assert texts(result) == ["go", "go"]
    assert result.dropped == {"empty": 1, "repetition": 1}


HALLUCINATIONS = [
    "Thanks for watching!",
    "Thank you for watching.",
    "Thank you so much for watching!",
    "Please subscribe",
    "Subtitles by the Amara.org community",
    "Transcribed by ESO, translated by -",
    "♪",
    "♪ ♪ ♪",
]


@pytest.mark.parametrize("text", HALLUCINATIONS)
def test_known_hallucination_dropped_when_low_confidence(text: str) -> None:
    result = run(seg(text, avg_logprob=-0.71, no_speech_prob=0.0))
    assert result.dropped == {"known_hallucination": 1}


@pytest.mark.parametrize("text", HALLUCINATIONS)
def test_known_hallucination_dropped_when_speech_is_doubtful(text: str) -> None:
    result = run(seg(text, avg_logprob=-0.1, no_speech_prob=0.31))
    assert result.dropped == {"known_hallucination": 1}


@pytest.mark.parametrize(
    ("avg_logprob", "no_speech_prob"), [(-0.7, 0.3), (-0.1, 0.01), (None, None)]
)
def test_confident_boilerplate_is_real_speech(
    avg_logprob: float | None, no_speech_prob: float | None
) -> None:
    result = run(
        seg("Thanks for watching!", avg_logprob=avg_logprob, no_speech_prob=no_speech_prob)
    )
    assert texts(result) == ["Thanks for watching!"]


def test_low_confidence_ordinary_speech_is_kept() -> None:
    result = run(seg("thanks for watching the kids", avg_logprob=-0.9, no_speech_prob=0.5))
    assert len(result.segments) == 1


def test_first_matching_reason_wins() -> None:
    both = seg("Thanks for watching", no_speech_prob=0.9, avg_logprob=-1.5, compression_ratio=3.0)
    assert run(both).dropped == {"silence_hallucination": 1}


def test_none_scores_never_trigger_a_guard() -> None:
    (out,) = run(seg("anything at all")).segments
    assert out.low_confidence is False


# --- confidence flag and metrics ---------------------------------------------------------


@pytest.mark.parametrize(("avg_logprob", "flagged"), [(-0.71, True), (-0.7, False), (None, False)])
def test_low_confidence_flag(avg_logprob: float | None, flagged: bool) -> None:
    assert run(seg(avg_logprob=avg_logprob)).segments[0].low_confidence is flagged


def test_metrics_count_drops_by_reason_and_flagged_segments() -> None:
    before_empty = metric("tx_segments_dropped_total", reason="empty")
    before_rep = metric("tx_segments_dropped_total", reason="repetition")
    before_low = metric("tx_low_confidence_segments_total")

    run(seg(""), seg("x", compression_ratio=9.0), seg("a", avg_logprob=-0.9), seg("b"))

    assert metric("tx_segments_dropped_total", reason="empty") == before_empty + 1
    assert metric("tx_segments_dropped_total", reason="repetition") == before_rep + 1
    assert metric("tx_low_confidence_segments_total") == before_low + 1
