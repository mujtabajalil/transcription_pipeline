from __future__ import annotations

from typing import Any

import pytest

from transcription.domain import AudioInfo, Segment, Transcript, TranscriptStats, Word
from transcription.formats import to_srt, to_text, to_vtt


def seg(start: float, end: float, text: str, **kw: Any) -> Segment:
    return Segment(start=start, end=end, text=text, **kw)


def transcript(*segments: Segment) -> Transcript:
    return Transcript(
        language="en",
        duration_s=max((s.end for s in segments), default=0.0),
        text=" ".join(s.text for s in segments),
        segments=list(segments),
        stats=TranscriptStats(
            audio_seconds=0, speech_seconds=0, transcribed_seconds=0, chunks=0, forced_cuts=0
        ),
        audio=AudioInfo(container="wav"),
    )


BASIC = transcript(seg(0.0, 2.5, "Hello there."), seg(3.25, 5.0, "General Kenobi!"))
CALL = transcript(seg(0.5, 1.5, "Thanks for calling.", channel=0), seg(1.75, 3.0, "Hi.", channel=1))


# --- golden output -------------------------------------------------------------------------


def test_srt() -> None:
    assert to_srt(BASIC) == (
        "1\n00:00:00,000 --> 00:00:02,500\nHello there.\n\n"
        "2\n00:00:03,250 --> 00:00:05,000\nGeneral Kenobi!\n\n"
    )


def test_vtt() -> None:
    assert to_vtt(BASIC) == (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:02.500\nHello there.\n\n"
        "00:00:03.250 --> 00:00:05.000\nGeneral Kenobi!\n\n"
    )


def test_text_is_one_line_per_segment() -> None:
    assert to_text(BASIC) == "Hello there.\nGeneral Kenobi!\n"


def test_split_channels_are_labelled_as_speakers() -> None:
    assert to_srt(CALL) == (
        "1\n00:00:00,500 --> 00:00:01,500\n[Speaker 1] Thanks for calling.\n\n"
        "2\n00:00:01,750 --> 00:00:03,000\n[Speaker 2] Hi.\n\n"
    )
    assert to_vtt(CALL) == (
        "WEBVTT\n\n"
        "00:00:00.500 --> 00:00:01.500\n<v Speaker 1>Thanks for calling.\n\n"
        "00:00:01.750 --> 00:00:03.000\n<v Speaker 2>Hi.\n\n"
    )
    assert to_text(CALL) == "[Speaker 1] Thanks for calling.\n[Speaker 2] Hi.\n"


def test_empty_transcript() -> None:
    empty = transcript()
    assert (to_srt(empty), to_vtt(empty), to_text(empty)) == ("", "WEBVTT\n\n", "")


# --- wrapping and splitting ------------------------------------------------------------


def test_lines_wrap_at_word_boundaries() -> None:
    t = transcript(seg(0, 2, "the quick brown fox"))
    assert to_srt(t, max_chars_per_line=10) == (
        "1\n00:00:00,000 --> 00:00:02,000\nthe quick\nbrown fox\n\n"
    )


def test_overlong_word_stays_whole_on_its_own_line() -> None:
    t = transcript(seg(0, 2, "a supercalifragilistic b"))
    assert to_vtt(t, max_chars_per_line=10, max_lines=3) == (
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\na\nsupercalifragilistic\nb\n\n"
    )


def test_long_segment_is_split_into_cues_timed_by_character_share() -> None:
    t = transcript(seg(10, 20, "aaaa bbbb cc dd eeeeeeee", channel=0))
    assert to_srt(t, max_chars_per_line=9, max_lines=1) == (
        "1\n00:00:10,000 --> 00:00:14,000\n[Speaker 1] aaaa bbbb\n\n"
        "2\n00:00:14,000 --> 00:00:16,000\n[Speaker 1] cc dd\n\n"
        "3\n00:00:16,000 --> 00:00:20,000\n[Speaker 1] eeeeeeee\n\n"
    )


def test_long_segment_is_split_into_cues_timed_by_words() -> None:
    words = [
        Word(start=1.0, end=1.4, text="one"),
        Word(start=1.5, end=2.0, text="two"),
        Word(start=6.0, end=6.5, text="three"),
        Word(start=7.0, end=7.25, text="four"),
    ]
    t = transcript(seg(0.5, 9.0, "one two three four", words=words))
    assert to_vtt(t, max_chars_per_line=8, max_lines=1) == (
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:02.000\none two\n\n"
        "00:00:06.000 --> 00:00:06.500\nthree\n\n"
        "00:00:07.000 --> 00:00:07.250\nfour\n\n"
    )


def test_words_that_do_not_align_with_the_text_fall_back_to_character_timing() -> None:
    words = [Word(start=1.0, end=2.0, text="onetwo")]
    t = transcript(seg(0.0, 2.0, "one two", words=words))
    assert to_srt(t, max_chars_per_line=3, max_lines=1) == (
        "1\n00:00:00,000 --> 00:00:01,000\none\n\n2\n00:00:01,000 --> 00:00:02,000\ntwo\n\n"
    )


def test_cue_never_exceeds_max_lines() -> None:
    text = " ".join(f"word{i}" for i in range(40))
    srt = to_srt(transcript(seg(0, 30, text)), max_chars_per_line=20, max_lines=2)
    for block in srt.strip().split("\n\n"):
        _, _, *lines = block.split("\n")
        assert 1 <= len(lines) <= 2
        assert all(len(line) <= 20 for line in lines)
    assert (
        " ".join(line for block in srt.strip().split("\n\n") for line in block.split("\n")[2:])
        == text
    )


# --- escaping and timing -----------------------------------------------------------------


def test_vtt_escapes_markup_but_srt_keeps_text_verbatim() -> None:
    t = transcript(seg(0, 1, "a < b & c > d"))
    assert "a &lt; b &amp; c &gt; d\n" in to_vtt(t)
    assert "a < b & c > d\n" in to_srt(t)


def test_timestamps_past_one_hour() -> None:
    t = transcript(seg(3725.5, 3727.125, "late"), seg(360_000.0, 360_001.0, "very late"))
    assert "01:02:05,500 --> 01:02:07,125" in to_srt(t)
    assert "100:00:00.000 --> 100:00:01.000" in to_vtt(t)


def test_times_are_never_negative_and_cues_last_at_least_one_millisecond() -> None:
    t = transcript(seg(-0.5, -0.2, "before"), seg(5.0, 5.0, "instant"))
    assert to_srt(t) == (
        "1\n00:00:00,000 --> 00:00:00,001\nbefore\n\n2\n00:00:05,000 --> 00:00:05,001\ninstant\n\n"
    )


def test_cues_are_ordered_by_start() -> None:
    t = transcript(seg(4, 5, "second", channel=1), seg(1, 2, "first", channel=0))
    assert to_vtt(t) == (
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:02.000\n<v Speaker 1>first\n\n"
        "00:00:04.000 --> 00:00:05.000\n<v Speaker 2>second\n\n"
    )


def test_blank_segments_produce_no_cue() -> None:
    assert to_srt(transcript(seg(0, 1, "   "))) == ""


@pytest.mark.parametrize(("chars", "lines"), [(0, 2), (42, 0)])
def test_rejects_non_positive_limits(chars: int, lines: int) -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        to_srt(BASIC, max_chars_per_line=chars, max_lines=lines)
