"""Plain-text, SubRip (SRT) and WebVTT renderings of a ``Transcript``.

A Whisper segment can run for 30 s, far longer than a caption should stay on screen,
so a segment that does not fit in ``max_lines`` lines is split into several cues at
word boundaries. Split cues are timed by word timestamps when the engine produced them
for every word, otherwise proportionally to their share of the segment's characters.
"""

from __future__ import annotations

import html
import textwrap
from dataclasses import dataclass
from itertools import accumulate

from transcription.domain import Segment, Transcript


@dataclass(frozen=True)
class _Cue:
    start_ms: int
    end_ms: int
    lines: list[str]
    channel: int | None


def to_text(t: Transcript) -> str:
    """One line per segment, labelled ``[Speaker N]`` when channels were split."""
    return "".join(f"{_srt_label(s.channel)}{s.text}\n" for s in t.segments if s.text)


def to_srt(t: Transcript, *, max_chars_per_line: int = 42, max_lines: int = 2) -> str:
    """SubRip captions. Speakers (split channels) are prefixed as ``[Speaker N]``."""
    blocks = [
        f"{n}\n{_timestamp(cue.start_ms, ',')} --> {_timestamp(cue.end_ms, ',')}\n"
        + _srt_label(cue.channel)
        + "\n".join(cue.lines)
        for n, cue in enumerate(_cues(t, max_chars_per_line, max_lines), start=1)
    ]
    return "".join(f"{block}\n\n" for block in blocks)


def to_vtt(t: Transcript, *, max_chars_per_line: int = 42, max_lines: int = 2) -> str:
    """WebVTT captions. Speakers (split channels) become voice tags ``<v Speaker N>``."""
    blocks = [
        f"{_timestamp(cue.start_ms, '.')} --> {_timestamp(cue.end_ms, '.')}\n"
        + _vtt_voice(cue.channel)
        + "\n".join(html.escape(line, quote=False) for line in cue.lines)
        for cue in _cues(t, max_chars_per_line, max_lines)
    ]
    return "WEBVTT\n\n" + "".join(f"{block}\n\n" for block in blocks)


def _srt_label(channel: int | None) -> str:
    return "" if channel is None else f"[Speaker {channel + 1}] "


def _vtt_voice(channel: int | None) -> str:
    return "" if channel is None else f"<v Speaker {channel + 1}>"


def _timestamp(ms: int, separator: str) -> str:
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    seconds, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{ms:03d}"


def _cues(t: Transcript, max_chars_per_line: int, max_lines: int) -> list[_Cue]:
    if max_chars_per_line < 1 or max_lines < 1:
        raise ValueError("max_chars_per_line and max_lines must be >= 1")
    cues = [cue for s in t.segments for cue in _segment_cues(s, max_chars_per_line, max_lines)]
    return sorted(cues, key=lambda cue: cue.start_ms)


def _segment_cues(segment: Segment, width: int, max_lines: int) -> list[_Cue]:
    tokens = segment.text.split()
    if not tokens:
        return []
    # An overlong single word stays whole on its own line rather than being hyphenated.
    lines = textwrap.wrap(
        " ".join(tokens), width=width, break_long_words=False, break_on_hyphens=False
    )
    groups = [lines[i : i + max_lines] for i in range(0, len(lines), max_lines)]
    spans = (
        [(segment.start, segment.end)]
        if len(groups) == 1
        else _split_spans(segment, tokens, [sum(len(line.split()) for line in g) for g in groups])
    )
    return [
        _cue(start, end, group, segment.channel)
        for group, (start, end) in zip(groups, spans, strict=True)
    ]


def _split_spans(
    segment: Segment, tokens: list[str], counts: list[int]
) -> list[tuple[float, float]]:
    """(start, end) seconds for consecutive cues holding ``counts`` tokens each."""
    words = segment.words if segment.words and len(segment.words) == len(tokens) else None
    chars_before = list(accumulate((len(token) for token in tokens), initial=0))
    duration = segment.end - segment.start
    spans = []
    first = 0
    for count in counts:
        last = first + count - 1
        if words:
            spans.append((words[first].start, words[last].end))
        else:
            spans.append(
                (
                    segment.start + duration * chars_before[first] / chars_before[-1],
                    segment.start + duration * chars_before[last + 1] / chars_before[-1],
                )
            )
        first += count
    return spans


def _cue(start: float, end: float, lines: list[str], channel: int | None) -> _Cue:
    start_ms = max(0, round(start * 1000))
    return _Cue(start_ms, max(round(end * 1000), start_ms + 1), lines, channel)
