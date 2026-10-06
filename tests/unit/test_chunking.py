from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from transcription.audio.chunking import plan_chunks, quietest_cut
from transcription.audio.decode import decode_to_pcm, open_pcm
from transcription.audio.vad import detect_speech
from transcription.domain import SAMPLE_RATE, PipelineConfig, PlannedChunk

SR = SAMPLE_RATE
MAX_CHUNK = 30 * SR
CUT_WINDOW = 5 * SR
CUT_FRAME = SR // 10
MAX_GAP = 2 * SR

NOISE = np.random.default_rng(0).integers(-3000, 3000, 70 * SR).astype(np.int16)


def plan(
    regions: list[tuple[int, int]],
    audio: npt.NDArray[np.int16] = NOISE,
    *,
    channel: int | None = None,
    start_index: int = 0,
) -> tuple[list[PlannedChunk], int]:
    return plan_chunks(
        regions,
        audio,
        max_chunk=MAX_CHUNK,
        cut_window=CUT_WINDOW,
        cut_frame=CUT_FRAME,
        max_gap=MAX_GAP,
        channel=channel,
        start_index=start_index,
    )


def secs(*spans: tuple[float, float]) -> list[tuple[int, int]]:
    return [(int(a * SR), int(b * SR)) for a, b in spans]


def spans(chunks: list[PlannedChunk]) -> list[tuple[int, int]]:
    return [(c.start, c.end) for c in chunks]


# --- quietest_cut ------------------------------------------------------------------------


def test_quietest_cut_returns_centre_of_quietest_frame() -> None:
    audio = NOISE[: 10 * SR].copy()
    quiet = 5 * SR + 23 * CUT_FRAME
    audio[quiet : quiet + CUT_FRAME] //= 10

    cut = quietest_cut(audio, 5 * SR, window=CUT_WINDOW, frame=CUT_FRAME)

    assert cut == quiet + CUT_FRAME // 2


def test_quietest_cut_does_not_overflow_int16() -> None:
    # 32767² wraps to 1 in int16 arithmetic, which would make full scale look silent.
    audio = np.full(10 * CUT_FRAME, 32767, np.int16)
    audio[3 * CUT_FRAME : 4 * CUT_FRAME] = 1000

    assert quietest_cut(audio, 0, window=10 * CUT_FRAME, frame=CUT_FRAME) == 3.5 * CUT_FRAME


def test_quietest_cut_ignores_samples_outside_its_window() -> None:
    audio = NOISE[: 4 * CUT_FRAME].copy()
    audio[:CUT_FRAME] = 0  # silent, but before the window

    cut = quietest_cut(audio, CUT_FRAME, window=3 * CUT_FRAME, frame=CUT_FRAME)

    assert CUT_FRAME < cut < 4 * CUT_FRAME


@pytest.mark.parametrize(
    ("start", "window", "frame"),
    [(0, 1000, 300), (0, 1000, 0), (-1, 1000, 100), (9500, 1000, 100)],
)
def test_quietest_cut_rejects_bad_windows(start: int, window: int, frame: int) -> None:
    with pytest.raises(ValueError):
        quietest_cut(NOISE[:10_000], start, window=window, frame=frame)


# --- plan_chunks -------------------------------------------------------------------------


def test_nonstop_speech_is_cut_in_the_planted_pause() -> None:
    audio = NOISE.copy()
    audio[27 * SR : int(27.2 * SR)] = 0

    chunks, forced = plan([(0, len(audio))], audio)

    assert 27 * SR <= chunks[0].end <= 27.2 * SR
    assert forced == 2 and len(chunks) == 3
    assert [c.forced_cut for c in chunks] == [True, True, False]
    assert chunks[0].start == 0 and chunks[-1].end == len(audio)
    assert all(a.end == b.start for a, b in pairwise(chunks))
    assert all(c.end - c.start <= MAX_CHUNK for c in chunks)


def test_packs_across_short_gaps_and_splits_on_long_ones() -> None:
    chunks, forced = plan(secs((0, 1), (2, 3), (10, 11)))

    assert spans(chunks) == secs((0, 3), (10, 11))
    assert forced == 0


def test_gap_rule_boundary() -> None:
    at_limit = [(0, SR), (SR + MAX_GAP, 2 * SR + MAX_GAP)]
    over_limit = [(0, SR), (SR + MAX_GAP + 1, 2 * SR + MAX_GAP + 1)]

    assert spans(plan(at_limit)[0]) == [(0, 2 * SR + MAX_GAP)]
    assert spans(plan(over_limit)[0]) == over_limit


def test_size_rule() -> None:
    assert spans(plan(secs((0, 20), (21, 40)))[0]) == secs((0, 20), (21, 40))
    assert spans(plan([(0, SR), (2 * SR, MAX_CHUNK)])[0]) == [(0, MAX_CHUNK)]
    assert spans(plan([(0, SR), (2 * SR, MAX_CHUNK + 1)])[0]) == [
        (0, SR),
        (2 * SR, MAX_CHUNK + 1),
    ]


def test_region_of_exactly_max_chunk_is_not_cut() -> None:
    chunks, forced = plan([(5, 5 + MAX_CHUNK)])

    assert spans(chunks) == [(5, 5 + MAX_CHUNK)] and forced == 0


def test_index_and_channel_propagate() -> None:
    chunks, _ = plan(secs((0, 1), (10, 11), (20, 61)), channel=1, start_index=7)

    assert [c.index for c in chunks] == list(range(7, 7 + len(chunks)))
    assert {c.channel for c in chunks} == {1}
    assert [c.forced_cut for c in chunks] == [False, False, True, False]


def test_forced_chunk_is_never_extended() -> None:
    # The tail after a forced cut may absorb the next region; the cut chunk may not.
    chunks, forced = plan(secs((0, 40), (41, 42)))

    assert forced == 1
    assert [c.forced_cut for c in chunks] == [True, False]
    assert chunks[1].start == chunks[0].end and chunks[1].end == 42 * SR


def test_no_regions_no_chunks() -> None:
    assert plan([]) == ([], 0)


@pytest.mark.parametrize(
    ("cut_window", "cut_frame"),
    [(0, CUT_FRAME), (MAX_CHUNK, CUT_FRAME), (MAX_CHUNK + 1, 1), (CUT_WINDOW, 3 * CUT_FRAME)],
)
def test_rejects_bad_cut_window(cut_window: int, cut_frame: int) -> None:
    with pytest.raises(ValueError):
        plan_chunks(
            [],
            NOISE,
            max_chunk=MAX_CHUNK,
            cut_window=cut_window,
            cut_frame=cut_frame,
            max_gap=MAX_GAP,
        )


def random_case(rng: np.random.Generator) -> tuple[list[tuple[int, int]], int, int, int, int]:
    frame = int(rng.integers(1, 20))
    cut_window = frame * int(rng.integers(1, 20))
    max_chunk = cut_window + int(rng.integers(1, 400))
    max_gap = int(rng.integers(0, 200))
    regions, pos = [], int(rng.integers(0, 50))
    for _ in range(int(rng.integers(0, 40))):
        # Mostly short regions (exercise packing), some longer than max_chunk (forced cuts).
        length = int(rng.integers(1, 3 * max_chunk if rng.random() < 0.3 else max_chunk // 3 + 2))
        regions.append((pos, pos + length))
        pos += length + int(rng.integers(1, 2 * max_gap + 2))
    return regions, max_chunk, cut_window, frame, max_gap


@pytest.mark.parametrize("seed", range(200))
def test_plan_properties(seed: int) -> None:
    rng = np.random.default_rng(seed)
    regions, max_chunk, cut_window, frame, max_gap = random_case(rng)
    total = regions[-1][1] if regions else 0
    audio = rng.integers(-32768, 32768, total, dtype=np.int16)

    chunks, forced = plan_chunks(
        regions,
        audio,
        max_chunk=max_chunk,
        cut_window=cut_window,
        cut_frame=frame,
        max_gap=max_gap,
        start_index=3,
    )

    assert [c.index for c in chunks] == list(range(3, 3 + len(chunks)))
    assert forced == sum(c.forced_cut for c in chunks)
    assert all(0 < c.end - c.start <= max_chunk for c in chunks)
    assert all(a.end <= b.start for a, b in pairwise(chunks))

    covered = np.zeros(total, bool)
    for c in chunks:
        covered[c.start : c.end] = True
    assert all(covered[s:e].all() for s, e in regions)

    end_of = dict(regions)
    starts = set(end_of)
    ends = set(end_of.values())
    for a, b in pairwise(chunks):
        if not a.forced_cut:  # greedy: b's first region could not have joined a
            assert end_of[b.start] - a.start > max_chunk or b.start - a.end > max_gap
    for i, c in enumerate(chunks):
        if i and chunks[i - 1].forced_cut:
            assert c.start == chunks[i - 1].end  # contiguous inside a split region
        else:
            assert c.start in starts
        if c.forced_cut:
            window_start = c.start + max_chunk - cut_window
            assert window_start <= c.end < window_start + cut_window
            assert (c.end - window_start - frame // 2) % frame == 0
        else:
            assert c.end in ends

    for (_, end), (start, _) in pairwise(regions):
        if start - end > max_gap:  # never packed across a long silence
            assert not any(c.start < end and start < c.end for c in chunks)


def test_monologue_end_to_end(samples_dir: Path, tmp_path: Path) -> None:
    cfg = PipelineConfig()
    out = tmp_path / "monologue.pcm"
    samples = decode_to_pcm(
        samples_dir / "monologue.mp3",
        out,
        max_seconds=cfg.max_audio_seconds,
        timeout_s=cfg.ffmpeg_timeout_s,
    )
    audio = open_pcm(out)
    regions = detect_speech(
        audio,
        block_samples=cfg.vad_block_s * SR,
        threshold=cfg.vad_threshold,
        min_silence_ms=cfg.vad_min_silence_ms,
        speech_pad_ms=cfg.vad_speech_pad_ms,
    )

    chunks, forced = plan_chunks(
        regions,
        audio,
        max_chunk=int(cfg.chunk_max_s * SR),
        cut_window=int(cfg.forced_cut_window_s * SR),
        cut_frame=cfg.forced_cut_frame_ms * SR // 1000,
        max_gap=int(cfg.max_pack_gap_s * SR),
    )

    assert forced == 1 and len(chunks) == 2
    first, second = chunks
    assert first.forced_cut and not second.forced_cut
    assert 27.5 < first.end_s < 29.0  # between words, found by the energy search
    assert (first.start, second.start, second.end) == (0, first.end, samples)
