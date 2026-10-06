"""Pack speech regions into chunks of at most one Whisper window that start and end in
pauses.

Chunks are contiguous slices of the original timeline and never overlap, so stitching
is ``absolute = chunk.start + t`` with nothing to de-duplicate at the seams.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from transcription.domain import PlannedChunk


def _check_window(window: int, frame: int) -> None:
    if frame <= 0 or window <= 0 or window % frame:
        raise ValueError(f"window ({window}) must be a positive multiple of frame ({frame})")


def quietest_cut(
    audio: npt.NDArray[np.int16], window_start: int, *, window: int, frame: int
) -> int:
    """Sample index at the centre of the lowest-energy ``frame`` in
    ``audio[window_start : window_start + window]``.

    Used to cut nonstop speech: the quietest 100 ms of the last few seconds is almost
    always a gap between words.
    """
    _check_window(window, frame)
    if window_start < 0 or window_start + window > len(audio):
        raise ValueError(
            f"window [{window_start}, {window_start + window}) outside audio of {len(audio)}"
        )
    # float64 because squared int16 overflows int16/int32 and loses precision in float32.
    frames = audio[window_start : window_start + window].astype(np.float64).reshape(-1, frame)
    quietest = int(np.argmin(np.square(frames).sum(axis=1)))
    return window_start + quietest * frame + frame // 2


def plan_chunks(
    regions: Iterable[tuple[int, int]],
    audio: npt.NDArray[np.int16],
    *,
    max_chunk: int,
    cut_window: int,
    cut_frame: int,
    max_gap: int,
    channel: int | None = None,
    start_index: int = 0,
) -> tuple[list[PlannedChunk], int]:
    """Greedily pack sorted speech ``regions`` into chunks; returns (chunks, forced_cuts).

    A region joins the current chunk while the chunk stays within ``max_chunk`` and the
    silence before the region is at most ``max_gap`` (long silences inside a window are
    hallucination bait). A region longer than ``max_chunk`` is split by forced cuts at
    the quietest ``cut_frame`` of the last ``cut_window`` samples of each window; those
    chunks get ``forced_cut=True``. Indexes run from ``start_index`` so per-channel plans
    concatenate into one job-wide numbering.
    """
    if not 0 < cut_window < max_chunk:
        raise ValueError(f"need 0 < cut_window ({cut_window}) < max_chunk ({max_chunk})")
    _check_window(cut_window, cut_frame)
    spans: list[tuple[int, int, bool]] = []  # (start, end, end_is_forced_cut)
    for start, end in regions:
        if spans and end - spans[-1][0] <= max_chunk and start - spans[-1][1] <= max_gap:
            spans[-1] = (spans[-1][0], end, False)
            continue
        while end - start > max_chunk:
            cut = quietest_cut(
                audio, start + max_chunk - cut_window, window=cut_window, frame=cut_frame
            )
            spans.append((start, cut, True))
            start = cut
        spans.append((start, end, False))
    chunks = [
        PlannedChunk(index=start_index + i, channel=channel, start=s, end=e, forced_cut=forced)
        for i, (s, e, forced) in enumerate(spans)
    ]
    return chunks, sum(forced for *_, forced in spans)
