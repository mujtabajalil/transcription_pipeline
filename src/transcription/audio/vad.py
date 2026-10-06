"""Silero VAD (bundled with faster-whisper) run block-by-block over a memmap.

Silence and music never reach the model: less compute and fewer hallucinations. Running
block by block keeps memory O(block) instead of O(file) for multi-hour recordings.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import numpy.typing as npt
from faster_whisper.vad import VadOptions, get_speech_timestamps

from transcription.audio.decode import to_float32
from transcription.domain import SAMPLE_RATE

log = logging.getLogger(__name__)


def detect_speech(
    audio: npt.NDArray[np.int16],
    *,
    block_samples: int,
    threshold: float,
    min_silence_ms: int,
    speech_pad_ms: int,
) -> list[tuple[int, int]]:
    """Sorted, non-overlapping ``(start, end)`` sample ranges of speech in ``audio``.

    Speech running across a block boundary comes back from Silero as two regions that
    touch at the boundary (padding is clamped to the block); they are merged so a block
    edge never becomes a chunk edge.
    """
    if block_samples <= 0:
        raise ValueError(f"block_samples must be > 0, got {block_samples}")
    options = VadOptions(
        threshold=threshold,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
    )
    started = time.monotonic()
    regions: list[tuple[int, int]] = []
    for offset in range(0, len(audio), block_samples):
        block = to_float32(audio[offset : offset + block_samples])
        for ts in get_speech_timestamps(block, options):
            start, end = offset + int(ts["start"]), offset + int(ts["end"])
            if regions and start <= regions[-1][1]:
                regions[-1] = (regions[-1][0], max(regions[-1][1], end))
            else:
                regions.append((start, end))
    log.info(
        "speech detected",
        extra={
            "regions": len(regions),
            "speech_s": round(sum(e - s for s, e in regions) / SAMPLE_RATE, 3),
            "audio_s": round(len(audio) / SAMPLE_RATE, 3),
            "elapsed_s": round(time.monotonic() - started, 3),
        },
    )
    return regions
