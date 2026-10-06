"""Decode once to Whisper's native input: 16 kHz, mono, s16le, raw PCM on disk.

Every supported format costs zero code past this point, and a raw file can be
memory-mapped: an hour is 115 MB of int16 on disk instead of ~1.3 GB of float32 in RAM.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from transcription.audio.probe import FORMAT_WHITELIST, media_url, run_tool
from transcription.domain import SAMPLE_RATE
from transcription.errors import AudioTooLongError, UndecodableAudioError

log = logging.getLogger(__name__)

_BYTES_PER_SAMPLE = 2


def decode_to_pcm(
    path: str | Path,
    out_path: str | Path,
    *,
    channel: int | None = None,
    max_seconds: int,
    timeout_s: float,
) -> int:
    """Decode the first audio track of ``path`` into raw 16 kHz mono s16le at ``out_path``.

    ``channel=None`` downmixes to mono; an int keeps only that channel (split-channel call
    recordings); it must be below ``AudioInfo.channels`` because ffmpeg's pan filter
    yields silence, not an error, for a channel that doesn't exist. Returns the decoded
    sample count, which is the true duration: VBR MP3 and streamed WebM headers lie or
    are absent. ``-t max_seconds+1`` caps the work a lying header can cause while still
    letting us detect "too long".

    Raises ``UndecodableAudioError`` (422) on ffmpeg failure, timeout or zero samples,
    ``AudioTooLongError`` (413) above ``max_seconds``. On any failure ``out_path`` is
    removed: a partial decode (up to ``max_seconds+1`` s, ~460 MB at the default 4 h cap)
    is never left on disk for a later ``open_pcm`` to trust.
    """
    if channel is not None and channel < 0:
        raise ValueError(f"channel must be >= 0, got {channel}")
    out = Path(out_path)
    if out.resolve() == Path(path).resolve():
        # ffmpeg refuses this too, but the failure cleanup below would then delete the input.
        raise ValueError("out_path must not be the input file")
    mix = ["-ac", "1"] if channel is None else ["-af", f"pan=mono|c0=c{channel}"]
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-y",
        "-format_whitelist",
        FORMAT_WHITELIST,
        "-i",
        media_url(path),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-t",
        str(max_seconds + 1),
        *mix,
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        "-f",
        "s16le",
        media_url(out),
    ]
    started = time.monotonic()
    try:
        if run_tool(cmd, timeout_s=timeout_s).returncode != 0:
            raise UndecodableAudioError()
        samples = out.stat().st_size // _BYTES_PER_SAMPLE
        if samples == 0:
            raise UndecodableAudioError("no decodable audio")
        if samples > max_seconds * SAMPLE_RATE:
            raise AudioTooLongError(f"audio exceeds the maximum duration of {max_seconds} s")
    except BaseException:
        out.unlink(missing_ok=True)
        raise
    log.info(
        "audio decoded",
        extra={
            "channel": channel,
            "samples": samples,
            "audio_s": round(samples / SAMPLE_RATE, 3),
            "elapsed_s": round(time.monotonic() - started, 3),
        },
    )
    return samples


def open_pcm(path: str | Path) -> np.memmap[Any, np.dtype[np.int16]]:
    """Read-only memory map of a file written by ``decode_to_pcm``.

    Pages load on demand, so slicing one chunk never reads the rest of the file.
    """
    if Path(path).stat().st_size < _BYTES_PER_SAMPLE:
        raise UndecodableAudioError("no decodable audio")
    return np.memmap(path, dtype=np.int16, mode="r")


def to_float32(samples: npt.NDArray[np.int16]) -> npt.NDArray[np.float32]:
    """int16 PCM to float32 in [-1, 1), the scale Whisper and Silero expect.

    Allocates only ``len(samples)`` floats, so converting a memmap slice touches just
    that slice.
    """
    return np.divide(samples, np.float32(32768), dtype=np.float32)
