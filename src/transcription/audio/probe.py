"""Identify the real container/codec from the bytes with ffprobe.

The extension and Content-Type are never consulted: clients mislabel files, and ffmpeg's
own content sniffing (restricted to ``ALLOWED_DEMUXERS``) is what decode will use anyway.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ValidationError

from transcription.domain import AudioInfo
from transcription.errors import NoAudioStreamError, UndecodableAudioError, UnsupportedMediaError

log = logging.getLogger(__name__)

ALLOWED_DEMUXERS: tuple[str, ...] = (
    "wav",
    "aiff",
    "mp3",
    "flac",
    "ogg",
    "mov",
    "matroska",
    "aac",
)
"""ffmpeg demuxer names we accept. Passed to ffprobe *and* ffmpeg as ``-format_whitelist``
so ffmpeg itself refuses everything else (hls, concat, image2, tty, ...): playlists and
concat scripts are how a media file turns into SSRF or a local-file read."""

FORMAT_WHITELIST = ",".join(ALLOWED_DEMUXERS)

_STDERR_TAIL = 2000


def media_url(path: str | Path) -> str:
    """``file:`` + absolute path, so a name like ``-i`` or ``http:x`` can never be read
    as an ffmpeg option or protocol."""
    return "file:" + str(Path(path).absolute())


def run_tool(cmd: list[str], *, timeout_s: float) -> subprocess.CompletedProcess[str]:
    """Run ffmpeg/ffprobe isolated in a child with a hard timeout.

    A hung or crashing decoder costs a child process, never the worker. On timeout the
    child is killed and ``UndecodableAudioError`` raised; a nonzero exit is returned for
    the caller to classify, with the stderr tail logged (never surfaced to clients: it
    contains filesystem paths).
    """
    started = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            errors="replace",  # ffmpeg echoes arbitrary bytes from metadata and file names
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        log.warning("media tool timed out", extra={"tool": cmd[0], "timeout_s": timeout_s})
        raise UndecodableAudioError("timed out reading audio") from exc
    if result.returncode != 0:
        log.warning(
            "media tool failed",
            extra={
                "tool": cmd[0],
                "returncode": result.returncode,
                "elapsed_s": round(time.monotonic() - started, 3),
                "stderr_tail": result.stderr[-_STDERR_TAIL:],
            },
        )
    return result


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class _Stream(BaseModel):
    codec_type: str | None = None
    codec_name: str | None = None
    channels: int | None = None
    sample_rate: int | None = None


class _Format(BaseModel):
    format_name: str
    duration: Annotated[float | None, BeforeValidator(_float_or_none)] = None


class _ProbeOutput(BaseModel):
    format: _Format
    streams: list[_Stream] = []


def probe(path: str | Path, *, timeout_s: float = 30) -> AudioInfo:
    """Describe the first audio stream of ``path`` as ffmpeg sees it.

    Raises ``UnsupportedMediaError`` (415) when ffprobe can't parse the file or its
    demuxer isn't allowlisted, ``NoAudioStreamError`` (422) when there is no audio
    stream, ``UndecodableAudioError`` (422) on timeout.
    """
    result = run_tool(
        [
            "ffprobe",
            "-v",
            "error",
            "-format_whitelist",
            FORMAT_WHITELIST,
            "-show_entries",
            "format=format_name,duration:stream=index,codec_type,codec_name,channels,sample_rate",
            "-of",
            "json",
            media_url(path),
        ],
        timeout_s=timeout_s,
    )
    if result.returncode != 0:
        raise UnsupportedMediaError()
    try:
        parsed = _ProbeOutput.model_validate_json(result.stdout)
    except ValidationError as exc:
        log.warning(
            "unparsable ffprobe output", extra={"stdout_tail": result.stdout[-_STDERR_TAIL:]}
        )
        raise UnsupportedMediaError() from exc
    # The first audio stream is the one decode's ``-map 0:a:0`` will pick.
    stream = next((s for s in parsed.streams if s.codec_type == "audio"), None)
    if stream is None:
        raise NoAudioStreamError()
    return AudioInfo(
        container=parsed.format.format_name,
        codec=stream.codec_name,
        channels=stream.channels or 1,
        sample_rate=stream.sample_rate,
        header_duration_s=parsed.format.duration,
    )
