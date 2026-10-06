from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from transcription.audio.probe import ALLOWED_DEMUXERS, FORMAT_WHITELIST, probe
from transcription.errors import (
    NoAudioStreamError,
    TranscriptionError,
    UndecodableAudioError,
    UnsupportedMediaError,
)

MakeAudio = Callable[..., Path]


def assert_no_path(exc: TranscriptionError) -> None:
    # Messages reach API clients: no server paths, no raw ffmpeg output.
    assert "/" not in exc.message
    assert "ffprobe" not in exc.message


def test_whitelist_is_the_allowlist() -> None:
    assert FORMAT_WHITELIST.split(",") == list(ALLOWED_DEMUXERS)
    assert not {"hls", "concat", "image2", "tty", "lavfi"} & set(ALLOWED_DEMUXERS)


def test_bytes_win_over_extension(make_audio: MakeAudio) -> None:
    path = make_audio("lies.mp3", "sine=frequency=440:duration=3", "-ac", "2", "-f", "wav")

    info = probe(path)

    assert info.container == "wav"
    assert info.codec == "pcm_s16le"
    assert info.channels == 2
    assert info.sample_rate == 44100
    assert info.header_duration_s == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("name", "extra", "container", "codec"),
    [
        ("a.wav", (), "wav", "pcm_s16le"),
        ("a.wav", ("-ar", "8000", "-c:a", "pcm_mulaw"), "wav", "pcm_mulaw"),
        ("a.aiff", (), "aiff", "pcm_s16be"),
        ("a.mp3", (), "mp3", "mp3"),
        ("a.flac", (), "flac", "flac"),
        ("a.ogg", ("-c:a", "libopus"), "ogg", "opus"),
        ("a.webm", ("-c:a", "libopus"), "matroska,webm", "opus"),
        ("a.m4a", (), "mov,mp4,m4a,3gp,3g2,mj2", "aac"),
        ("a.aac", (), "aac", "aac"),
    ],
)
def test_supported_containers(
    make_audio: MakeAudio, name: str, extra: tuple[str, ...], container: str, codec: str
) -> None:
    info = probe(make_audio(name, "sine=frequency=440:duration=2", *extra))

    assert (info.container, info.codec, info.channels) == (container, codec, 1)
    assert info.header_duration_s == pytest.approx(2.0, abs=0.05)


def test_first_audio_stream_is_described_even_after_video(make_audio: MakeAudio) -> None:
    path = make_audio(
        "clip.mkv",
        "testsrc=duration=1:size=32x32:rate=5[out0];"
        "sine=duration=1:sample_rate=8000[out1];"
        "aevalsrc=exprs=0|0:duration=1:sample_rate=48000[out2]",
        "-map", "0:0", "-map", "0:1", "-map", "0:2",
        "-c:v", "libx264", "-c:a", "flac",
    )  # fmt: skip

    info = probe(path)

    assert (info.container, info.codec, info.channels, info.sample_rate) == (
        "matroska,webm",
        "flac",
        1,
        8000,
    )


@pytest.mark.parametrize("sample", ["evil.m3u8", "evil.mp3", "evil_concat.wav", "text.wav"])
def test_hostile_samples_are_unsupported(samples_dir: Path, sample: str) -> None:
    with pytest.raises(UnsupportedMediaError) as caught:
        probe(samples_dir / sample)
    assert_no_path(caught.value)


def test_hls_playlist_is_refused_by_the_whitelist(samples_dir: Path, tmp_path: Path) -> None:
    # A well-formed playlist with a standard extension pointing at a real local file:
    # plain ffprobe opens it as hls and reads the segment.
    target = shutil.copy(samples_dir / "hello.mp3", tmp_path / "secret.mp3")
    playlist = tmp_path / "list.m3u8"
    playlist.write_text(
        f"#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXTINF:10,\n{target}\n#EXT-X-ENDLIST\n"
    )
    unguarded = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=format_name", "-of", "csv=p=0",
         str(playlist)],
        capture_output=True, text=True,
    )  # fmt: skip
    assert unguarded.stdout.strip() == "hls"  # so the refusal below is the whitelist's doing

    with pytest.raises(UnsupportedMediaError) as caught:
        probe(playlist)
    assert_no_path(caught.value)


@pytest.mark.parametrize(
    "content",
    [b"", np.random.default_rng(7).bytes(64 * 1024)],
    ids=["empty", "random-bytes"],
)
def test_garbage_is_unsupported(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "upload.wav"
    path.write_bytes(content)

    with pytest.raises(UnsupportedMediaError):
        probe(path)


def test_missing_file_is_unsupported_without_leaking_its_path(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedMediaError) as caught:
        probe(tmp_path / "nope.wav")
    assert_no_path(caught.value)


def test_video_only_has_no_audio_stream(samples_dir: Path) -> None:
    with pytest.raises(NoAudioStreamError):
        probe(samples_dir / "video_only.mp4")


@pytest.mark.parametrize("name", ["-i", "-y.wav", "http:x", "concat:a.wav", "pipe:0"])
def test_hostile_file_names_are_read_as_local_files(
    samples_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    shutil.copy(samples_dir / "hello.aiff", tmp_path / name)
    monkeypatch.chdir(tmp_path)  # relative path: the exact string an attacker controls

    assert probe(name).container == "aiff"


def test_timeout_is_undecodable(samples_dir: Path) -> None:
    with pytest.raises(UndecodableAudioError) as caught:
        probe(samples_dir / "monologue.mp3", timeout_s=0.001)
    assert_no_path(caught.value)
