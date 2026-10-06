from __future__ import annotations

import contextlib
import os
import shutil
import struct
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from transcription.audio.decode import decode_to_pcm, open_pcm, to_float32
from transcription.domain import SAMPLE_RATE
from transcription.errors import AudioTooLongError, TranscriptionError, UndecodableAudioError

MakeAudio = Callable[..., Path]

SINE_2S = "sine=frequency=440:duration=2"


def decode(
    path: Path,
    out: Path,
    *,
    channel: int | None = None,
    max_seconds: int = 60,
    timeout_s: float = 60,
) -> int:
    return decode_to_pcm(path, out, channel=channel, max_seconds=max_seconds, timeout_s=timeout_s)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(to_float32(x) ** 2)))


def assert_no_path(exc: TranscriptionError) -> None:
    assert "/" not in exc.message
    assert "ffmpeg" not in exc.message


@pytest.mark.parametrize(
    ("name", "lavfi", "extra"),
    [
        ("pcm.wav", SINE_2S, ("-ac", "2")),
        ("ulaw.wav", SINE_2S + ":sample_rate=8000", ("-c:a", "pcm_mulaw")),
        ("a.aiff", SINE_2S, ()),
        ("a.mp3", SINE_2S, ()),
        ("a.flac", SINE_2S, ()),
        ("a.ogg", SINE_2S, ("-c:a", "libopus")),
        ("a.webm", SINE_2S, ("-c:a", "libopus")),
        ("a.m4a", SINE_2S, ()),
        (
            "video.mp4",
            "testsrc=duration=2:size=32x32:rate=5[out0];" + SINE_2S + "[out1]",
            ("-c:v", "libx264", "-c:a", "aac"),
        ),
    ],
)
def test_every_container_decodes_to_exact_length(
    make_audio: MakeAudio, tmp_path: Path, name: str, lavfi: str, extra: tuple[str, ...]
) -> None:
    out = tmp_path / "out.pcm"

    samples = decode(make_audio(name, lavfi, *extra), out)

    assert samples == 2 * SAMPLE_RATE
    audio = open_pcm(out)
    assert len(audio) == samples
    assert rms(audio) > 0.05  # the tone survived (lavfi sine RMS is 0.088 before downmix)


def test_raw_adts_aac_decodes(make_audio: MakeAudio, tmp_path: Path) -> None:
    # No gapless metadata in raw ADTS, so the encoder's priming frame stays in.
    samples = decode(make_audio("a.aac", SINE_2S), tmp_path / "out.pcm")

    assert 2 * SAMPLE_RATE <= samples <= 2 * SAMPLE_RATE + 2048


def test_bytes_win_over_extension(make_audio: MakeAudio, tmp_path: Path) -> None:
    path = make_audio("lies.mp3", SINE_2S, "-f", "wav")

    assert decode(path, tmp_path / "out.pcm") == 2 * SAMPLE_RATE


@pytest.mark.parametrize("sample", ["hello.mp3", "hello.aiff", "hello.m4a", "hello.webm"])
def test_samples_agree_on_duration(samples_dir: Path, tmp_path: Path, sample: str) -> None:
    samples = decode(samples_dir / sample, tmp_path / "out.pcm")

    assert samples / SAMPLE_RATE == pytest.approx(6.544, abs=0.01)


def test_split_channels_pick_the_requested_channel(make_audio: MakeAudio, tmp_path: Path) -> None:
    path = make_audio("lr.wav", "aevalsrc=exprs=0.5*sin(2*PI*440*t)|0:duration=1")

    left, right, mixed = (tmp_path / f"{n}.pcm" for n in ("left", "right", "mixed"))
    assert decode(path, left, channel=0) == SAMPLE_RATE
    assert decode(path, right, channel=1) == SAMPLE_RATE
    assert decode(path, mixed) == SAMPLE_RATE

    assert rms(open_pcm(left)) == pytest.approx(0.5 / np.sqrt(2), rel=0.02)
    assert rms(open_pcm(right)) < 1e-4
    assert 0.05 < rms(open_pcm(mixed)) < rms(open_pcm(left))


def test_negative_channel_is_rejected(make_audio: MakeAudio, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="channel"):
        decode(make_audio("a.wav", SINE_2S), tmp_path / "out.pcm", channel=-1)


def test_exactly_max_seconds_is_accepted(make_audio: MakeAudio, tmp_path: Path) -> None:
    assert decode(make_audio("a.flac", SINE_2S), tmp_path / "o.pcm", max_seconds=2) == 32000


def test_longer_than_max_seconds_is_too_long_and_leaves_no_partial_output(
    make_audio: MakeAudio, tmp_path: Path
) -> None:
    out = tmp_path / "out.pcm"

    with pytest.raises(AudioTooLongError):
        decode(make_audio("long.flac", "sine=duration=20"), out, max_seconds=2)

    assert not out.exists()


def test_endless_input_stops_at_the_cap_instead_of_the_timeout(tmp_path: Path) -> None:
    # A WAV stream that never ends: without -t max_seconds+1 ffmpeg would read it until
    # the timeout (undecodable) instead of failing fast as too long.
    fifo = tmp_path / "endless.wav"
    os.mkfifo(fifo)
    header = (
        b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, SAMPLE_RATE, 2 * SAMPLE_RATE, 2, 16)
        + b"data" + struct.pack("<I", 0xFFFFFFFF)
    )  # fmt: skip
    stop = threading.Event()

    def feed() -> None:
        with contextlib.suppress(BrokenPipeError), fifo.open("wb") as f:
            f.write(header)
            while not stop.wait(0.001):  # <= 1000x real time, so only the cap ends it early
                f.write(bytes(2 * SAMPLE_RATE))

    writer = threading.Thread(target=feed, daemon=True)
    writer.start()
    try:
        with pytest.raises(AudioTooLongError):
            decode(fifo, tmp_path / "out.pcm", max_seconds=2, timeout_s=5)
    finally:
        stop.set()
        os.close(os.open(fifo, os.O_RDONLY | os.O_NONBLOCK))  # frees a writer blocked in open
        writer.join(5)


def test_timeout_kills_ffmpeg(make_audio: MakeAudio, tmp_path: Path) -> None:
    path = make_audio("long.wav", "sine=duration=900:sample_rate=8000")
    out = tmp_path / "timeout-marker.pcm"

    with pytest.raises(UndecodableAudioError) as caught:
        decode(path, out, max_seconds=3600, timeout_s=0.01)

    assert_no_path(caught.value)
    still_running = subprocess.run(["pgrep", "-f", out.name], capture_output=True)
    assert still_running.returncode == 1, still_running.stdout
    assert not out.exists()


def test_zero_samples_is_undecodable(make_audio: MakeAudio, tmp_path: Path) -> None:
    path = make_audio("empty.wav", "anullsrc", "-t", "0")
    out = tmp_path / "out.pcm"

    with pytest.raises(UndecodableAudioError, match="no decodable audio"):
        decode(path, out)
    assert not out.exists()


@pytest.mark.parametrize(
    "sample", ["evil.m3u8", "evil.mp3", "evil_concat.wav", "text.wav", "video_only.mp4"]
)
def test_unreadable_inputs_are_undecodable(samples_dir: Path, tmp_path: Path, sample: str) -> None:
    out = tmp_path / "out.pcm"
    out.write_bytes(b"stale output from an earlier attempt")

    with pytest.raises(UndecodableAudioError) as caught:
        decode(samples_dir / sample, out)
    assert_no_path(caught.value)
    assert not out.exists()


def test_output_over_the_input_is_refused_and_the_input_survives(
    make_audio: MakeAudio, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = make_audio("a.wav", SINE_2S)
    size = path.stat().st_size
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="input"):
        decode(path, Path("a.wav"))
    assert path.stat().st_size == size


def test_concat_script_cannot_pull_in_local_files(make_audio: MakeAudio, tmp_path: Path) -> None:
    make_audio("secret.wav", SINE_2S)
    script = tmp_path / "upload.wav"
    script.write_text("ffconcat version 1.0\nfile 'secret.wav'\n")
    # Without the whitelist ffmpeg happily follows the script.
    unguarded = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(script), "-f", "s16le", str(tmp_path / "x")],
        capture_output=True,
    )
    assert unguarded.returncode == 0

    with pytest.raises(UndecodableAudioError):
        decode(script, tmp_path / "out.pcm")


def test_hostile_file_names(
    samples_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shutil.copy(samples_dir / "hello.aiff", tmp_path / "-i")
    monkeypatch.chdir(tmp_path)

    samples = decode(Path("-i"), Path("http:out"))

    assert (tmp_path / "http:out").stat().st_size == samples * 2


def test_open_pcm_is_a_read_only_int16_map(tmp_path: Path) -> None:
    path = tmp_path / "x.pcm"
    np.array([0, 1, -1, 32767, -32768], dtype=np.int16).tofile(path)

    audio = open_pcm(path)

    assert isinstance(audio, np.memmap)
    assert audio.dtype == np.int16
    assert audio.tolist() == [0, 1, -1, 32767, -32768]
    with pytest.raises(ValueError, match="read-only"):
        audio[0] = 5


def test_open_pcm_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "x.pcm"
    path.touch()

    with pytest.raises(UndecodableAudioError):
        open_pcm(path)


def test_to_float32_scales_a_memmap_slice(tmp_path: Path) -> None:
    path = tmp_path / "x.pcm"
    np.array([7, -32768, 16384, 0, 32767, 7], dtype=np.int16).tofile(path)

    out = to_float32(open_pcm(path)[1:5])

    assert out.dtype == np.float32
    assert not isinstance(out, np.memmap)
    assert out.tolist() == [-1.0, 0.5, 0.0, pytest.approx(32767 / 32768)]
