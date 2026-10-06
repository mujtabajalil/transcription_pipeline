from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from transcription.audio.decode import decode_to_pcm, open_pcm
from transcription.audio.vad import detect_speech
from transcription.domain import SAMPLE_RATE, PipelineConfig

CFG = PipelineConfig()
SR = SAMPLE_RATE


def vad(
    audio: npt.NDArray[np.int16], block_samples: int = CFG.vad_block_s * SR
) -> list[tuple[int, int]]:
    return detect_speech(
        audio,
        block_samples=block_samples,
        threshold=CFG.vad_threshold,
        min_silence_ms=CFG.vad_min_silence_ms,
        speech_pad_ms=CFG.vad_speech_pad_ms,
    )


def load(samples_dir: Path, tmp_path: Path, name: str) -> npt.NDArray[np.int16]:
    out = tmp_path / f"{name}.pcm"
    decode_to_pcm(samples_dir / name, out, max_seconds=600, timeout_s=60)
    return open_pcm(out)


def assert_sorted_disjoint(regions: list[tuple[int, int]], n: int) -> None:
    assert all(0 <= s < e <= n for s, e in regions)
    assert all(a[1] < b[0] for a, b in pairwise(regions))


def test_empty_audio_has_no_speech() -> None:
    assert vad(np.zeros(0, np.int16)) == []


def test_silence_has_no_speech() -> None:
    assert vad(np.zeros(10 * SR, np.int16)) == []


def test_pure_tone_is_not_speech() -> None:
    t = np.arange(10 * SR) / SR
    assert vad((np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)) == []


def test_block_samples_must_be_positive() -> None:
    with pytest.raises(ValueError, match="block_samples"):
        vad(np.zeros(SR, np.int16), block_samples=0)


def test_hello_is_mostly_speech(samples_dir: Path, tmp_path: Path) -> None:
    audio = load(samples_dir, tmp_path, "hello.mp3")

    regions = vad(audio)

    assert_sorted_disjoint(regions, len(audio))
    assert sum(e - s for s, e in regions) > 0.8 * len(audio)


def test_gaps_yields_three_regions_without_the_long_silence(
    samples_dir: Path, tmp_path: Path
) -> None:
    audio = load(samples_dir, tmp_path, "gaps.mp3")

    regions = vad(audio)

    assert len(regions) == 3, regions
    assert_sorted_disjoint(regions, len(audio))
    (_, end0), (start1, end1), (start2, _) = regions
    assert start1 - end0 > 19 * SR  # the 20 s silence, minus speech padding
    assert start2 - end1 > 2 * SR  # the 3 s silence


@pytest.mark.parametrize("block_s", [5, 3.0077, 1])
def test_speech_across_block_boundaries_is_merged(
    samples_dir: Path, tmp_path: Path, block_s: float
) -> None:
    # 40 s of nonstop speech: every block boundary falls inside a word run.
    audio = load(samples_dir, tmp_path, "monologue.mp3")

    regions = vad(audio, block_samples=int(block_s * SR))

    assert regions == vad(audio) == [(0, len(audio))]


def test_small_blocks_keep_real_pauses(samples_dir: Path, tmp_path: Path) -> None:
    audio = load(samples_dir, tmp_path, "gaps.mp3")

    small = vad(audio, block_samples=4 * SR)
    whole = vad(audio)

    assert len(small) == len(whole) == 3
    for (s1, e1), (s2, e2) in zip(small, whole, strict=True):
        assert abs(s1 - s2) < SR // 10 and abs(e1 - e2) < SR // 10
