from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from transcription.asr.faster_whisper import FasterWhisperEngine
from transcription.domain import SAMPLE_RATE, Segment, Word
from transcription.errors import EngineError

AUDIO = np.zeros(SAMPLE_RATE, dtype=np.float32)


@dataclass
class StubWord:
    start: float
    end: float
    word: str
    probability: float


@dataclass
class StubSegment:
    start: float
    end: float
    text: str
    avg_logprob: float = -0.2
    no_speech_prob: float = 0.01
    compression_ratio: float = 1.3
    words: list[StubWord] | None = None


@dataclass
class StubInfo:
    language: str = "en"
    language_probability: float = 0.93


@dataclass
class StubModel:
    """Mimics WhisperModel.transcribe: segments come from a lazy generator."""

    segments: list[StubSegment] = field(default_factory=list)
    info: StubInfo = field(default_factory=StubInfo)
    error: BaseException | None = None
    calls: list[tuple[Any, dict[str, Any]]] = field(default_factory=list)

    def transcribe(self, audio: Any, **kwargs: Any) -> tuple[Iterator[StubSegment], StubInfo]:
        self.calls.append((audio, kwargs))

        def decode() -> Iterator[StubSegment]:
            if self.error is not None:
                raise self.error
            yield from self.segments

        return decode(), self.info


def make_engine(model: StubModel, **kwargs: Any) -> FasterWhisperEngine:
    return FasterWhisperEngine("small", model=model, **kwargs)


def test_passes_the_decoding_contract_to_the_model() -> None:
    model = StubModel()
    engine = make_engine(
        model,
        beam_size=3,
        no_speech_threshold=0.5,
        logprob_threshold=-0.8,
        compression_ratio_threshold=2.2,
    )

    engine.transcribe(AUDIO, language="eng", prompt="Acme, Kubernetes", word_timestamps=True)

    ((audio, kwargs),) = model.calls
    assert audio is AUDIO
    assert kwargs == {
        "language": "en",
        "initial_prompt": "Acme, Kubernetes",
        "beam_size": 3,
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        "compression_ratio_threshold": 2.2,
        "log_prob_threshold": -0.8,
        "no_speech_threshold": 0.5,
        "condition_on_previous_text": False,
        "vad_filter": False,
        "word_timestamps": True,
        "without_timestamps": False,
    }


def test_maps_segments_words_and_scores() -> None:
    words = [StubWord(0.0, 0.4, " Hello", 0.9), StubWord(0.5, 0.9, " world.", 0.6)]
    model = StubModel(
        segments=[
            StubSegment(0.0, 0.9, " Hello world.", -0.25, 0.02, 1.1, words),
            StubSegment(1.0, 2.0, " Again.", words=None),
        ]
    )

    result = make_engine(model).transcribe(AUDIO, language=None, prompt=None)

    assert result.engine == "faster_whisper:small"
    assert result.segments == [
        Segment(
            start=0.0,
            end=0.9,
            text=" Hello world.",
            avg_logprob=-0.25,
            no_speech_prob=0.02,
            compression_ratio=1.1,
            words=[
                Word(start=0.0, end=0.4, text=" Hello", probability=0.9),
                Word(start=0.5, end=0.9, text=" world.", probability=0.6),
            ],
        ),
        Segment(
            start=1.0,
            end=2.0,
            text=" Again.",
            avg_logprob=-0.2,
            no_speech_prob=0.01,
            compression_ratio=1.3,
        ),
    ]


def test_detected_language_reports_its_probability() -> None:
    model = StubModel(info=StubInfo("de", 0.81))

    result = make_engine(model).transcribe(AUDIO, language=None, prompt=None)

    assert model.calls[0][1]["language"] is None
    assert (result.language, result.language_probability) == ("de", 0.81)


def test_pinned_language_has_no_probability() -> None:
    model = StubModel(info=StubInfo("de", 1.0))
    result = make_engine(model).transcribe(AUDIO, language="deu", prompt=None)
    assert (result.language, result.language_probability) == ("de", None)


@pytest.mark.parametrize(
    "error", [RuntimeError("CUDA failed"), ValueError("bad shape"), MemoryError()]
)
def test_lazy_decoding_failures_become_engine_errors(error: BaseException) -> None:
    engine = make_engine(StubModel(error=error))

    with pytest.raises(EngineError, match="faster-whisper") as info:
        engine.transcribe(AUDIO, language=None, prompt=None)

    assert info.value.__cause__ is error


def test_eager_model_failures_become_engine_errors() -> None:
    class Exploding:
        def transcribe(self, audio: Any, **kwargs: Any) -> Any:
            raise RuntimeError("out of memory")

    engine = FasterWhisperEngine("small", model=Exploding())
    with pytest.raises(EngineError, match="out of memory"):
        engine.transcribe(AUDIO, language=None, prompt=None)


@pytest.mark.parametrize(
    "audio",
    [
        np.zeros(10, dtype=np.float64),
        np.zeros(10, dtype=np.int16),
        np.zeros((2, 10), dtype=np.float32),
    ],
    ids=["float64", "int16", "stereo"],
)
def test_rejects_audio_that_is_not_mono_float32(audio: Any) -> None:
    model = StubModel()
    with pytest.raises(ValueError, match="1-D float32"):
        make_engine(model).transcribe(audio, language=None, prompt=None)
    assert model.calls == []


def test_importing_the_package_does_not_load_the_runtime() -> None:
    code = (
        "import sys, transcription.asr; "
        "print(sorted(m for m in ('faster_whisper', 'ctranslate2', 'av') if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == "[]"


@pytest.mark.slow
def test_real_model_transcribes_speech(samples_dir: Path) -> None:
    pcm = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(samples_dir / "hello.mp3"),
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "f32le",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    audio = np.frombuffer(pcm, dtype=np.float32)
    engine = FasterWhisperEngine("small", device="cpu")

    result = engine.transcribe(audio, language=None, prompt=None, word_timestamps=True)

    text = " ".join(s.text for s in result.segments).lower()
    assert "quick brown fox" in text
    assert result.language == "en"
    assert result.language_probability is not None and result.language_probability > 0.5
    assert all(s.words for s in result.segments)
    assert all(0 <= s.start <= s.end <= len(audio) / SAMPLE_RATE + 0.5 for s in result.segments)
