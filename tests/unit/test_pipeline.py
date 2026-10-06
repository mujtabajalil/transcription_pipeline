from __future__ import annotations

import logging
import subprocess
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from prometheus_client import REGISTRY

from tests.fakes import FakeEngine
from transcription.asr.base import Audio
from transcription.domain import (
    SAMPLE_RATE,
    ChunkTranscript,
    PipelineConfig,
    Transcript,
    TranscriptionOptions,
)
from transcription.errors import (
    EngineError,
    EngineUnavailableError,
    LeaseLostError,
    UndecodableAudioError,
    UnsupportedMediaError,
)
from transcription.pipeline import InMemoryCheckpoint, ProgressCallback, transcribe_file

CONFIG = PipelineConfig(chunk_retry_base_delay_s=0.0)


class Crash(Exception):
    """Stands in for a worker dying mid-job."""


def run(
    path: Path,
    engine: FakeEngine,
    *,
    options: TranscriptionOptions | None = None,
    config: PipelineConfig = CONFIG,
    checkpoint: InMemoryCheckpoint | None = None,
    on_progress: ProgressCallback | None = None,
    workdir: Path | None = None,
) -> Transcript:
    return transcribe_file(
        path,
        engine,
        options or TranscriptionOptions(),
        config,
        checkpoint=checkpoint,
        on_progress=on_progress,
        workdir=workdir,
    )


def texts(t: Transcript) -> list[str]:
    return [s.text for s in t.segments]


def metric(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def without_timing(t: Transcript) -> dict[str, Any]:
    return t.model_dump(exclude={"stats": {"processing_seconds", "realtime_factor"}})


@pytest.fixture
def gaps(samples_dir: Path) -> Path:
    """speech / 20 s silence / speech / 3 s silence / speech: three chunks."""
    return samples_dir / "gaps.mp3"


# --- chunking end to end -----------------------------------------------------------------


def test_long_silences_separate_chunks(gaps: Path) -> None:
    t = run(gaps, FakeEngine())

    assert texts(t) == ["chunk 0", "chunk 1", "chunk 2"]
    assert [s.start for s in t.segments] == pytest.approx([0.0, 22.4, 28.2], abs=0.3)
    assert t.text == "chunk 0 chunk 1 chunk 2"
    assert t.duration_s == t.stats.audio_seconds == pytest.approx(30.5, abs=0.2)
    assert t.audio.container == "mp3"
    stats = t.stats
    assert (stats.chunks, stats.forced_cuts, stats.engines) == (3, 0, {"fake": 3})
    assert stats.speech_seconds == pytest.approx(8.1, abs=0.5)
    # FakeEngine's segment spans its whole chunk.
    assert stats.transcribed_seconds == pytest.approx(sum(s.end - s.start for s in t.segments))
    assert stats.processing_seconds is not None and stats.processing_seconds > 0
    assert stats.realtime_factor is not None and stats.realtime_factor > 0


def test_nonstop_speech_is_cut_into_contiguous_chunks(samples_dir: Path) -> None:
    before = metric("tx_forced_cuts_total")

    t = run(samples_dir / "monologue.mp3", FakeEngine())

    assert (t.stats.chunks, t.stats.forced_cuts) == (2, 1)
    first, second = t.segments
    assert 25 <= first.end <= 30
    assert first.end == second.start
    assert metric("tx_forced_cuts_total") - before == 1


@pytest.mark.parametrize("name", ["monologue.mp3", "gaps.mp3"])
def test_engine_sees_at_most_30_s_of_float32(samples_dir: Path, name: str) -> None:
    engine = FakeEngine()
    run(samples_dir / name, engine)
    assert engine.calls
    assert all(0 < c["samples"] <= 30 * SAMPLE_RATE for c in engine.calls)  # type: ignore[operator]
    assert all(c["dtype"] == np.float32 for c in engine.calls)


def test_split_channels_interleaves_speakers_by_time(samples_dir: Path, tmp_path: Path) -> None:
    # Swap channels so the speaker on channel 1 talks first: timeline order must win
    # over chunk (channel-major) order.
    swapped = tmp_path / "swapped.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(samples_dir / "call_stereo.wav"),
            "-af",
            "pan=stereo|c0=c1|c1=c0",
            str(swapped),
        ],
        check=True,
    )

    t = run(swapped, FakeEngine(), options=TranscriptionOptions(split_channels=True))

    assert [s.channel for s in t.segments] == [1, 0]
    assert texts(t) == ["chunk 1", "chunk 0"]
    assert t.segments[0].end <= t.segments[1].start
    assert t.stats.chunks == 2
    assert t.stats.speech_seconds == pytest.approx(6.4, abs=0.5)


def test_stereo_is_downmixed_unless_split(samples_dir: Path) -> None:
    t = run(samples_dir / "call_stereo.wav", FakeEngine())
    assert {s.channel for s in t.segments} == {None}


def test_silence_yields_empty_transcript_without_engine_calls(make_audio: Any) -> None:
    path = make_audio("silence.wav", "anullsrc=r=16000:cl=mono", "-t", "5")
    engine = FakeEngine()
    progress: list[tuple[int, int]] = []

    t = run(path, engine, on_progress=lambda done, total: progress.append((done, total)))

    assert engine.calls == []
    assert (t.text, t.segments, t.language) == ("", [], None)
    assert (t.stats.chunks, t.stats.speech_seconds, t.stats.engines) == (0, 0, {})
    assert t.duration_s == pytest.approx(5.0)
    assert progress == [(0, 0)]


def test_non_audio_is_rejected_before_any_work(samples_dir: Path) -> None:
    engine = FakeEngine()
    with pytest.raises(UnsupportedMediaError):
        run(samples_dir / "text.wav", engine)
    assert engine.calls == []


def test_decoded_pcm_goes_to_the_given_workdir(gaps: Path, tmp_path: Path) -> None:
    t = run(gaps, FakeEngine(), workdir=tmp_path)
    (pcm,) = tmp_path.glob("*.pcm")
    assert pcm.stat().st_size == 2 * round(t.duration_s * SAMPLE_RATE)


@pytest.mark.parametrize("fails", [False, True])
def test_temporary_pcm_is_removed_without_a_workdir(
    gaps: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fails: bool
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    decoded: list[Path] = []
    engine = FakeEngine(
        fail_times=1 if fails else 0,
        error=RuntimeError("bug"),
        on_call=lambda _: decoded.extend(tmp_path.rglob("*.pcm")),
    )

    with pytest.raises(RuntimeError) if fails else nullcontext():
        run(gaps, engine)

    assert decoded  # the PCM really was in the temporary directory...
    assert list(tmp_path.iterdir()) == []  # ...which is gone, even after a failure


def test_stats_aggregate_quality_guard_outcomes(gaps: Path) -> None:
    flagged = run(
        gaps, FakeEngine(), config=CONFIG.model_copy(update={"low_confidence_logprob": 0})
    )
    assert flagged.stats.low_confidence_segments == 3
    assert all(s.low_confidence for s in flagged.segments)

    # FakeEngine reports compression_ratio=1.2: above this threshold every segment is a loop.
    dropped = run(
        gaps, FakeEngine(), config=CONFIG.model_copy(update={"compression_ratio_threshold": 1})
    )
    assert (dropped.segments, dropped.text) == ([], "")
    assert dropped.stats.dropped_segments == {"repetition": 3}


# --- language and prompt -----------------------------------------------------------------


def test_language_is_detected_once_then_pinned(gaps: Path) -> None:
    engine = FakeEngine(language="eng")  # hosted engines answer in ISO 639-3
    checkpoint = InMemoryCheckpoint()

    t = run(gaps, engine, checkpoint=checkpoint)

    assert [c["language"] for c in engine.calls] == [None, "en", "en"]
    assert (t.language, t.language_probability) == ("en", 0.99)
    assert checkpoint.language == ("en", 0.99)


def test_requested_language_is_pinned_from_the_first_chunk(gaps: Path) -> None:
    engine = FakeEngine()
    t = run(gaps, engine, options=TranscriptionOptions(language="deu"))
    assert [c["language"] for c in engine.calls] == ["de", "de", "de"]
    assert (t.language, t.language_probability) == ("de", None)


def test_checkpointed_language_is_reused(gaps: Path) -> None:
    checkpoint = InMemoryCheckpoint()
    checkpoint.save_language("fr", 0.8)
    engine = FakeEngine()

    t = run(gaps, engine, checkpoint=checkpoint)

    assert [c["language"] for c in engine.calls] == ["fr", "fr", "fr"]
    assert (t.language, t.language_probability) == ("fr", 0.8)


def test_prompt_is_passed_on_every_call(gaps: Path) -> None:
    engine = FakeEngine()
    run(gaps, engine, options=TranscriptionOptions(prompt="Kenobi, Coruscant"))
    assert [c["prompt"] for c in engine.calls] == ["Kenobi, Coruscant"] * 3


class WordTimestampsSpy(FakeEngine):
    """FakeEngine does not record ``word_timestamps``."""

    def __init__(self) -> None:
        super().__init__()
        self.word_timestamps: list[bool] = []

    def transcribe(
        self,
        audio: Audio,
        *,
        language: str | None,
        prompt: str | None,
        word_timestamps: bool = False,
    ) -> ChunkTranscript:
        self.word_timestamps.append(word_timestamps)
        return super().transcribe(audio, language=language, prompt=prompt)


@pytest.mark.parametrize("requested", [True, False])
def test_word_timestamps_option_is_passed_on_every_call(gaps: Path, requested: bool) -> None:
    engine = WordTimestampsSpy()
    run(gaps, engine, options=TranscriptionOptions(word_timestamps=requested))
    assert engine.word_timestamps == [requested] * 3


# --- retries -----------------------------------------------------------------------------


def test_retryable_failures_are_retried_within_the_job(gaps: Path) -> None:
    engine = FakeEngine(fail_times=2, name="flaky")
    before = {o: metric("tx_chunks_total", engine="flaky", outcome=o) for o in ("ok", "retry")}

    t = run(gaps, engine)

    assert len(engine.calls) == 3 + 2
    assert texts(t) == ["chunk 2", "chunk 3", "chunk 4"]
    assert [c["language"] for c in engine.calls] == [None, None, None, "en", "en"]
    assert metric("tx_chunks_total", engine="flaky", outcome="retry") - before["retry"] == 2
    assert metric("tx_chunks_total", engine="flaky", outcome="ok") - before["ok"] == 3


@pytest.mark.parametrize("error", [RuntimeError("bug"), UndecodableAudioError()])
def test_non_retryable_errors_propagate_immediately(gaps: Path, error: Exception) -> None:
    engine = FakeEngine(fail_times=1, error=error, name="broken")
    before = metric("tx_chunks_total", engine="broken", outcome="error")

    with pytest.raises(type(error)):
        run(gaps, engine)

    assert len(engine.calls) == 1
    assert metric("tx_chunks_total", engine="broken", outcome="error") - before == 1


@pytest.mark.parametrize("error", [EngineError("bad chunk"), EngineUnavailableError("down")])
def test_exhausted_retries_raise_the_engine_error(gaps: Path, error: EngineError) -> None:
    engine = FakeEngine(fail_times=1_000, error=error, name="dead")
    config = CONFIG.model_copy(update={"chunk_max_retries": 2})

    with pytest.raises(type(error)):
        run(gaps, engine, config=config)

    assert len(engine.calls) == 1 + 2


def test_retry_backoff_is_exponential_with_full_jitter(
    gaps: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jitter_ranges: list[tuple[float, float]] = []
    sleeps: list[float] = []

    def upper_bound(low: float, high: float) -> float:
        jitter_ranges.append((low, high))
        return high

    monkeypatch.setattr("transcription.pipeline.random.uniform", upper_bound)
    monkeypatch.setattr("transcription.pipeline.time.sleep", sleeps.append)
    config = CONFIG.model_copy(update={"chunk_retry_base_delay_s": 0.5, "chunk_max_retries": 3})

    run(gaps, FakeEngine(fail_times=3), config=config)

    assert jitter_ranges == [(0, 0.5), (0, 1.0), (0, 2.0)]
    assert sleeps == [0.5, 1.0, 2.0]


def test_zero_base_delay_retries_without_sleeping(
    gaps: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("transcription.pipeline.time.sleep", sleeps.append)
    run(gaps, FakeEngine(fail_times=2), config=CONFIG)
    assert sleeps == []


# --- progress, cancellation, resume ------------------------------------------------------


def test_progress_is_reported_before_work_and_after_each_chunk(gaps: Path) -> None:
    progress: list[tuple[int, int]] = []
    run(gaps, FakeEngine(), on_progress=lambda done, total: progress.append((done, total)))
    assert progress == [(0, 3), (1, 3), (2, 3), (3, 3)]


def test_progress_callback_cancels_after_a_checkpointed_chunk(gaps: Path) -> None:
    def lose_lease(done: int, total: int) -> None:
        if done == 1:
            raise LeaseLostError()  # retryable, but must not be retried as a chunk failure

    engine = FakeEngine()
    checkpoint = InMemoryCheckpoint()

    with pytest.raises(LeaseLostError):
        run(gaps, engine, checkpoint=checkpoint, on_progress=lose_lease)

    assert len(engine.calls) == 1
    assert list(checkpoint.results) == [0]


def test_resume_transcribes_only_missing_chunks(gaps: Path) -> None:
    scratch_engine = FakeEngine()
    scratch = run(gaps, scratch_engine)

    def crash_on_third_call(n: int) -> None:
        if n == 2:
            raise Crash

    checkpoint = InMemoryCheckpoint()
    with pytest.raises(Crash):
        run(gaps, FakeEngine(on_call=crash_on_third_call), checkpoint=checkpoint)
    assert sorted(checkpoint.results) == [0, 1]

    # Continue FakeEngine's call numbering so texts match the uninterrupted run.
    engine = FakeEngine()
    engine.calls.extend(scratch_engine.calls[:2])
    progress: list[tuple[int, int]] = []
    resumed = run(
        gaps,
        engine,
        checkpoint=checkpoint,
        on_progress=lambda done, total: progress.append((done, total)),
    )

    assert engine.calls[2:] == scratch_engine.calls[2:]  # one call: chunk 2, language pinned
    assert progress == [(2, 3), (3, 3)]
    assert without_timing(resumed) == without_timing(scratch)


def test_plan_for_different_audio_is_replaced_and_its_results_ignored(
    gaps: Path, caplog: pytest.LogCaptureFixture
) -> None:
    previous = InMemoryCheckpoint()
    run(gaps, FakeEngine(prefix="stale"), checkpoint=previous)
    assert previous.plan is not None
    checkpoint = InMemoryCheckpoint()
    checkpoint.save_plan(
        previous.plan.model_copy(update={"audio_samples": previous.plan.audio_samples + 1})
    )
    for result in previous.results.values():
        checkpoint.save_result(result)
    engine = FakeEngine()

    with caplog.at_level(logging.WARNING, logger="transcription.pipeline"):
        t = run(gaps, engine, checkpoint=checkpoint)

    assert len(engine.calls) == 3
    assert texts(t) == ["chunk 0", "chunk 1", "chunk 2"]
    assert checkpoint.plan == previous.plan
    assert "replanning" in caplog.text


def test_checkpointed_result_for_other_chunk_bounds_is_redone(gaps: Path) -> None:
    checkpoint = InMemoryCheckpoint()
    run(gaps, FakeEngine(prefix="old"), checkpoint=checkpoint)
    # A row left over from an earlier plan: same index, different audio.
    checkpoint.save_result(checkpoint.results[1].model_copy(update={"start_s": 0.0}))
    engine = FakeEngine(prefix="new")

    t = run(gaps, engine, checkpoint=checkpoint)

    assert len(engine.calls) == 1
    assert texts(t) == ["old 0", "new 0", "old 2"]
