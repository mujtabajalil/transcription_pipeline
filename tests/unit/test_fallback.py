from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
import pytest
from prometheus_client import REGISTRY
from pydantic import SecretStr

from tests.fakes import FakeEngine
from transcription.asr import factory
from transcription.asr.elevenlabs import ElevenLabsEngine
from transcription.asr.fallback import CircuitBreaker, FallbackEngine
from transcription.config import Settings
from transcription.errors import EngineError, EngineUnavailableError

AUDIO = np.zeros(16_000, dtype=np.float32)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


# --- CircuitBreaker ----------------------------------------------------------------------


def make_breaker(clock: FakeClock, threshold: int = 3) -> CircuitBreaker:
    return CircuitBreaker(failure_threshold=threshold, reset_s=60, clock=clock)


def test_breaker_opens_after_consecutive_failures(clock: FakeClock) -> None:
    breaker = make_breaker(clock)
    for _ in range(2):
        breaker.record_failure()
        assert breaker.state == "closed" and breaker.allow()

    breaker.record_failure()

    assert breaker.state == "open"
    assert not breaker.allow()


def test_success_resets_the_failure_streak(clock: FakeClock) -> None:
    breaker = make_breaker(clock)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == "closed"


def test_breaker_half_opens_after_reset_and_admits_one_trial(clock: FakeClock) -> None:
    breaker = make_breaker(clock, threshold=1)
    breaker.record_failure()

    clock.now += 59.999
    assert breaker.state == "open" and not breaker.allow()
    clock.now += 0.001
    assert breaker.state == "half_open"

    assert breaker.allow()
    assert not breaker.allow(), "only one trial while it is in flight"


def test_half_open_success_closes(clock: FakeClock) -> None:
    breaker = make_breaker(clock, threshold=1)
    breaker.record_failure()
    clock.now += 60
    assert breaker.allow()

    breaker.record_success()

    assert breaker.state == "closed"
    assert breaker.allow() and breaker.allow()


def test_half_open_failure_reopens_for_another_period(clock: FakeClock) -> None:
    breaker = make_breaker(clock, threshold=3)
    for _ in range(3):
        breaker.record_failure()
    clock.now += 60
    assert breaker.allow()

    breaker.record_failure()  # one failure is enough while half-open

    assert breaker.state == "open"
    clock.now += 59
    assert not breaker.allow()
    clock.now += 1
    assert breaker.state == "half_open"


def test_trial_that_never_reports_back_is_retried_after_reset(clock: FakeClock) -> None:
    breaker = make_breaker(clock, threshold=1)
    breaker.record_failure()
    clock.now += 60
    assert breaker.allow()

    clock.now += 60

    assert breaker.allow()


def test_concurrent_callers_get_exactly_one_trial(clock: FakeClock) -> None:
    breaker = make_breaker(clock, threshold=1)
    breaker.record_failure()
    clock.now += 60
    barrier = threading.Barrier(16)
    granted: list[bool] = []

    def call() -> None:
        barrier.wait()
        granted.append(breaker.allow())

    threads = [threading.Thread(target=call) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert granted.count(True) == 1


def test_breaker_rejects_nonpositive_threshold() -> None:
    with pytest.raises(ValueError, match="failure_threshold"):
        CircuitBreaker(failure_threshold=0, reset_s=1)


# --- FallbackEngine ----------------------------------------------------------------------


def unavailable(name: str = "primary") -> FakeEngine:
    return FakeEngine(name=name, fail_times=10**6, error=EngineUnavailableError("down"))


def make_engine(primary: FakeEngine, fallback: FakeEngine, clock: FakeClock) -> FallbackEngine:
    return FallbackEngine(primary, fallback, failure_threshold=2, reset_s=30, clock=clock)


def transcribe(engine: FallbackEngine, **kw: Any) -> str:
    return engine.transcribe(AUDIO, language="en", prompt="glossary", **kw).engine


def breaker_gauge(engine: str) -> float | None:
    return REGISTRY.get_sample_value("tx_engine_breaker_open", {"engine": engine})


def fallbacks(primary: str, fallback: str) -> float:
    labels = {"primary": primary, "fallback": fallback}
    return REGISTRY.get_sample_value("tx_engine_fallbacks_total", labels) or 0.0


def test_healthy_primary_serves(clock: FakeClock) -> None:
    primary, fallback = FakeEngine(name="p-ok"), FakeEngine(name="f-ok")
    engine = make_engine(primary, fallback, clock)

    assert engine.name == "p-ok"
    assert transcribe(engine) == "p-ok"
    assert fallback.calls == []
    assert breaker_gauge("p-ok") == 0


def test_unavailable_primary_is_served_by_fallback(
    clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    primary, fallback = unavailable("p-down"), FakeEngine(name="f-down")
    engine = make_engine(primary, fallback, clock)
    before = fallbacks("p-down", "f-down")

    with caplog.at_level(logging.WARNING, logger="transcription.asr.fallback"):
        served_by = transcribe(engine, word_timestamps=True)

    assert served_by == "f-down"
    assert fallback.calls == [
        {"samples": 16_000, "language": "en", "prompt": "glossary", "dtype": np.float32}
    ]
    assert fallbacks("p-down", "f-down") == before + 1
    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    assert (record.primary, record.fallback) == ("p-down", "f-down")  # type: ignore[attr-defined]


def test_open_breaker_skips_the_primary_until_reset(clock: FakeClock) -> None:
    primary, fallback = unavailable("p-open"), FakeEngine(name="f-open")
    engine = make_engine(primary, fallback, clock)
    before = fallbacks("p-open", "f-open")

    for _ in range(4):
        assert transcribe(engine) == "f-open"

    assert len(primary.calls) == 2, "breaker opened after failure_threshold=2"
    assert breaker_gauge("p-open") == 1
    assert fallbacks("p-open", "f-open") == before + 4

    primary.fail_times = 0  # the primary recovers
    clock.now += 30
    assert transcribe(engine) == "p-open"
    assert len(primary.calls) == 3
    assert breaker_gauge("p-open") == 0


def test_failed_trial_keeps_serving_from_fallback(clock: FakeClock) -> None:
    primary, fallback = unavailable("p-trial"), FakeEngine(name="f-trial")
    engine = make_engine(primary, fallback, clock)
    transcribe(engine)
    transcribe(engine)

    clock.now += 30
    assert transcribe(engine) == "f-trial"
    assert transcribe(engine) == "f-trial"

    assert len(primary.calls) == 3, "one half-open trial, then open again"
    assert breaker_gauge("p-trial") == 1


def test_plain_engine_error_propagates_without_fallback(clock: FakeClock) -> None:
    primary = FakeEngine(name="p-err", fail_times=10, error=EngineError("bad chunk"))
    fallback = FakeEngine(name="f-err")
    engine = make_engine(primary, fallback, clock)

    for _ in range(5):
        with pytest.raises(EngineError, match="bad chunk"):
            transcribe(engine)

    assert fallback.calls == []
    assert len(primary.calls) == 5, "non-availability errors never open the breaker"
    assert breaker_gauge("p-err") == 0


def test_engine_error_breaks_the_unavailability_streak(clock: FakeClock) -> None:
    primary = FakeEngine(name="p-mix", fail_times=10, error=EngineUnavailableError("down"))
    engine = make_engine(primary, FakeEngine(name="f-mix"), clock)
    transcribe(engine)
    primary.error = EngineError("bad chunk")
    with pytest.raises(EngineError):
        transcribe(engine)
    primary.error = EngineUnavailableError("down")

    transcribe(engine)

    assert breaker_gauge("p-mix") == 0


def test_fallback_failure_propagates(clock: FakeClock) -> None:
    engine = make_engine(unavailable("p-both"), unavailable("f-both"), clock)
    with pytest.raises(EngineUnavailableError):
        transcribe(engine)


# --- factory -----------------------------------------------------------------------------


class StubWhisper:
    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.name = f"faster_whisper:{model_name}"
        self.kwargs = kwargs


@pytest.fixture
def stub_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory, "FasterWhisperEngine", StubWhisper)


@pytest.mark.usefixtures("stub_whisper")
def test_factory_builds_whisper_from_settings(settings: Settings) -> None:
    engine = factory.build_engine(
        settings.model_copy(update={"asr_fallback": "faster_whisper", "whisper_beam_size": 2})
    )

    assert isinstance(engine, StubWhisper), "whisper is never its own fallback"
    assert engine.name == "faster_whisper:small"
    assert engine.kwargs["beam_size"] == 2
    assert engine.kwargs["compression_ratio_threshold"] == settings.compression_ratio_threshold


@pytest.mark.usefixtures("stub_whisper")
def test_factory_wraps_elevenlabs_with_whisper_fallback(settings: Settings) -> None:
    engine = factory.build_engine(
        settings.model_copy(
            update={
                "asr_engine": "elevenlabs",
                "asr_fallback": "faster_whisper",
                "elevenlabs_api_key": SecretStr("k"),
            }
        )
    )

    assert isinstance(engine, FallbackEngine)
    assert engine.name == "elevenlabs:scribe_v1"


def test_factory_builds_bare_elevenlabs(settings: Settings) -> None:
    engine = factory.build_engine(
        settings.model_copy(
            update={"asr_engine": "elevenlabs", "elevenlabs_api_key": SecretStr("k")}
        )
    )
    assert isinstance(engine, ElevenLabsEngine)


@pytest.mark.parametrize("key", [None, SecretStr("")])
def test_factory_requires_an_elevenlabs_key(settings: Settings, key: SecretStr | None) -> None:
    config = settings.model_copy(update={"asr_engine": "elevenlabs", "elevenlabs_api_key": key})
    with pytest.raises(ValueError, match="TX_ELEVENLABS_API_KEY"):
        factory.build_engine(config)
