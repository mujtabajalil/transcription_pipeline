from __future__ import annotations

import io
import json
import logging
import math
import wave
from collections.abc import Iterator
from email.message import EmailMessage
from email.parser import BytesParser
from email.policy import HTTP
from typing import Any

import httpx
import numpy as np
import pytest

from transcription.asr.elevenlabs import ElevenLabsEngine
from transcription.domain import SAMPLE_RATE
from transcription.errors import EngineError, EngineUnavailableError

URL = "https://scribe.test/v1/speech-to-text"
AUDIO = (np.arange(-800, 800, dtype=np.int16) * 40 / np.float32(32768)).astype(np.float32)


def word(text: str, start: float, end: float, logprob: float | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"text": text, "start": start, "end": end, "type": "word"}
    if logprob is not None:
        item["logprob"] = logprob
    return item


def space(at: float) -> dict[str, Any]:
    return {"text": " ", "start": at, "end": at, "type": "spacing"}


def body(*words: dict[str, Any], language: str | None = "eng") -> dict[str, Any]:
    text = "".join(w["text"] for w in words)
    return {"language_code": language, "language_probability": 0.97, "text": text, "words": words}


class Scribe:
    """The Scribe endpoint, served to the engine through ``httpx.MockTransport``."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self._reply: httpx.Response | Exception = httpx.Response(200, json=body())

    def respond(self, status: int = 200, **kwargs: Any) -> None:
        self._reply = httpx.Response(status, **kwargs)

    def fail(self, error: Exception) -> None:
        self._reply = error

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert (request.method, str(request.url)) == ("POST", URL)
        self.requests.append(request)
        if isinstance(self._reply, Exception):
            raise self._reply
        return self._reply


@pytest.fixture
def scribe() -> Scribe:
    return Scribe()


@pytest.fixture
def engine(scribe: Scribe) -> Iterator[ElevenLabsEngine]:
    with httpx.Client(transport=httpx.MockTransport(scribe)) as client:
        yield ElevenLabsEngine("secret-key", base_url="https://scribe.test/", client=client)


def form_fields(request: httpx.Request) -> dict[str, EmailMessage]:
    head = f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode()
    message = BytesParser(policy=HTTP).parsebytes(head + request.content)
    assert isinstance(message, EmailMessage)
    return {
        str(part.get_param("name", header="content-disposition")): part
        for part in message.iter_parts()
        if isinstance(part, EmailMessage)
    }


def text_fields(request: httpx.Request) -> dict[str, str]:
    return {
        name: part.get_content().strip()
        for name, part in form_fields(request).items()
        if name != "file"
    }


# --- request -----------------------------------------------------------------------------


def test_request_shape(engine: ElevenLabsEngine, scribe: Scribe) -> None:
    engine.transcribe(AUDIO, language="en-US", prompt="Acme, Kubernetes")

    request = scribe.requests[-1]
    assert request.headers["xi-api-key"] == "secret-key"
    assert text_fields(request) == {
        "model_id": "scribe_v1",
        "language_code": "en",
        "timestamps_granularity": "word",
        "diarize": "false",
        "tag_audio_events": "false",
    }


def test_audio_is_sent_as_exact_16khz_mono_pcm16_wav(
    engine: ElevenLabsEngine, scribe: Scribe
) -> None:
    engine.transcribe(AUDIO, language=None, prompt=None)

    upload = form_fields(scribe.requests[-1])["file"]
    assert upload.get_content_type() == "audio/wav"
    payload = upload.get_payload(decode=True)
    assert isinstance(payload, bytes)
    with wave.open(io.BytesIO(payload)) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (SAMPLE_RATE, 1, 2)
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    np.testing.assert_array_equal(samples, np.arange(-800, 800, dtype=np.int16) * 40)


def test_out_of_range_samples_are_clipped(engine: ElevenLabsEngine, scribe: Scribe) -> None:
    engine.transcribe(np.array([-2.0, 1.0, 2.0], dtype=np.float32), language=None, prompt=None)

    payload = form_fields(scribe.requests[-1])["file"].get_payload(decode=True)
    assert isinstance(payload, bytes)
    with wave.open(io.BytesIO(payload)) as wav:
        samples = np.frombuffer(wav.readframes(3), dtype="<i2")
    assert samples.tolist() == [-32768, 32767, 32767]


def test_detection_omits_language_code(engine: ElevenLabsEngine, scribe: Scribe) -> None:
    engine.transcribe(AUDIO, language=None, prompt=None)
    assert "language_code" not in text_fields(scribe.requests[-1])


@pytest.mark.parametrize(
    "audio",
    [np.zeros((2, 10), np.float32), np.zeros(10, np.int16)],
    ids=["stereo", "int16"],  # int16 * 32768 would silently overflow into noise
)
def test_rejects_audio_that_is_not_mono_float32(engine: ElevenLabsEngine, audio: Any) -> None:
    with pytest.raises(ValueError, match="1-D float32"):
        engine.transcribe(audio, language=None, prompt=None)


# --- response ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reported", "expected"), [("eng", "en"), ("cmn", "zh"), ("en-US", "en"), (None, None)]
)
def test_language_is_normalized(
    engine: ElevenLabsEngine,
    scribe: Scribe,
    reported: str | None,
    expected: str | None,
) -> None:
    scribe.respond(json=body(language=reported))

    result = engine.transcribe(AUDIO, language=None, prompt=None)

    assert (result.language, result.language_probability) == (expected, 0.97)
    assert result.engine == engine.name == "elevenlabs:scribe_v1"


def test_pinned_language_has_no_probability(engine: ElevenLabsEngine, scribe: Scribe) -> None:
    scribe.respond(json=body(language="deu"))
    result = engine.transcribe(AUDIO, language="de-DE", prompt=None)
    assert (result.language, result.language_probability) == ("de", None)


def test_segments_split_at_sentence_end_and_long_gaps(
    engine: ElevenLabsEngine, scribe: Scribe
) -> None:
    scribe.respond(
        json=body(
            word("Hello", 0.0, 0.4, -0.1),
            space(0.4),
            word("there.", 0.5, 0.9, -0.3),
            space(0.9),
            word("How", 1.0, 1.2),
            space(1.2),
            {"text": "(laughs)", "start": 1.2, "end": 1.5, "type": "audio_event"},
            word("are", 2.0, 2.2),  # gap of exactly 0.8 s: same segment
            space(2.2),
            word("you", 3.01, 3.2, -0.5),  # gap > 0.8 s: new segment
            word("\uff1f", 3.2, 3.25, -0.7),
            space(3.25),
            word("Fine", 3.3, 3.6, -0.2),
            space(3.6),
        )
    )

    segments = engine.transcribe(AUDIO, language="en", prompt=None).segments

    assert [(s.text, s.start, s.end) for s in segments] == [
        ("Hello there.", 0.0, 0.9),
        ("How are", 1.0, 2.2),
        ("you\uff1f", 3.01, 3.25),
        ("Fine", 3.3, 3.6),
    ]
    assert [s.avg_logprob for s in segments] == [pytest.approx(-0.2), None, -0.6, -0.2]
    assert all(s.words is None for s in segments)
    assert all(s.no_speech_prob is None and s.compression_ratio is None for s in segments)


def test_word_timestamps_carry_probabilities(engine: ElevenLabsEngine, scribe: Scribe) -> None:
    scribe.respond(json=body(word("Hi", 0.1, 0.3, -0.5), space(0.3), word("x", 0.4, 0.5)))

    (segment,) = engine.transcribe(AUDIO, language=None, prompt=None, word_timestamps=True).segments

    assert segment.words is not None
    assert [(w.text, w.start, w.end) for w in segment.words] == [("Hi", 0.1, 0.3), ("x", 0.4, 0.5)]
    assert segment.words[0].probability == pytest.approx(math.exp(-0.5))
    assert segment.words[1].probability is None


def test_no_words_means_no_segments(engine: ElevenLabsEngine, scribe: Scribe) -> None:
    scribe.respond(json=body(space(0.0)))
    assert engine.transcribe(AUDIO, language=None, prompt=None).segments == []


@pytest.mark.parametrize(
    "content",
    [
        b"<html>gateway</html>",
        json.dumps({"language_code": "en"}).encode(),
        json.dumps({"words": [{"text": "hi", "type": "word"}]}).encode(),
    ],
    ids=["not-json", "no-words", "word-without-times"],
)
def test_malformed_response_is_an_engine_error(
    engine: ElevenLabsEngine, scribe: Scribe, content: bytes
) -> None:
    scribe.respond(200, content=content)
    with pytest.raises(EngineError, match="malformed") as info:
        engine.transcribe(AUDIO, language=None, prompt=None)
    assert not isinstance(info.value, EngineUnavailableError)


# --- error mapping -----------------------------------------------------------------------


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_overload_and_outages_are_unavailable(
    engine: ElevenLabsEngine, scribe: Scribe, status: int
) -> None:
    scribe.respond(status, json={"detail": "busy"})
    with pytest.raises(EngineUnavailableError, match=str(status)):
        engine.transcribe(AUDIO, language=None, prompt=None)


@pytest.mark.parametrize("status", [401, 403])
def test_rejected_credentials_are_unavailable_and_logged(
    engine: ElevenLabsEngine,
    scribe: Scribe,
    caplog: pytest.LogCaptureFixture,
    status: int,
) -> None:
    scribe.respond(status, json={"detail": "quota exceeded, 12 credits left"})

    with pytest.raises(EngineUnavailableError) as info:
        engine.transcribe(AUDIO, language=None, prompt=None)

    (record,) = caplog.records
    assert record.levelno == logging.ERROR
    assert "12 credits left" in record.body  # type: ignore[attr-defined]
    assert "credits" not in str(info.value), "account details must not reach clients"


@pytest.mark.parametrize("status", [400, 404, 413, 422])
def test_other_client_errors_are_plain_engine_errors(
    engine: ElevenLabsEngine,
    scribe: Scribe,
    caplog: pytest.LogCaptureFixture,
    status: int,
) -> None:
    scribe.respond(status, json={"detail": "bad audio"})

    with pytest.raises(EngineError, match=f"HTTP {status}") as info:
        engine.transcribe(AUDIO, language=None, prompt=None)

    assert not isinstance(info.value, EngineUnavailableError)
    assert "bad audio" not in str(info.value)
    assert "bad audio" in caplog.records[-1].body  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "error",
    [httpx.ConnectError("refused"), httpx.ConnectTimeout("slow"), httpx.ReadTimeout("slow")],
    ids=lambda e: type(e).__name__,
)
def test_transport_failures_are_unavailable(
    engine: ElevenLabsEngine, scribe: Scribe, error: httpx.TransportError
) -> None:
    scribe.fail(error)
    with pytest.raises(EngineUnavailableError) as info:
        engine.transcribe(AUDIO, language=None, prompt=None)
    assert info.value.__cause__ is error


def test_undecodable_body_is_an_engine_error(engine: ElevenLabsEngine, scribe: Scribe) -> None:
    error = httpx.DecodingError("incorrect header check")
    scribe.fail(error)
    with pytest.raises(EngineError, match="malformed") as info:
        engine.transcribe(AUDIO, language=None, prompt=None)
    assert not isinstance(info.value, EngineUnavailableError)
    assert info.value.__cause__ is error
