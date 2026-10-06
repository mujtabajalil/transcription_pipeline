"""Hosted ElevenLabs Scribe speech-to-text over HTTP."""

from __future__ import annotations

import io
import logging
import math
import wave

import httpx
import numpy as np
from pydantic import BaseModel, ValidationError

from transcription.asr.base import Audio
from transcription.asr.languages import normalize_language
from transcription.domain import SAMPLE_RATE, ChunkTranscript, Segment, Word
from transcription.errors import EngineError, EngineUnavailableError

log = logging.getLogger(__name__)

_SEGMENT_GAP_S = 0.8
_SENTENCE_END = (".", "?", "!", "\u2026", "\u3002", "\uff1f", "\uff01")
"""ASCII terminators plus the ellipsis and the CJK full stop/question/exclamation marks."""


class _ScribeWord(BaseModel):
    text: str
    start: float
    end: float
    type: str
    """``word``, ``spacing`` or ``audio_event``; anything else is ignored."""
    logprob: float | None = None


class _ScribeResponse(BaseModel):
    language_code: str | None = None
    language_probability: float | None = None
    words: list[_ScribeWord]


class ElevenLabsEngine:
    """ElevenLabs Scribe behind the ``ASREngine`` seam.

    Scribe returns a flat word list; it is regrouped into sentence-like segments so
    subtitles and the quality guards see the same shape Whisper produces. Scribe reports
    no ``no_speech_prob``/``compression_ratio``, so only the confidence guards apply.
    Outages, throttling and rejected credentials raise ``EngineUnavailableError`` so a
    ``FallbackEngine`` can keep the job moving on local Whisper.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model_id: str = "scribe_v1",
        base_url: str = "https://api.elevenlabs.io",
        timeout_s: float = 60,
        client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key
        self._model_id = model_id
        self._url = f"{base_url.rstrip('/')}/v1/speech-to-text"
        self._client = client if client is not None else httpx.Client(timeout=timeout_s)

    @property
    def name(self) -> str:
        return f"elevenlabs:{self._model_id}"

    def transcribe(
        self,
        audio: Audio,
        *,
        language: str | None,
        prompt: str | None,
        word_timestamps: bool = False,
    ) -> ChunkTranscript:
        if audio.ndim != 1 or audio.dtype != np.float32:
            raise ValueError(f"expected 1-D float32 audio, got {audio.ndim}-D {audio.dtype}")
        if prompt:
            log.debug("ElevenLabs Scribe has no prompt support; prompt ignored")
        pinned = normalize_language(language)
        fields = {
            "model_id": self._model_id,
            "timestamps_granularity": "word",
            "diarize": "false",
            "tag_audio_events": "false",
        }
        if pinned:
            fields["language_code"] = pinned
        try:
            response = self._client.post(
                self._url,
                headers={"xi-api-key": self._api_key},
                data=fields,
                files={"file": ("chunk.wav", _to_wav(audio), "audio/wav")},
            )
        except httpx.TransportError as exc:
            raise EngineUnavailableError(f"ElevenLabs unreachable: {exc!r}") from exc
        except httpx.RequestError as exc:  # e.g. a body that fails its Content-Encoding
            raise EngineError(f"malformed ElevenLabs response: {exc!r}") from exc
        self._raise_for_status(response)
        try:
            body = _ScribeResponse.model_validate_json(response.content)
        except ValidationError as exc:
            raise EngineError(f"malformed ElevenLabs response: {exc}") from exc
        return ChunkTranscript(
            segments=_group_segments(body.words, keep_words=word_timestamps),
            language=normalize_language(body.language_code),
            # Same as Whisper: a pinned language was not detected, so it has no probability.
            language_probability=None if pinned else body.language_probability,
            engine=self.name,
        )

    def _raise_for_status(self, response: httpx.Response) -> None:
        status = response.status_code
        if status < 400:
            return
        # The body stays in the log: error messages reach clients (job error, webhook)
        # and Scribe's can describe our account, e.g. remaining quota on a 401.
        extra = {"status": status, "body": response.text[:500]}
        detail = f"ElevenLabs returned HTTP {status}"
        if status in (401, 403):
            # Not transient, but failing over beats failing every job until someone
            # fixes the key; the error log is what gets it fixed.
            log.error("ElevenLabs rejected the API key", extra=extra)
            raise EngineUnavailableError(detail)
        log.warning("ElevenLabs request failed", extra=extra)
        if status == 429 or status >= 500:
            raise EngineUnavailableError(detail)
        raise EngineError(detail)


def _to_wav(audio: Audio) -> bytes:
    """16 kHz mono PCM16 WAV: half the upload of float32, and exact for audio that
    ``to_float32`` scaled from int16 by 1/32768."""
    pcm = np.clip(np.round(audio * 32768), -32768, 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _group_segments(items: list[_ScribeWord], *, keep_words: bool) -> list[Segment]:
    """Split Scribe's word stream at sentence-final punctuation or pauses > 0.8 s."""
    groups: list[list[_ScribeWord]] = [[]]
    previous_end = 0.0
    for item in items:
        current = groups[-1]
        if item.type == "spacing" and current:
            current.append(item)
        elif item.type == "word":
            if current and item.start - previous_end > _SEGMENT_GAP_S:
                current = []
                groups.append(current)
            current.append(item)
            previous_end = item.end
            if item.text.rstrip().endswith(_SENTENCE_END):
                groups.append([])
    return [_to_segment(group, keep_words=keep_words) for group in groups if group]


def _to_segment(group: list[_ScribeWord], *, keep_words: bool) -> Segment:
    words = [item for item in group if item.type == "word"]
    logprobs = [w.logprob for w in words if w.logprob is not None]
    return Segment(
        start=words[0].start,
        end=words[-1].end,
        text="".join(item.text for item in group).strip(),
        avg_logprob=sum(logprobs) / len(logprobs) if logprobs else None,
        words=[_to_word(w) for w in words] if keep_words else None,
    )


def _to_word(item: _ScribeWord) -> Word:
    probability = None if item.logprob is None else math.exp(item.logprob)
    return Word(start=item.start, end=item.end, text=item.text, probability=probability)
