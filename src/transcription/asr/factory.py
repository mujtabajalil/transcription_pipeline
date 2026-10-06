"""Builds the configured ASR engine (with optional fallback) from Settings."""

from __future__ import annotations

from transcription.asr.base import ASREngine
from transcription.asr.elevenlabs import ElevenLabsEngine
from transcription.asr.fallback import FallbackEngine
from transcription.asr.faster_whisper import FasterWhisperEngine
from transcription.config import Settings


def build_engine(settings: Settings) -> ASREngine:
    """Engine for one worker process. Loads the Whisper model if one is configured.

    Raises:
        ValueError: ``asr_engine="elevenlabs"`` without ``elevenlabs_api_key``.
    """
    if settings.asr_engine == "faster_whisper":
        # Falling back from Whisper to another copy of itself would only double the load.
        return _whisper(settings)
    primary = _elevenlabs(settings)
    if settings.asr_fallback == "faster_whisper":
        return FallbackEngine(
            primary,
            _whisper(settings),
            failure_threshold=settings.breaker_failure_threshold,
            reset_s=settings.breaker_reset_s,
        )
    return primary


def _whisper(settings: Settings) -> FasterWhisperEngine:
    return FasterWhisperEngine(
        settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        beam_size=settings.whisper_beam_size,
        cpu_threads=settings.whisper_cpu_threads,
        download_root=settings.whisper_download_root,
        no_speech_threshold=settings.no_speech_threshold,
        logprob_threshold=settings.logprob_threshold,
        compression_ratio_threshold=settings.compression_ratio_threshold,
    )


def _elevenlabs(settings: Settings) -> ElevenLabsEngine:
    key = settings.elevenlabs_api_key
    if key is None or not key.get_secret_value():
        raise ValueError("TX_ELEVENLABS_API_KEY is required when TX_ASR_ENGINE=elevenlabs")
    return ElevenLabsEngine(
        key.get_secret_value(),
        model_id=settings.elevenlabs_model,
        base_url=settings.elevenlabs_base_url,
        timeout_s=settings.elevenlabs_timeout_s,
    )
