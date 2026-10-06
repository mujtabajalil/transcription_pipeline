"""Runtime configuration. Every knob is an env var with the ``TX_`` prefix.

AWS credentials are deliberately *not* settings: boto3 resolves them through its normal
chain (env vars, profile, instance/task role), which is what production uses.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from transcription.domain import PipelineConfig

MiB = 1024 * 1024


class Settings(PipelineConfig, BaseSettings):
    """Audio-pipeline and quality-guard knobs (``TX_CHUNK_MAX_S``, ``TX_VAD_THRESHOLD``,
    ...) are inherited from ``PipelineConfig``, so their defaults live in one place."""

    model_config = SettingsConfigDict(
        env_prefix="TX_", env_file=".env", extra="ignore", frozen=False
    )

    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = True

    # --- state -----------------------------------------------------------------------
    database_url: str = "postgresql+psycopg://tx:tx@localhost:5432/tx"
    redis_url: str = "redis://localhost:6379/0"

    # --- object storage (S3) ---------------------------------------------------------
    s3_bucket: str = "transcription-audio"
    s3_region: str = "us-east-1"
    s3_endpoint_url: str | None = None
    """Override only for S3-compatible stand-ins (moto in compose). None = AWS."""
    s3_public_endpoint_url: str | None = None
    """Endpoint baked into presigned URLs handed to clients, if it differs from the one
    the service itself uses (e.g. ``http://s3:5000`` inside compose vs
    ``http://localhost:5050`` on the host)."""
    s3_prefix: str = "audio/"
    s3_sse: Literal["AES256", "aws:kms"] = "AES256"
    s3_kms_key_id: str | None = None
    s3_manage_bucket: bool = True
    """Create the bucket and apply encryption + lifecycle config at startup. Turn off
    when the bucket is owned by infra-as-code."""
    presign_ttl_s: int = 900
    audio_retention_days: int = 30

    # --- admission limits ------------------------------------------------------------
    max_direct_upload_bytes: int = 100 * MiB
    """Raw-body uploads through the API. Bigger files must use a presigned upload."""
    max_presigned_upload_bytes: int = 2048 * MiB
    max_pending_jobs: int = 200
    """Global backpressure: queued+processing jobs above this → 429 queue_full."""
    rate_limit_per_minute: int = 60
    """Default per-API-key request budget; api_keys.rate_limit_per_minute overrides."""
    dedupe_by_content: bool = True
    """Same key + same audio bytes + same options → return the existing job."""

    # --- queue / worker --------------------------------------------------------------
    queue_stream: str = "tx:jobs"
    queue_group: str = "workers"
    dlq_stream: str = "tx:jobs:dlq"
    visibility_timeout_s: int = 120
    """A job whose worker stops heart-beating for this long is redelivered."""
    heartbeat_interval_s: int = 20
    max_job_attempts: int = 3
    sweep_interval_s: int = 60
    """How often a worker re-enqueues jobs orphaned between DB commit and XADD."""
    stale_queued_after_s: int = 300
    worker_metrics_port: int = 9100
    worker_id: str | None = None
    """Defaults to hostname-pid."""

    # --- ASR engines -----------------------------------------------------------------
    asr_engine: Literal["faster_whisper", "elevenlabs"] = "faster_whisper"
    asr_fallback: Literal["faster_whisper", "none"] = "none"
    whisper_model: str = "small"
    whisper_device: str = "auto"
    whisper_compute_type: str = "int8"
    whisper_beam_size: int = 5
    whisper_cpu_threads: int = 0
    whisper_download_root: str | None = None
    elevenlabs_api_key: SecretStr | None = None
    elevenlabs_model: str = "scribe_v1"
    elevenlabs_base_url: str = "https://api.elevenlabs.io"
    elevenlabs_timeout_s: float = 60
    breaker_failure_threshold: int = 3
    breaker_reset_s: float = 60

    # --- webhooks --------------------------------------------------------------------
    webhook_timeout_s: float = 10
    webhook_max_attempts: int = 5
    webhook_allow_private_targets: bool = False
    """Dev only: allow http:// and private/loopback webhook targets."""

    # --- bootstrap -------------------------------------------------------------------
    bootstrap_api_key: SecretStr | None = Field(default=None)
    """Dev convenience: ensure this raw key exists at API startup."""

    def pipeline_config(self) -> PipelineConfig:
        return PipelineConfig(**self.model_dump(include=set(PipelineConfig.model_fields)))


@lru_cache
def get_settings() -> Settings:
    return Settings()
