"""Postgres schema. Postgres is the source of truth for job state; Redis only carries
wake-up messages, so a lost message can always be rebuilt from a row (see sweeper)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

JOB_STATUSES = ("queued", "processing", "succeeded", "failed")
WEBHOOK_STATUSES = ("pending", "sending", "delivered", "failed")


def _ts(nullable: bool = True) -> Any:
    return mapped_column(DateTime(timezone=True), nullable=nullable)


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200))
    key_prefix: Mapped[str] = mapped_column(String(16), comment="First chars, for display only")
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, comment="sha256 of the raw key")
    webhook_secret: Mapped[str] = mapped_column(String(128), comment="HMAC key for webhooks")
    rate_limit_per_minute: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    revoked_at: Mapped[datetime | None] = _ts()


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    api_key_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("api_keys.id", ondelete="RESTRICT")
    )
    status: Mapped[str] = mapped_column(String(16), default="queued", server_default="queued")

    # --- request ---
    idempotency_key: Mapped[str | None] = mapped_column(String(255))
    request_fingerprint: Mapped[str] = mapped_column(String(64))
    audio_key: Mapped[str] = mapped_column(String(1024))
    audio_bytes: Mapped[int | None] = mapped_column(BigInteger)
    audio_sha256: Mapped[str | None] = mapped_column(String(64))
    options: Mapped[dict[str, Any]] = mapped_column(JSONB)
    webhook_url: Mapped[str | None] = mapped_column(Text)

    # --- what the worker learned ---
    audio_info: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    duration_s: Mapped[float | None] = mapped_column(Float)
    language: Mapped[str | None] = mapped_column(String(8))
    language_probability: Mapped[float | None] = mapped_column(Float)
    plan: Mapped[dict[str, Any] | None] = mapped_column(JSONB, comment="ChunkPlan checkpoint")
    chunks_total: Mapped[int | None] = mapped_column(Integer)
    chunks_done: Mapped[int] = mapped_column(Integer, default=0, server_default="0")

    # --- lease / retries ---
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    worker_id: Mapped[str | None] = mapped_column(String(255))
    lease_expires_at: Mapped[datetime | None] = _ts()

    # --- outcome ---
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_status: Mapped[int | None] = mapped_column(Integer)
    error_message: Mapped[str | None] = mapped_column(Text)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB, comment="Transcript")

    # --- webhook outbox ---
    webhook_status: Mapped[str | None] = mapped_column(String(16))
    webhook_attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    webhook_next_at: Mapped[datetime | None] = _ts()
    webhook_last_error: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    started_at: Mapped[datetime | None] = _ts()
    finished_at: Mapped[datetime | None] = _ts()
    audio_deleted_at: Mapped[datetime | None] = _ts()

    __table_args__ = (
        UniqueConstraint("api_key_id", "idempotency_key", name="uq_jobs_idempotency"),
        CheckConstraint(f"status IN {JOB_STATUSES}", name="status_valid"),
        CheckConstraint(
            f"webhook_status IS NULL OR webhook_status IN {WEBHOOK_STATUSES}",
            name="webhook_status_valid",
        ),
        Index("ix_jobs_owner_created", "api_key_id", "created_at"),
        Index("ix_jobs_owner_fingerprint", "api_key_id", "request_fingerprint"),
        Index(
            "ix_jobs_active",
            "status",
            "updated_at",
            postgresql_where=text("status IN ('queued', 'processing')"),
        ),
        Index(
            "ix_jobs_webhook_due",
            "webhook_next_at",
            postgresql_where=text("webhook_status IN ('pending', 'sending')"),
        ),
    )


class JobChunk(Base):
    """One row per finished chunk: the resume checkpoint."""

    __tablename__ = "job_chunks"

    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True
    )
    idx: Mapped[int] = mapped_column(Integer, primary_key=True)
    result: Mapped[dict[str, Any]] = mapped_column(JSONB, comment="ChunkResult")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
