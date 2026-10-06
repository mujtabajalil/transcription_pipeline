"""Initial schema: api_keys, jobs, job_chunks.

Hand-written to match transcription.db.models exactly (constraint names follow the
models' naming convention); tests/integration/test_migrations.py proves it.

Revision ID: 0001
Revises:
Create Date: 2026-10-05
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _ts(name: str, *, nullable: bool = True, now: bool = False) -> sa.Column[datetime]:
    return sa.Column(
        name,
        sa.DateTime(timezone=True),
        server_default=sa.func.now() if now else None,
        nullable=nullable,
    )


def upgrade() -> None:
    op.create_table(
        "api_keys",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column(
            "key_prefix", sa.String(16), nullable=False, comment="First chars, for display only"
        ),
        sa.Column("key_hash", sa.String(64), nullable=False, comment="sha256 of the raw key"),
        sa.Column(
            "webhook_secret", sa.String(128), nullable=False, comment="HMAC key for webhooks"
        ),
        sa.Column("rate_limit_per_minute", sa.Integer(), nullable=True),
        _ts("created_at", nullable=False, now=True),
        _ts("revoked_at"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_api_keys")),
        sa.UniqueConstraint("key_hash", name=op.f("uq_api_keys_key_hash")),
    )

    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("api_key_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.String(16), server_default="queued", nullable=False),
        sa.Column("idempotency_key", sa.String(255), nullable=True),
        sa.Column("request_fingerprint", sa.String(64), nullable=False),
        sa.Column("audio_key", sa.String(1024), nullable=False),
        sa.Column("audio_bytes", sa.BigInteger(), nullable=True),
        sa.Column("audio_sha256", sa.String(64), nullable=True),
        sa.Column("options", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("webhook_url", sa.Text(), nullable=True),
        sa.Column("audio_info", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("duration_s", sa.Float(), nullable=True),
        sa.Column("language", sa.String(8), nullable=True),
        sa.Column("language_probability", sa.Float(), nullable=True),
        sa.Column(
            "plan",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="ChunkPlan checkpoint",
        ),
        sa.Column("chunks_total", sa.Integer(), nullable=True),
        sa.Column("chunks_done", sa.Integer(), server_default="0", nullable=False),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("worker_id", sa.String(255), nullable=True),
        _ts("lease_expires_at"),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("error_status", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "result", postgresql.JSONB(astext_type=sa.Text()), nullable=True, comment="Transcript"
        ),
        sa.Column("webhook_status", sa.String(16), nullable=True),
        sa.Column("webhook_attempts", sa.Integer(), server_default="0", nullable=False),
        _ts("webhook_next_at"),
        sa.Column("webhook_last_error", sa.Text(), nullable=True),
        _ts("created_at", nullable=False, now=True),
        _ts("updated_at", nullable=False, now=True),
        _ts("started_at"),
        _ts("finished_at"),
        _ts("audio_deleted_at"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_jobs")),
        sa.ForeignKeyConstraint(
            ["api_key_id"],
            ["api_keys.id"],
            name=op.f("fk_jobs_api_key_id_api_keys"),
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint("api_key_id", "idempotency_key", name=op.f("uq_jobs_idempotency")),
        sa.CheckConstraint(
            "status IN ('queued', 'processing', 'succeeded', 'failed')",
            name=op.f("ck_jobs_status_valid"),
        ),
        sa.CheckConstraint(
            "webhook_status IS NULL OR webhook_status IN "
            "('pending', 'sending', 'delivered', 'failed')",
            name=op.f("ck_jobs_webhook_status_valid"),
        ),
    )
    op.create_index("ix_jobs_owner_created", "jobs", ["api_key_id", "created_at"])
    op.create_index("ix_jobs_owner_fingerprint", "jobs", ["api_key_id", "request_fingerprint"])
    op.create_index(
        "ix_jobs_active",
        "jobs",
        ["status", "updated_at"],
        postgresql_where=sa.text("status IN ('queued', 'processing')"),
    )
    op.create_index(
        "ix_jobs_webhook_due",
        "jobs",
        ["webhook_next_at"],
        postgresql_where=sa.text("webhook_status IN ('pending', 'sending')"),
    )

    op.create_table(
        "job_chunks",
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("idx", sa.Integer(), nullable=False),
        sa.Column(
            "result", postgresql.JSONB(astext_type=sa.Text()), nullable=False, comment="ChunkResult"
        ),
        _ts("created_at", nullable=False, now=True),
        sa.PrimaryKeyConstraint("job_id", "idx", name=op.f("pk_job_chunks")),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["jobs.id"],
            name=op.f("fk_job_chunks_job_id_jobs"),
            ondelete="CASCADE",
        ),
    )


def downgrade() -> None:
    # Dropping a table drops its indexes and constraints with it.
    op.drop_table("job_chunks")
    op.drop_table("jobs")
    op.drop_table("api_keys")
