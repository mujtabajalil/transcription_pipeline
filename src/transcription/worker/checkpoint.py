"""Postgres-backed pipeline checkpoint: what lets a redelivered job skip finished chunks."""

from __future__ import annotations

import uuid

from transcription.db.repo import Repository
from transcription.domain import ChunkPlan, ChunkResult


class DbCheckpoint:
    """``pipeline.Checkpoint`` stored on the job row (plan, language) and in
    ``job_chunks`` (one row per finished chunk).

    Writes go through the repository's lease fencing, so a worker that lost the job gets
    ``LeaseLostError`` at its next checkpoint instead of overwriting the new owner's
    progress.
    """

    def __init__(self, repo: Repository, job_id: uuid.UUID, worker_id: str) -> None:
        self._repo = repo
        self._job_id = job_id
        self._worker_id = worker_id

    def load_plan(self) -> ChunkPlan | None:
        job = self._repo.get_job(self._job_id)
        return job.plan if job else None

    def save_plan(self, plan: ChunkPlan) -> None:
        self._repo.save_plan(self._job_id, worker_id=self._worker_id, plan=plan)

    def load_results(self) -> dict[int, ChunkResult]:
        return self._repo.load_chunks(self._job_id)

    def save_result(self, result: ChunkResult) -> None:
        self._repo.save_chunk(self._job_id, worker_id=self._worker_id, result=result)

    def load_language(self) -> tuple[str, float | None] | None:
        job = self._repo.get_job(self._job_id)
        if job is None or job.language is None:
            return None
        return job.language, job.language_probability

    def save_language(self, language: str, probability: float | None) -> None:
        self._repo.save_language(
            self._job_id, worker_id=self._worker_id, language=language, probability=probability
        )
