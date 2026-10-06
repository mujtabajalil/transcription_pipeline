"""ASGI application factory.

    uvicorn transcription.api.app:create_app --factory

Nothing connects at import or factory time: services are built (or the injected ones
adopted) in the lifespan, so importing the app for OpenAPI generation or tests needs no
infrastructure.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import FastAPI
from pydantic import SecretStr
from starlette.concurrency import run_in_threadpool

from transcription.api.middleware import RequestContextMiddleware
from transcription.api.problems import install_problem_handlers
from transcription.api.routes import health, transcriptions, uploads
from transcription.config import Settings, get_settings
from transcription.db.repo import Repository
from transcription.errors import DependencyUnavailableError
from transcription.logging import configure_logging
from transcription.services import Services, build_services

log = logging.getLogger(__name__)

_TAGS = [
    {"name": "transcriptions", "description": "Create, poll, list and delete jobs."},
    {"name": "uploads", "description": "Presigned S3 uploads for files too large to POST."},
    {"name": "health", "description": "Liveness, readiness and Prometheus metrics."},
]


def create_app(services: Services | None = None, *, settings: Settings | None = None) -> FastAPI:
    """Build the API.

    ``services`` injected (tests, embedding) are used as-is and left open on shutdown:
    their owner closes them. Otherwise they are built from ``settings`` (default: the
    environment) at startup and closed at shutdown. Endpoints read configuration from
    ``services.settings``."""
    resolved = settings or (services.settings if services is not None else get_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(resolved.log_level, json_output=resolved.log_json)
        active = services if services is not None else build_services(resolved)
        app.state.services = active
        try:
            await _prepare(active)
            yield
        finally:
            if services is None:
                active.close()

    app = FastAPI(
        title="Transcription API",
        version=version("transcription"),
        description="Asynchronous speech-to-text: submit audio, poll the job (or receive "
        "a signed webhook), fetch the timestamped transcript or captions. Errors are RFC "
        "9457 `application/problem+json`.",
        openapi_tags=_TAGS,
        lifespan=lifespan,
    )
    app.add_middleware(RequestContextMiddleware)
    install_problem_handlers(app)
    app.include_router(transcriptions.router)
    app.include_router(uploads.router)
    app.include_router(health.router)
    return app


async def _prepare(services: Services) -> None:
    settings = services.settings
    if settings.s3_manage_bucket:
        await run_in_threadpool(
            services.store.ensure_bucket, retention_days=settings.audio_retention_days
        )
    try:
        await run_in_threadpool(services.queue.ensure_group)
    except DependencyUnavailableError:
        # Start anyway: /readyz reports Redis as down, enqueue failures are covered by
        # the sweeper, and workers recreate a missing group themselves.
        log.warning("queue group not ensured: redis unavailable")
    if settings.bootstrap_api_key is not None:
        await run_in_threadpool(_bootstrap_key, services.repo, settings.bootstrap_api_key)


def _bootstrap_key(repo: Repository, raw_key: SecretStr) -> None:
    record, _ = repo.create_api_key("bootstrap", raw_key=raw_key.get_secret_value())
    log.info(
        "bootstrap api key ready", extra={"key_id": str(record.id), "key_prefix": record.key_prefix}
    )
