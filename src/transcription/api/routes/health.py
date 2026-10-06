"""Liveness, readiness and Prometheus metrics. Unauthenticated and not rate limited:
they are scraped by the orchestrator and the monitoring stack, not by clients."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Literal

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.concurrency import run_in_threadpool

from transcription.api.deps import ServicesDep
from transcription.api.schemas import HealthResponse, ReadinessResponse

log = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    """Liveness: the process serves requests. Touches no dependency, so an outage of
    Postgres/Redis/S3 never gets healthy API pods restarted."""
    return HealthResponse()


@router.get(
    "/readyz",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse, "description": "A dependency is down"}},
)
async def readyz(services: ServicesDep, response: Response) -> ReadinessResponse:
    """Readiness: Postgres, Redis and S3 all answer. Checks run concurrently and all of
    them are reported; failure details go to the log, not to this unauthenticated
    endpoint."""
    probes: dict[str, Callable[[], None]] = {
        "postgres": services.repo.ping,
        "redis": services.queue.ping,
        "s3": services.store.ping,
    }
    results = await asyncio.gather(*(_check(name, ping) for name, ping in probes.items()))
    checks = dict(zip(probes, results, strict=True))
    if all(result == "ok" for result in results):
        return ReadinessResponse(status="ok", checks=checks)
    response.status_code = 503
    return ReadinessResponse(status="unavailable", checks=checks)


@router.get(
    "/metrics",
    response_class=Response,
    responses={200: {"content": {CONTENT_TYPE_LATEST: {"schema": {"type": "string"}}}}},
)
def metrics() -> Response:
    """Prometheus exposition of this process's metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


async def _check(name: str, ping: Callable[[], None]) -> Literal["ok", "error"]:
    try:
        await run_in_threadpool(ping)
    except Exception:
        # Any failure, not only DependencyUnavailableError, means "not ready": a probe
        # must report, never crash.
        log.warning("readiness check failed", extra={"check": name}, exc_info=True)
        return "error"
    return "ok"
