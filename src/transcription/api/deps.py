"""FastAPI dependencies: services, API-key auth and per-key rate limiting.

These are plain ``def`` functions on purpose: FastAPI runs them in its threadpool, so
the blocking Postgres/Redis calls never stall the event loop.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from transcription.api.middleware import add_response_headers
from transcription.db.repo import ApiKeyRecord
from transcription.errors import RateLimitedError, UnauthorizedError
from transcription.services import Services

READ_LIMIT_MULTIPLIER = 10
"""Reads (polling) are cheap and frequent; they get 10x the write budget, in their own
bucket so polling can't starve job creation."""

_bearer = HTTPBearer(auto_error=False, description="API key: `Authorization: Bearer <key>`")


def get_services(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


ServicesDep = Annotated[Services, Depends(get_services)]


def require_api_key(
    services: ServicesDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> ApiKeyRecord:
    """The caller's active API key; 401 (with ``WWW-Authenticate: Bearer``) when the
    header is missing, malformed, unknown or revoked. All cases look the same to the
    client so key probing learns nothing."""
    if credentials is None:
        raise UnauthorizedError()
    api_key = services.repo.authenticate(credentials.credentials)
    if api_key is None:
        raise UnauthorizedError()
    return api_key


def rate_limit(
    request: Request,
    services: ServicesDep,
    api_key: Annotated[ApiKeyRecord, Depends(require_api_key)],
) -> ApiKeyRecord:
    """Count this request against the key's per-minute budget and return the key.

    Writes use ``api_keys.rate_limit_per_minute`` (or the global default); GETs use
    ``READ_LIMIT_MULTIPLIER`` times that in a separate bucket. RateLimit-* headers go
    on every response, Retry-After on the 429."""
    limit = api_key.rate_limit_per_minute or services.settings.rate_limit_per_minute
    if request.method in ("GET", "HEAD"):
        bucket, limit = "read", limit * READ_LIMIT_MULTIPLIER
    else:
        bucket = "write"
    decision = services.rate_limiter.hit(f"{api_key.id}:{bucket}", limit=limit)
    add_response_headers(
        request,
        {
            "RateLimit-Limit": str(decision.limit),
            "RateLimit-Remaining": str(decision.remaining),
            "RateLimit-Reset": str(decision.reset_s),
        },
    )
    if not decision.allowed:
        raise RateLimitedError(retry_after_s=decision.reset_s)
    return api_key


AuthorizedKey = Annotated[ApiKeyRecord, Depends(rate_limit)]
"""An authenticated key whose request was admitted by the rate limiter."""
