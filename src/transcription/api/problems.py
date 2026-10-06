"""RFC 9457 problem details for every error the API returns.

Clients get one machine-readable shape whatever failed: our own ``TranscriptionError``
taxonomy, framework errors (unknown route, wrong method), request validation and
unexpected crashes. ``type`` is ``about:blank`` (so ``title`` is the HTTP status
phrase, as the RFC requires) and the stable machine identifier is the ``code``
extension member.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from transcription.errors import TranscriptionError

log = logging.getLogger(__name__)

PROBLEM_MEDIA_TYPE = "application/problem+json"
REQUEST_ID_HEADER = "X-Request-ID"


class FieldError(BaseModel):
    loc: list[str | int] = Field(
        description="Where the invalid value is, e.g. ['query', 'language']."
    )
    msg: str
    type: str


class Problem(BaseModel):
    """RFC 9457 problem details, served as ``application/problem+json``."""

    type: str = "about:blank"
    title: str
    status: int
    detail: str
    code: str = Field(description="Stable machine-readable error code, e.g. 'queue_full'.")
    request_id: str | None = Field(default=None, description="Echo of the X-Request-ID header.")
    errors: list[FieldError] | None = Field(
        default=None, description="Per-field details; only on validation_error."
    )


def problem_responses(*statuses: int) -> dict[int | str, dict[str, Any]]:
    """OpenAPI ``responses`` entries documenting ``statuses`` as problem+json.

    ``model`` registers the Problem schema (FastAPI files it under the route's own media
    type); the explicit content entry documents the media type actually sent."""
    content = {PROBLEM_MEDIA_TYPE: {"schema": {"$ref": "#/components/schemas/Problem"}}}
    return {
        status: {"model": Problem, "description": HTTPStatus(status).phrase, "content": content}
        for status in statuses
    }


def install_problem_handlers(app: FastAPI) -> None:
    """Render every error as problem+json. ``Exception`` covers what nothing else
    caught; Starlette runs that handler outermost, outside all middleware."""
    for exc_class in (
        TranscriptionError,
        StarletteHTTPException,
        RequestValidationError,
        Exception,
    ):
        app.add_exception_handler(exc_class, _handle)


async def _handle(request: Request, exc: Exception) -> Response:
    if isinstance(exc, TranscriptionError):
        return _transcription_error(request, exc)
    if isinstance(exc, RequestValidationError):
        errors = [
            FieldError(loc=list(error["loc"]), msg=error["msg"], type=error["type"])
            for error in exc.errors()
        ]
        return _problem(
            request, 422, "validation_error", "request validation failed", errors=errors
        )
    if isinstance(exc, StarletteHTTPException):
        code = HTTPStatus(exc.status_code).phrase.lower().replace(" ", "_").replace("-", "_")
        return _problem(request, exc.status_code, code, str(exc.detail), headers=exc.headers)
    request_id = _request_id(request)
    log.error(
        "unhandled error",
        exc_info=exc,
        extra={"request_id": request_id, "method": request.method, "path": request.url.path},
    )
    # This handler runs outside the request-id middleware, so it echoes the id itself.
    headers = {REQUEST_ID_HEADER: request_id} if request_id else None
    return _problem(request, 500, "internal_error", "internal server error", headers=headers)


def _transcription_error(request: Request, exc: TranscriptionError) -> Response:
    headers: dict[str, str] = {}
    if exc.retry_after_s is not None:
        headers["Retry-After"] = str(exc.retry_after_s)
    if exc.http_status == HTTPStatus.UNAUTHORIZED:
        headers["WWW-Authenticate"] = "Bearer"
    if exc.http_status >= HTTPStatus.INTERNAL_SERVER_ERROR:
        log.warning("request failed", extra={"code": exc.code, "status": exc.http_status})
    return _problem(request, exc.http_status, exc.code, exc.message, headers=headers)


def _problem(
    request: Request,
    status: int,
    code: str,
    detail: str,
    *,
    headers: Mapping[str, str] | None = None,
    errors: list[FieldError] | None = None,
) -> JSONResponse:
    problem = Problem(
        title=HTTPStatus(status).phrase,
        status=status,
        detail=detail,
        code=code,
        request_id=_request_id(request),
        errors=errors,
    )
    return JSONResponse(
        problem.model_dump(mode="json", exclude_none=True),
        status_code=status,
        headers=headers,
        media_type=PROBLEM_MEDIA_TYPE,
    )


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)
