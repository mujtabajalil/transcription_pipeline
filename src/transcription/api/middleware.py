"""Per-request context: request id, access log, HTTP metrics, and response headers that
dependencies attach.

Pure ASGI rather than ``BaseHTTPMiddleware`` so request bodies stream through untouched
(direct uploads are read chunk by chunk) and nothing is buffered.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Mapping

from fastapi import Request
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from transcription.api.problems import REQUEST_ID_HEADER
from transcription.logging import log_context
from transcription.metrics import HTTP_LATENCY, HTTP_REQUESTS

log = logging.getLogger("transcription.api.access")

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_EXTRA_HEADERS = "response_headers"
# Requests that matched no route share one label so random paths can't explode cardinality.
_UNMATCHED_ROUTE = "<unmatched>"


def add_response_headers(request: Request, headers: Mapping[str, str]) -> None:
    """Attach headers to whatever response this request ends with.

    Unlike setting them on FastAPI's injected ``Response``, these also reach error
    responses rendered by exception handlers and ``Response`` objects that endpoints
    return themselves (e.g. RateLimit-* on a 429 or on subtitles)."""
    extra: dict[str, str] = request.scope.setdefault("state", {}).setdefault(_EXTRA_HEADERS, {})
    extra.update(headers)


class RequestContextMiddleware:
    """Assigns the request id (a well-formed incoming ``X-Request-ID`` or a uuid4),
    binds it into the log context, echoes it, and records one access-log line plus
    ``HTTP_REQUESTS``/``HTTP_LATENCY`` labelled by route template."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = Headers(scope=scope).get(REQUEST_ID_HEADER)
        request_id = incoming if incoming and _REQUEST_ID.match(incoming) else str(uuid.uuid4())
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        # Stays 500 if the app raises before starting a response: the outer error
        # handler then sends a 500.
        status = 500
        started = time.perf_counter()

        async def send_with_context(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = MutableHeaders(scope=message)
                headers.update(state.get(_EXTRA_HEADERS, {}))
                headers[REQUEST_ID_HEADER] = request_id
            await send(message)

        with log_context(request_id=request_id):
            try:
                await self.app(scope, receive, send_with_context)
            finally:
                _record(scope, status, time.perf_counter() - started)


def _record(scope: Scope, status: int, elapsed_s: float) -> None:
    route = getattr(scope.get("route"), "path", _UNMATCHED_ROUTE)
    method = scope["method"]
    HTTP_REQUESTS.labels(method=method, route=route, status=str(status)).inc()
    HTTP_LATENCY.labels(method=method, route=route).observe(elapsed_s)
    log.info(
        "request",
        extra={
            "method": method,
            "route": route,
            "status": status,
            "duration_ms": round(elapsed_s * 1000, 1),
        },
    )
