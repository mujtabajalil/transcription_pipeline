"""Signed webhook delivery with SSRF protection.

Signature (Stripe-style): header ``X-Transcription-Signature: t=<unix>,v1=<hex>`` where
``v1 = HMAC-SHA256(secret, f"{t}." + body)``. Receivers verify with ``verify()`` and
reject stale timestamps to stop replays.

SSRF: targets must be https and resolve only to globally routable addresses (no
loopback, RFC1918, link-local/metadata 169.254.169.254, CGNAT, multicast, reserved).
Checked at job creation *and* again right before each send (DNS can change), and the
send connects to the address that was checked rather than resolving the name again.
Redirects are not followed.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import logging
import socket
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

import httpx

from transcription.db.repo import JobRecord
from transcription.domain import JobStatus
from transcription.errors import BadRequestError
from transcription.metrics import WEBHOOK_DELIVERIES

log = logging.getLogger(__name__)

SIGNATURE_HEADER = "X-Transcription-Signature"
EVENT_ID_HEADER = "X-Transcription-Event-Id"
USER_AGENT = "transcription-webhooks/1"

_NAT64 = ipaddress.IPv6Network("64:ff9b::/96")
_GLOBAL_UNICAST_V6 = ipaddress.IPv6Network("2000::/3")
_RETRYABLE_STATUS = frozenset({408, 429})


def _hmac_hex(secret: str, timestamp: str, body: bytes) -> str:
    return hmac.new(secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256).hexdigest()


def sign(secret: str, body: bytes, *, timestamp: int) -> str:
    """Header value ``t=<timestamp>,v1=<hex>`` for ``body``."""
    return f"t={timestamp},v1={_hmac_hex(secret, str(timestamp), body)}"


def verify(
    secret: str, body: bytes, header: str, *, tolerance_s: int = 300, now: int | None = None
) -> bool:
    """Constant-time compare; False on malformed header or stale timestamp.

    Several ``v1`` entries are accepted (any may match) so a sender can sign with an
    old and a new secret while rotating."""
    pairs = [part.strip().split("=", 1) for part in header.split(",")]
    if any(len(pair) != 2 for pair in pairs):
        return False
    timestamps = [value for name, value in pairs if name == "t"]
    signatures = [value for name, value in pairs if name == "v1"]
    if len(timestamps) != 1 or not signatures:
        return False
    try:
        signed_at = int(timestamps[0])
    except ValueError:
        return False
    current = int(time.time()) if now is None else now
    if abs(current - signed_at) > tolerance_s:
        return False
    expected = _hmac_hex(secret, timestamps[0], body).encode()
    # Compare bytes: compare_digest rejects non-ASCII str with TypeError.
    return any(hmac.compare_digest(expected, sig.encode()) for sig in signatures)


def _is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        elif ip in _NAT64:  # NAT64 gateways translate these to the embedded IPv4 address
            ip = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        elif ip not in _GLOBAL_UNICAST_V6:
            # is_global misses legacy forms such as IPv4-compatible ::7f00:1, SIIT
            # ::ffff:0:a.b.c.d and site-local fec0::/10, so allowlist 2000::/3 instead.
            return False
    # is_global already excludes CGNAT (100.64/10); multicast counts as global.
    return ip.is_global and not ip.is_multicast


def validate_target(url: str, *, allow_private: bool) -> str | None:
    """Raise BadRequestError if ``url`` is not an acceptable webhook target, else return
    the resolved address the request must be sent to. allow_private=True (dev only) also
    permits http:// and private addresses, and skips DNS (returns None).

    Connecting to the address that was checked, instead of letting the HTTP client
    resolve the name again, is what defeats DNS rebinding (a TTL-0 record that answers
    public for the check and 169.254.169.254 for the connect) and any difference
    between how urllib and httpx encode internationalised host names."""
    try:
        parts = urlsplit(url)
    except ValueError as exc:  # e.g. "https://[127.0.0.1]/"
        raise BadRequestError("webhook_url is not a valid URL") from exc
    allowed_schemes = {"https", "http"} if allow_private else {"https"}
    if parts.scheme not in allowed_schemes:
        raise BadRequestError("webhook_url must be an https URL")
    if "@" in parts.netloc:
        raise BadRequestError("webhook_url must not contain credentials")
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError as exc:
        raise BadRequestError("webhook_url has an invalid port") from exc
    host = parts.hostname
    if not host:
        raise BadRequestError("webhook_url must include a host")
    if allow_private:
        return None
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError) as exc:
        raise BadRequestError("webhook_url host does not resolve") from exc
    addresses = [str(info[4][0]) for info in infos]
    if not addresses or not all(_is_public(address) for address in addresses):
        raise BadRequestError("webhook_url must resolve to a public internet address")
    return addresses[0]


def build_payload(job: JobRecord) -> dict[str, Any]:
    """Event envelope: {"type": "transcription.succeeded"|"transcription.failed",
    "id": <event id>, "created_at", "data": {"id", "status", "error"?,
    "duration_s"?, "language"?}}. No transcript body: receivers GET it with their API
    key, so a misconfigured URL never leaks transcript content.

    The outbox calls this on every attempt, so ``id`` and ``created_at`` derive from the
    job's terminal transition instead of the clock: a retry carries the same event id
    and receivers can drop duplicates by it."""
    if not job.status.terminal:
        raise ValueError(f"no webhook event for a {job.status} job")
    data: dict[str, Any] = {"id": str(job.id), "status": job.status.value}
    if job.status is JobStatus.FAILED:
        data["error"] = {"code": job.error_code, "message": job.error_message}
    if job.duration_s is not None:
        data["duration_s"] = job.duration_s
    if job.language is not None:
        data["language"] = job.language
    return {
        "type": f"transcription.{job.status.value}",
        "id": str(uuid.uuid5(job.id, job.status.value)),
        "created_at": (job.finished_at or datetime.now(UTC)).astimezone(UTC).isoformat(),
        "data": data,
    }


@dataclass(frozen=True)
class DeliveryResult:
    ok: bool
    status_code: int | None
    error: str | None
    retryable: bool
    """False for 4xx other than 408/429 and for targets that fail validation."""


def _result_for_status(status: int) -> DeliveryResult:
    if 200 <= status < 300:
        return DeliveryResult(ok=True, status_code=status, error=None, retryable=False)
    retryable = status in _RETRYABLE_STATUS or status >= 500
    return DeliveryResult(ok=False, status_code=status, error=f"HTTP {status}", retryable=retryable)


class WebhookSender:
    def __init__(
        self, *, timeout_s: float, allow_private: bool, client: httpx.Client | None = None
    ) -> None:
        self._allow_private = allow_private
        self._client = client or httpx.Client(timeout=timeout_s, follow_redirects=False)

    def send(self, url: str, secret: str, payload: dict[str, Any]) -> DeliveryResult:
        """One attempt (the outbox in Postgres schedules retries with backoff).

        The target is re-validated first because DNS may have changed since the job was
        created, and the request goes to the address that passed. The response body is
        never read, so a hostile receiver can't make us buffer an arbitrarily large
        reply."""
        event_id = str(payload.get("id", ""))
        try:
            address = validate_target(url, allow_private=self._allow_private)
        except BadRequestError as exc:
            result = DeliveryResult(ok=False, status_code=None, error=exc.message, retryable=False)
        else:
            result = self._post(url, address, secret, payload, event_id)
        self._record(url, event_id, result)
        return result

    def _post(
        self, url: str, address: str | None, secret: str, payload: dict[str, Any], event_id: str
    ) -> DeliveryResult:
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        headers = {
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
            SIGNATURE_HEADER: sign(secret, body, timestamp=int(time.time())),
            EVENT_ID_HEADER: event_id,
        }
        extensions: dict[str, Any] = {}
        try:
            target = httpx.URL(url)
            if address is not None:
                # Host header and TLS SNI/certificate check still use the name. The pool
                # keys connections by IP alone, so never let another name reuse one whose
                # certificate was verified for this name.
                headers["Host"] = target.netloc.decode("ascii")
                headers["Connection"] = "close"
                extensions["sni_hostname"] = target.raw_host.decode("ascii")
                target = target.copy_with(host=address)
            with self._client.stream(
                "POST",
                target,
                content=body,
                headers=headers,
                extensions=extensions,
                follow_redirects=False,
            ) as response:
                return _result_for_status(response.status_code)
        except httpx.TimeoutException:
            return DeliveryResult(ok=False, status_code=None, error="timeout", retryable=True)
        except httpx.TransportError as exc:
            error = f"{type(exc).__name__}: {exc}"
            return DeliveryResult(ok=False, status_code=None, error=error, retryable=True)
        except httpx.InvalidURL as exc:
            return DeliveryResult(ok=False, status_code=None, error=str(exc), retryable=False)

    def _record(self, url: str, event_id: str, result: DeliveryResult) -> None:
        if result.ok:
            outcome = "delivered"
        else:
            outcome = "retryable_error" if result.retryable else "permanent_error"
        WEBHOOK_DELIVERIES.labels(outcome=outcome).inc()
        # Host only: query strings of webhook URLs often carry receiver-side tokens.
        try:
            host = urlsplit(url).hostname
        except ValueError:  # the URL that failed validation may not even parse
            host = None
        extra = {
            "webhook_host": host,
            "event_id": event_id,
            "outcome": outcome,
            "status_code": result.status_code,
            "error": result.error,
        }
        if result.ok:
            log.info("webhook delivered", extra=extra)
        else:
            log.warning("webhook delivery failed", extra=extra)

    def close(self) -> None:
        self._client.close()
