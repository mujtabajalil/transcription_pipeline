"""Webhook signing, SSRF target validation, delivery classification and payloads."""

from __future__ import annotations

import json
import socket
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import httpx
import pytest
from prometheus_client import REGISTRY

from transcription.db.repo import JobRecord
from transcription.domain import JobStatus, TranscriptionOptions
from transcription.errors import BadRequestError
from transcription.webhooks import (
    SIGNATURE_HEADER,
    WebhookSender,
    build_payload,
    sign,
    validate_target,
    verify,
)

SECRET = "whsec_test"
BODY = b'{"id":"evt"}'
NOW = 1_800_000_000

# --- signatures --------------------------------------------------------------------------


def test_sign_verify_round_trip() -> None:
    header = sign(SECRET, BODY, timestamp=NOW)
    assert header.startswith(f"t={NOW},v1=")
    assert verify(SECRET, BODY, header, now=NOW + 10)


def test_verify_rejects_tampered_body_and_wrong_secret() -> None:
    header = sign(SECRET, BODY, timestamp=NOW)
    assert not verify(SECRET, BODY + b" ", header, now=NOW)
    assert not verify("other", BODY, header, now=NOW)


def test_verify_rejects_timestamp_swapped_onto_old_signature() -> None:
    _, sig = sign(SECRET, BODY, timestamp=NOW).split(",")
    assert not verify(SECRET, BODY, f"t={NOW + 1},{sig}", now=NOW)


@pytest.mark.parametrize("skew", [301, -301])
def test_verify_rejects_stale_or_future_timestamp(skew: int) -> None:
    header = sign(SECRET, BODY, timestamp=NOW)
    assert verify(SECRET, BODY, header, now=NOW + 300)
    assert not verify(SECRET, BODY, header, now=NOW + skew)


def test_verify_accepts_any_of_several_v1_during_rotation() -> None:
    _, sig = sign(SECRET, BODY, timestamp=NOW).split(",")
    assert verify(SECRET, BODY, f"t={NOW},v1=deadbeef,{sig}", now=NOW)


@pytest.mark.parametrize(
    "header",
    [
        "",
        "garbage",
        f"t={NOW}",
        "v1=abc",
        f"t=abc,v1={'0' * 64}",
        f"t={NOW},t={NOW},v1={'0' * 64}",
        f"t={NOW},v1=é",
        f"t={NOW};v1=abc",
    ],
)
def test_verify_rejects_malformed_header(header: str) -> None:
    assert not verify(SECRET, BODY, header, now=NOW)


# --- target validation -------------------------------------------------------------------


@pytest.fixture
def resolve_to(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Make socket.getaddrinfo return the given addresses for any host."""

    def _set(*addresses: str) -> None:
        def fake(host: str, port: int, *args: Any, **kwargs: Any) -> list[Any]:
            return [
                (
                    socket.AF_INET6 if ":" in a else socket.AF_INET,
                    socket.SOCK_STREAM,
                    6,
                    "",
                    (a, port, 0, 0) if ":" in a else (a, port),
                )
                for a in addresses
            ]

        monkeypatch.setattr(socket, "getaddrinfo", fake)

    return _set


@pytest.mark.parametrize(
    "address", ["93.184.215.14", "2606:4700:4700::1111", "::ffff:8.8.8.8", "64:ff9b::808:808"]
)
def test_public_targets_accepted(resolve_to: Callable[..., None], address: str) -> None:
    resolve_to(address)
    validate_target("https://hooks.example.com/tx?token=abc", allow_private=False)


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.1.2.3",
        "172.16.0.1",
        "172.31.255.254",
        "192.168.1.1",
        "169.254.169.254",
        "100.64.0.1",
        "100.127.255.254",
        "0.0.0.0",
        "255.255.255.255",
        "224.0.0.1",
        "239.255.255.250",
        "::1",
        "::",
        "fc00::1",
        "fd12:3456::1",
        "fe80::1",
        "fe80::1%en0",
        "ff0e::1",
        "::ffff:127.0.0.1",
        "::ffff:169.254.169.254",
        "64:ff9b::a9fe:a9fe",
        "::7f00:1",  # IPv4-compatible 127.0.0.1
        "::ffff:0:7f00:1",  # SIIT-translated 127.0.0.1
        "fec0::1",  # deprecated site-local
        "2002:7f00:1::1",  # 6to4 around 127.0.0.1
    ],
)
def test_private_and_special_targets_rejected(
    resolve_to: Callable[..., None], address: str
) -> None:
    resolve_to(address)
    with pytest.raises(BadRequestError, match="public"):
        validate_target("https://hooks.example.com/", allow_private=False)


def test_any_private_address_among_public_ones_rejects(resolve_to: Callable[..., None]) -> None:
    resolve_to("93.184.215.14", "10.0.0.5")
    with pytest.raises(BadRequestError):
        validate_target("https://hooks.example.com/", allow_private=False)


def test_unresolvable_host_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> list[Any]:
        raise socket.gaierror(socket.EAI_NONAME, "nodename nor servname provided")

    monkeypatch.setattr(socket, "getaddrinfo", fail)
    with pytest.raises(BadRequestError, match="resolve"):
        validate_target("https://nope.invalid/", allow_private=False)


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("http://hooks.example.com/", "https"),
        ("ftp://hooks.example.com/", "https"),
        ("hooks.example.com/path", "https"),
        ("https://user:pass@hooks.example.com/", "credentials"),
        ("https://user@hooks.example.com/", "credentials"),
        ("https:///path-only", "host"),
        ("https://hooks.example.com:99999/", "port"),
        ("https://hooks.example.com:abc/", "port"),
        ("https://[127.0.0.1]/", "valid URL"),
        ("https://[::1/", "valid URL"),
    ],
)
def test_malformed_urls_rejected(resolve_to: Callable[..., None], url: str, message: str) -> None:
    resolve_to("93.184.215.14")
    with pytest.raises(BadRequestError, match=message):
        validate_target(url, allow_private=False)


def test_allow_private_permits_http_and_private_without_dns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_dns(*args: Any, **kwargs: Any) -> list[Any]:
        raise AssertionError("allow_private must not resolve")

    monkeypatch.setattr(socket, "getaddrinfo", no_dns)
    validate_target("http://localhost:8080/hook", allow_private=True)
    validate_target("https://10.0.0.5/hook", allow_private=True)
    with pytest.raises(BadRequestError):
        validate_target("http://user@localhost/hook", allow_private=True)
    with pytest.raises(BadRequestError):
        validate_target("file:///etc/passwd", allow_private=True)


# --- delivery ----------------------------------------------------------------------------

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture
def requests_seen() -> list[httpx.Request]:
    return []


@pytest.fixture
def make_sender(
    requests_seen: list[httpx.Request], resolve_to: Callable[..., None]
) -> Iterator[Callable[[Handler], WebhookSender]]:
    resolve_to("93.184.215.14")
    senders: list[WebhookSender] = []

    def _make(handler: Handler) -> WebhookSender:
        def record(request: httpx.Request) -> httpx.Response:
            request.read()
            requests_seen.append(request)
            return handler(request)

        client = httpx.Client(transport=httpx.MockTransport(record))
        sender = WebhookSender(timeout_s=1, allow_private=False, client=client)
        senders.append(sender)
        return sender

    yield _make
    for sender in senders:
        sender.close()


PAYLOAD = {"type": "transcription.succeeded", "id": "evt-1", "data": {"id": "job-1"}}
URL = "https://hooks.example.com/tx"


def _deliveries(outcome: str) -> float:
    value = REGISTRY.get_sample_value("tx_webhook_deliveries_total", {"outcome": outcome})
    return value or 0.0


def test_successful_delivery_is_signed_and_verifiable(
    make_sender: Callable[[Handler], WebhookSender], requests_seen: list[httpx.Request]
) -> None:
    before = _deliveries("delivered")
    sender = make_sender(lambda r: httpx.Response(204))

    result = sender.send(URL, SECRET, PAYLOAD)

    assert result.ok
    assert result.status_code == 204
    assert result.error is None
    assert _deliveries("delivered") == before + 1
    [request] = requests_seen
    assert request.method == "POST"
    assert request.url == httpx.URL("https://93.184.215.14/tx")  # the vetted address
    assert request.headers["Host"] == "hooks.example.com"
    assert request.extensions["sni_hostname"] == "hooks.example.com"
    assert request.headers["Connection"] == "close"  # pooled by IP: no cross-name reuse
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["User-Agent"] == "transcription-webhooks/1"
    assert request.headers["X-Transcription-Event-Id"] == "evt-1"
    assert (
        request.content == b'{"data":{"id":"job-1"},"id":"evt-1","type":"transcription.succeeded"}'
    )
    assert verify(SECRET, request.content, request.headers[SIGNATURE_HEADER])
    assert json.loads(request.content) == PAYLOAD


@pytest.mark.parametrize("status", [500, 502, 503, 429, 408])
def test_server_errors_and_throttling_are_retryable(
    make_sender: Callable[[Handler], WebhookSender], status: int
) -> None:
    before = _deliveries("retryable_error")
    result = make_sender(lambda r: httpx.Response(status)).send(URL, SECRET, PAYLOAD)

    assert not result.ok
    assert result.retryable
    assert result.status_code == status
    assert result.error == f"HTTP {status}"
    assert _deliveries("retryable_error") == before + 1


@pytest.mark.parametrize("status", [400, 401, 404, 410])
def test_client_errors_are_permanent(
    make_sender: Callable[[Handler], WebhookSender], status: int
) -> None:
    before = _deliveries("permanent_error")
    result = make_sender(lambda r: httpx.Response(status)).send(URL, SECRET, PAYLOAD)

    assert not result.ok
    assert not result.retryable
    assert result.status_code == status
    assert _deliveries("permanent_error") == before + 1


def test_redirect_is_not_followed_and_is_permanent(
    make_sender: Callable[[Handler], WebhookSender], requests_seen: list[httpx.Request]
) -> None:
    sender = make_sender(
        lambda r: httpx.Response(302, headers={"Location": "http://169.254.169.254/latest"})
    )

    result = sender.send(URL, SECRET, PAYLOAD)

    assert (result.ok, result.retryable, result.status_code) == (False, False, 302)
    assert len(requests_seen) == 1


def test_injected_client_that_follows_redirects_still_does_not(
    requests_seen: list[httpx.Request], resolve_to: Callable[..., None]
) -> None:
    resolve_to("93.184.215.14")

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        return httpx.Response(307, headers={"Location": "https://internal/"})

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    sender = WebhookSender(timeout_s=1, allow_private=False, client=client)

    assert sender.send(URL, SECRET, PAYLOAD).status_code == 307
    assert len(requests_seen) == 1
    sender.close()


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ReadTimeout("read timed out"),
        httpx.ConnectTimeout("connect timed out"),
        httpx.ConnectError("connection refused"),
        httpx.RemoteProtocolError("server disconnected"),
    ],
)
def test_network_failures_are_retryable(
    make_sender: Callable[[Handler], WebhookSender], exc: httpx.TransportError
) -> None:
    def raise_(request: httpx.Request) -> httpx.Response:
        raise exc

    result = make_sender(raise_).send(URL, SECRET, PAYLOAD)

    assert (result.ok, result.retryable, result.status_code) == (False, True, None)
    assert result.error


def test_invalid_target_is_permanent_and_never_sent(
    make_sender: Callable[[Handler], WebhookSender],
    requests_seen: list[httpx.Request],
    resolve_to: Callable[..., None],
) -> None:
    sender = make_sender(lambda r: httpx.Response(200))
    resolve_to("169.254.169.254")  # DNS changed since the job was created
    before = _deliveries("permanent_error")

    result = sender.send(URL, SECRET, PAYLOAD)
    http_result = sender.send("http://hooks.example.com/", SECRET, PAYLOAD)
    unparseable_result = sender.send("https://[127.0.0.1]/", SECRET, PAYLOAD)

    assert (result.ok, result.retryable, result.status_code) == (False, False, None)
    assert result.error is not None
    assert "public" in result.error
    assert not http_result.retryable
    assert not unparseable_result.retryable
    assert requests_seen == []
    assert _deliveries("permanent_error") == before + 3


def test_send_connects_to_the_vetted_address_not_a_rebound_one(
    make_sender: Callable[[Handler], WebhookSender],
    requests_seen: list[httpx.Request],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DNS rebinding: a TTL-0 name answers public for the check, then loopback."""
    answers = iter(["93.184.215.14"])
    lookups: list[str] = []

    def rebinding(host: str, port: int, *args: Any, **kwargs: Any) -> list[Any]:
        lookups.append(host)
        address = next(answers, "127.0.0.1")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    sender = make_sender(lambda r: httpx.Response(200))
    monkeypatch.setattr(socket, "getaddrinfo", rebinding)

    assert sender.send(URL, SECRET, PAYLOAD).ok

    assert lookups == ["hooks.example.com"]
    [request] = requests_seen
    assert request.url.host == "93.184.215.14"


def test_ipv6_pinning_keeps_host_and_port(
    make_sender: Callable[[Handler], WebhookSender],
    requests_seen: list[httpx.Request],
    resolve_to: Callable[..., None],
) -> None:
    sender = make_sender(lambda r: httpx.Response(200))
    resolve_to("2606:4700:4700::1111")

    assert sender.send("https://hooks.example.com:8443/tx?token=abc", SECRET, PAYLOAD).ok

    [request] = requests_seen
    assert request.url == httpx.URL("https://[2606:4700:4700::1111]:8443/tx?token=abc")
    assert request.headers["Host"] == "hooks.example.com:8443"
    assert request.extensions["sni_hostname"] == "hooks.example.com"


def test_allow_private_sender_connects_by_name(requests_seen: list[httpx.Request]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        return httpx.Response(200)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sender = WebhookSender(timeout_s=1, allow_private=True, client=client)

    assert sender.send("http://localhost:8080/hook", SECRET, PAYLOAD).ok

    [request] = requests_seen
    assert request.url == httpx.URL("http://localhost:8080/hook")
    assert "sni_hostname" not in request.extensions
    sender.close()


# --- payloads ----------------------------------------------------------------------------


def _job(status: JobStatus, **overrides: Any) -> JobRecord:
    now = datetime.now(UTC)
    fields: dict[str, Any] = {
        "id": uuid.uuid4(),
        "api_key_id": uuid.uuid4(),
        "status": status,
        "idempotency_key": None,
        "request_fingerprint": "fp",
        "audio_key": "audio/x",
        "audio_bytes": 10,
        "audio_sha256": None,
        "options": TranscriptionOptions(),
        "webhook_url": URL,
        "audio_info": None,
        "duration_s": None,
        "language": None,
        "language_probability": None,
        "plan": None,
        "chunks_total": None,
        "chunks_done": 0,
        "attempts": 1,
        "worker_id": None,
        "lease_expires_at": None,
        "error_code": None,
        "error_status": None,
        "error_message": None,
        "result": None,
        "webhook_status": "pending",
        "webhook_attempts": 0,
        "created_at": now,
        "updated_at": now,
        "started_at": now,
        "finished_at": now,
        "audio_deleted_at": None,
    }
    return JobRecord(**(fields | overrides))


def test_payload_for_succeeded_job() -> None:
    job = _job(JobStatus.SUCCEEDED, duration_s=12.5, language="en")

    payload = build_payload(job)

    assert payload["type"] == "transcription.succeeded"
    assert datetime.fromisoformat(payload["created_at"]).utcoffset() == timedelta(0)
    assert payload["data"] == {
        "id": str(job.id),
        "status": "succeeded",
        "duration_s": 12.5,
        "language": "en",
    }
    json.dumps(payload)  # wire-serialisable as is


def test_payload_is_identical_across_outbox_retries() -> None:
    """Receivers dedupe on the event id, so a retry must not mint a new one."""
    finished = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone(timedelta(hours=2)))
    job = _job(JobStatus.SUCCEEDED, finished_at=finished)

    first, retry = build_payload(job), build_payload(job)

    assert first == retry
    assert first["created_at"] == "2026-01-02T01:04:05+00:00"
    other_job = _job(JobStatus.SUCCEEDED, finished_at=finished)
    failed = _job(JobStatus.FAILED, id=job.id, finished_at=finished)
    assert len({first["id"], build_payload(other_job)["id"], build_payload(failed)["id"]}) == 3


def test_payload_for_failed_job_carries_error_and_omits_unknowns() -> None:
    job = _job(
        JobStatus.FAILED,
        error_code="unsupported_media_type",
        error_status=415,
        error_message="unsupported or corrupt file",
    )

    payload = build_payload(job)

    assert payload["type"] == "transcription.failed"
    assert payload["data"] == {
        "id": str(job.id),
        "status": "failed",
        "error": {"code": "unsupported_media_type", "message": "unsupported or corrupt file"},
    }


@pytest.mark.parametrize("status", [JobStatus.QUEUED, JobStatus.PROCESSING])
def test_payload_refuses_non_terminal_job(status: JobStatus) -> None:
    with pytest.raises(ValueError, match="no webhook event"):
        build_payload(_job(status))
