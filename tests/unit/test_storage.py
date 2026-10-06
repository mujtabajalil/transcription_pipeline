"""ObjectStore against moto: in-process for bucket/object semantics, and a threaded moto
server for the real HTTP paths (presigned POST, endpoint selection, connection errors)."""

from __future__ import annotations

import base64
import json
import socket
import threading
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import boto3
import httpx
import pytest
from boto3.exceptions import S3UploadFailedError
from botocore.config import Config
from botocore.exceptions import ClientError
from botocore.stub import Stubber
from moto.server import ThreadedMotoServer

from transcription.config import Settings
from transcription.errors import DependencyUnavailableError, NotFoundError
from transcription.storage import ObjectStore

BUCKET = "tx-test"
OWNER = uuid.UUID("6f1c2a9e-8a4b-4d1e-9f53-2b7c0e9d4a11")
UPLOAD = uuid.UUID("0b9d7c35-1e2f-4a6b-8c3d-5e7f9a1b2c4d")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
        return port


@pytest.fixture
def store(s3_client: Any) -> ObjectStore:
    store = ObjectStore(bucket=BUCKET, region="us-east-1", client=s3_client)
    store.ensure_bucket(retention_days=30)
    return store


@pytest.fixture(scope="module")
def moto_endpoint() -> Iterator[str]:
    """A real S3-compatible HTTP endpoint, so the default boto3 clients are exercised."""
    port = free_port()
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=port, verbose=False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("AWS_ACCESS_KEY_ID", "testing")
        mp.setenv("AWS_SECRET_ACCESS_KEY", "testing")
        server.start()
        yield f"http://127.0.0.1:{port}"
        server.stop()


def bucket_name() -> str:
    return f"tx-{uuid.uuid4().hex[:12]}"


def offline_client(**kwargs: Any) -> Any:
    """A real boto3 client for Stubber-scripted responses or unreachable endpoints."""
    return boto3.client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
        **kwargs,
    )


def decode_policy(fields: dict[str, str]) -> dict[str, Any]:
    policy: dict[str, Any] = json.loads(base64.b64decode(fields["policy"]))
    return policy


# --- bucket setup ------------------------------------------------------------------------


def test_ensure_bucket_is_idempotent_and_locks_the_bucket_down(s3_client: Any) -> None:
    store = ObjectStore(bucket=BUCKET, region="us-east-1", prefix="audio/", client=s3_client)

    store.ensure_bucket(retention_days=30)
    store.ensure_bucket(retention_days=7)  # re-applied, not duplicated

    [rule] = s3_client.get_bucket_encryption(Bucket=BUCKET)["ServerSideEncryptionConfiguration"][
        "Rules"
    ]
    assert rule["ApplyServerSideEncryptionByDefault"]["SSEAlgorithm"] == "AES256"
    block = s3_client.get_public_access_block(Bucket=BUCKET)["PublicAccessBlockConfiguration"]
    assert block == dict.fromkeys(
        ("BlockPublicAcls", "IgnorePublicAcls", "BlockPublicPolicy", "RestrictPublicBuckets"),
        True,
    )
    [lifecycle] = s3_client.get_bucket_lifecycle_configuration(Bucket=BUCKET)["Rules"]
    assert lifecycle["Status"] == "Enabled"
    assert lifecycle["Filter"] == {"Prefix": "audio/"}
    assert lifecycle["Expiration"] == {"Days": 7}
    assert lifecycle["AbortIncompleteMultipartUpload"] == {"DaysAfterInitiation": 1}


@pytest.mark.usefixtures("s3_client")
def test_ensure_bucket_outside_us_east_1_and_with_kms() -> None:
    client = boto3.client("s3", region_name="eu-west-1")
    store = ObjectStore(
        bucket="tx-eu", region="eu-west-1", sse="aws:kms", kms_key_id="alias/tx", client=client
    )

    store.ensure_bucket(retention_days=30)
    store.ensure_bucket(retention_days=30)

    assert client.get_bucket_location(Bucket="tx-eu")["LocationConstraint"] == "eu-west-1"
    [rule] = client.get_bucket_encryption(Bucket="tx-eu")["ServerSideEncryptionConfiguration"][
        "Rules"
    ]
    assert rule["ApplyServerSideEncryptionByDefault"] == {
        "SSEAlgorithm": "aws:kms",
        "KMSMasterKeyID": "alias/tx",
    }


def test_ensure_bucket_owned_by_someone_else_fails() -> None:
    stubbed = offline_client()
    store = ObjectStore(bucket=BUCKET, region="us-east-1", client=stubbed)
    with Stubber(stubbed) as stub:
        stub.add_client_error("create_bucket", "BucketAlreadyExists", http_status_code=409)
        with pytest.raises(ClientError, match="BucketAlreadyExists"):
            store.ensure_bucket(retention_days=30)


# --- keys --------------------------------------------------------------------------------


def test_key_round_trip() -> None:
    store = ObjectStore(bucket=BUCKET, region="us-east-1", client=object())
    key = store.key_for(OWNER, UPLOAD)

    assert key == f"audio/{OWNER}/{UPLOAD}"
    assert store.parse_key(key) == (OWNER, UPLOAD)


@pytest.mark.parametrize(
    "key",
    [
        f"other/{OWNER}/{UPLOAD}",
        f"audio/{OWNER}",
        f"audio/{OWNER}/{UPLOAD}/extra",
        f"audio/{OWNER}/../{UPLOAD}",
        f"audio/not-a-uuid/{UPLOAD}",
        f"audio/{str(OWNER).upper()}/{UPLOAD}",
        f"audio/{OWNER.hex}/{UPLOAD}",
        f"audio/{{{OWNER}}}/{UPLOAD}",
        f"audio/urn:uuid:{OWNER}/{UPLOAD}",
        "",
    ],
)
def test_parse_key_rejects_foreign_keys(key: str) -> None:
    store = ObjectStore(bucket=BUCKET, region="us-east-1", client=object())
    assert store.parse_key(key) is None


# --- objects -----------------------------------------------------------------------------


def test_upload_head_download_delete(store: ObjectStore, s3_client: Any, tmp_path: Path) -> None:
    src = tmp_path / "in.mp3"
    src.write_bytes(b"ID3" + bytes(range(256)) * 40)
    key = store.key_for(OWNER, UPLOAD)

    store.upload_file(key, src, content_type="audio/mpeg")

    info = store.head(key)
    assert info is not None
    assert (info.size, info.content_type) == (src.stat().st_size, "audio/mpeg")
    assert info.etag and '"' not in info.etag
    assert s3_client.head_object(Bucket=BUCKET, Key=key)["ServerSideEncryption"] == "AES256"
    dest = tmp_path / "out.mp3"
    assert store.download_file(key, dest) == src.stat().st_size
    assert dest.read_bytes() == src.read_bytes()

    store.delete(key)
    store.delete(key)  # idempotent
    assert store.head(key) is None


def test_missing_object(store: ObjectStore, tmp_path: Path) -> None:
    key = store.key_for(OWNER, uuid.uuid4())

    assert store.head(key) is None
    with pytest.raises(NotFoundError):
        store.download_file(key, tmp_path / "missing")


# --- presigned uploads -------------------------------------------------------------------


def test_presign_policy_limits_size_and_forces_sse(store: ObjectStore) -> None:
    key = store.key_for(OWNER, UPLOAD)
    before = datetime.now(UTC)

    upload = store.presign_upload(key, max_bytes=2048, expires_s=900)

    assert upload.max_bytes == 2048
    assert upload.fields["key"] == key
    assert upload.fields["x-amz-server-side-encryption"] == "AES256"
    policy = decode_policy(upload.fields)
    conditions = policy["conditions"]
    assert ["content-length-range", 1, 2048] in conditions
    assert {"x-amz-server-side-encryption": "AES256"} in conditions
    assert {"key": key} in conditions
    assert {"bucket": BUCKET} in conditions
    expiration = datetime.fromisoformat(policy["expiration"])
    assert before + timedelta(seconds=899) <= upload.expires_at <= expiration


def test_presign_with_kms_requires_the_key_id(s3_client: Any) -> None:
    store = ObjectStore(
        bucket=BUCKET, region="us-east-1", sse="aws:kms", kms_key_id="alias/tx", client=s3_client
    )

    upload = store.presign_upload(store.key_for(OWNER, UPLOAD), max_bytes=10, expires_s=60)

    assert upload.fields["x-amz-server-side-encryption"] == "aws:kms"
    assert upload.fields["x-amz-server-side-encryption-aws-kms-key-id"] == "alias/tx"
    conditions = decode_policy(upload.fields)["conditions"]
    assert {"x-amz-server-side-encryption": "aws:kms"} in conditions
    assert {"x-amz-server-side-encryption-aws-kms-key-id": "alias/tx"} in conditions


def test_presigned_post_round_trip(moto_endpoint: str, tmp_path: Path) -> None:
    store = ObjectStore(bucket=bucket_name(), region="us-east-1", endpoint_url=moto_endpoint)
    store.ensure_bucket(retention_days=1)
    key = store.key_for(OWNER, uuid.uuid4())
    upload = store.presign_upload(key, max_bytes=1024, expires_s=60)

    response = httpx.post(
        upload.url, data=upload.fields, files={"file": ("hello.mp3", b"ID3 fake audio")}
    )

    assert response.status_code == 204, response.text
    info = store.head(key)
    assert info is not None and info.size == len(b"ID3 fake audio")
    assert store.download_file(key, tmp_path / "got") == info.size


def test_presigned_url_uses_public_endpoint(moto_endpoint: str, tmp_path: Path) -> None:
    public = "http://s3.public.test:9000"
    bucket = bucket_name()
    store = ObjectStore(
        bucket=bucket, region="us-east-1", endpoint_url=moto_endpoint, public_endpoint_url=public
    )

    upload = store.presign_upload(store.key_for(OWNER, UPLOAD), max_bytes=10, expires_s=60)

    assert upload.url == f"{public}/{bucket}"  # path-style for an S3 stand-in
    # The service itself must keep talking to the internal endpoint.
    store.ensure_bucket(retention_days=1)
    src = tmp_path / "a.wav"
    src.write_bytes(b"RIFF")
    store.upload_file(store.key_for(OWNER, UPLOAD), src)
    store.ping()


def test_from_settings_wires_endpoints(
    moto_endpoint: str, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    configured = settings.model_copy(
        update={
            "s3_bucket": bucket_name(),
            "s3_endpoint_url": moto_endpoint,
            "s3_public_endpoint_url": "http://localhost:5050",
            "s3_prefix": "uploads/",
        }
    )

    store = ObjectStore.from_settings(configured)
    store.ensure_bucket(retention_days=1)

    assert store.key_for(OWNER, UPLOAD).startswith("uploads/")
    url = store.presign_upload(store.key_for(OWNER, UPLOAD), max_bytes=1, expires_s=60).url
    assert url == f"http://localhost:5050/{configured.s3_bucket}"
    store.ping()


# --- failure mapping ---------------------------------------------------------------------


def test_unreachable_endpoint_is_dependency_unavailable(tmp_path: Path) -> None:
    port = free_port()
    client = offline_client(
        endpoint_url=f"http://127.0.0.1:{port}",
        config=Config(retries={"mode": "standard", "max_attempts": 1}, connect_timeout=1),
    )
    store = ObjectStore(bucket=BUCKET, region="us-east-1", client=client)
    src = tmp_path / "a"
    src.write_bytes(b"x")

    with pytest.raises(DependencyUnavailableError) as raised:
        store.ping()
    assert str(port) not in raised.value.message  # rendered to API clients
    with pytest.raises(DependencyUnavailableError):
        store.head("audio/x")
    with pytest.raises(DependencyUnavailableError):
        store.upload_file("audio/x", src)
    with pytest.raises(DependencyUnavailableError):
        store.download_file("audio/x", tmp_path / "b")


def test_ping_missing_bucket_is_dependency_unavailable(s3_client: Any) -> None:
    store = ObjectStore(bucket="tx-never-created", region="us-east-1", client=s3_client)

    with pytest.raises(DependencyUnavailableError) as raised:
        store.ping()
    assert "tx-never-created" not in raised.value.message


def test_server_errors_are_dependency_unavailable_client_errors_are_not() -> None:
    client = offline_client()
    store = ObjectStore(bucket=BUCKET, region="us-east-1", client=client)
    with Stubber(client) as stub:
        stub.add_client_error("head_object", "SlowDown", http_status_code=503)
        stub.add_client_error("delete_object", "AccessDenied", http_status_code=403)

        with pytest.raises(DependencyUnavailableError):
            store.head("audio/x")
        with pytest.raises(ClientError, match="AccessDenied"):
            store.delete("audio/x")


def test_upload_server_error_is_dependency_unavailable(tmp_path: Path) -> None:
    """upload_file hides S3's ClientError inside S3UploadFailedError."""
    client = offline_client()
    store = ObjectStore(bucket=BUCKET, region="us-east-1", client=client)
    src = tmp_path / "a.wav"
    src.write_bytes(b"RIFF")
    with Stubber(client) as stub:
        stub.add_client_error("put_object", "SlowDown", http_status_code=503)
        stub.add_client_error("put_object", "AccessDenied", http_status_code=403)

        with pytest.raises(DependencyUnavailableError):
            store.upload_file("audio/x", src)
        with pytest.raises(S3UploadFailedError, match="AccessDenied"):
            store.upload_file("audio/x", src)


class _HangUpMidBody(BaseHTTPRequestHandler):
    """Announces a 1 KiB object, then drops the connection after 10 bytes."""

    def do_HEAD(self) -> None:
        self._headers()

    def do_GET(self) -> None:
        self._headers()
        self.wfile.write(b"x" * 10)

    def _headers(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", "1024")
        self.send_header("ETag", '"abc"')
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        pass


def test_download_broken_off_mid_stream_is_dependency_unavailable(tmp_path: Path) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _HangUpMidBody)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = offline_client(endpoint_url=f"http://127.0.0.1:{server.server_port}")
        store = ObjectStore(bucket=BUCKET, region="us-east-1", client=client)

        with pytest.raises(DependencyUnavailableError):
            store.download_file("audio/x", tmp_path / "partial")
    finally:
        server.shutdown()
        server.server_close()
    assert not (tmp_path / "partial").exists()
