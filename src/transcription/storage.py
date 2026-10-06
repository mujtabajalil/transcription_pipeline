"""S3 object storage for audio.

Clients upload large files straight to S3 with a presigned POST whose policy carries a
``content-length-range`` condition, so S3 itself rejects oversize uploads and the bytes
never touch the API. Objects are written with server-side encryption and expire via a
bucket lifecycle rule (audio retention), while transcripts live on in Postgres.

Connection failures and S3 5xx responses (after botocore's own retries) surface as
``DependencyUnavailableError`` so callers treat them as transient.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import boto3
from boto3.exceptions import RetriesExceededError, S3UploadFailedError
from botocore.config import Config
from botocore.exceptions import ClientError, HTTPClientError
from botocore.exceptions import ConnectionError as BotoConnectionError
from pydantic import BaseModel

from transcription.errors import DependencyUnavailableError, NotFoundError

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client
    from mypy_boto3_s3.literals import BucketLocationConstraintType, ServerSideEncryptionType
    from mypy_boto3_s3.type_defs import ServerSideEncryptionByDefaultTypeDef

    from transcription.config import Settings

log = logging.getLogger(__name__)

_NOT_FOUND = frozenset({"404", "NoSuchKey", "NotFound"})
_SSE_FIELD = "x-amz-server-side-encryption"
_KMS_KEY_FIELD = "x-amz-server-side-encryption-aws-kms-key-id"


class PresignedUpload(BaseModel):
    url: str
    fields: dict[str, str]
    """Form fields the client must send *before* the ``file`` field (multipart POST)."""
    expires_at: datetime
    max_bytes: int


class ObjectInfo(BaseModel):
    size: int
    etag: str | None = None
    content_type: str | None = None


def _make_client(region: str, endpoint_url: str | None) -> S3Client:
    config = Config(
        signature_version="s3v4",
        retries={"mode": "adaptive", "max_attempts": 5},
        # S3 stand-ins (moto, MinIO) don't serve virtual-hosted bucket subdomains.
        s3={"addressing_style": "path" if endpoint_url else "auto"},
    )
    return boto3.client("s3", region_name=region, endpoint_url=endpoint_url, config=config)


def _error_code(exc: ClientError) -> str:
    return str(exc.response.get("Error", {}).get("Code", ""))


def _http_status(exc: ClientError) -> int:
    return int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))


def _unavailable(operation: str, exc: Exception) -> DependencyUnavailableError:
    # botocore's text names internal endpoints, buckets and keys: operators get it in
    # the log, while the raised error (rendered to API clients) stays generic.
    log.warning("S3 unavailable", extra={"operation": operation, "error": str(exc)})
    return DependencyUnavailableError(f"S3 {operation} failed")


class ObjectStore:
    def __init__(
        self,
        *,
        bucket: str,
        region: str,
        prefix: str = "audio/",
        endpoint_url: str | None = None,
        public_endpoint_url: str | None = None,
        sse: str = "AES256",
        kms_key_id: str | None = None,
        client: Any | None = None,
        presign_client: Any | None = None,
    ) -> None:
        """``client``/``presign_client`` let tests inject boto3 clients (moto). By default
        both are built from boto3's credential chain; the presign client uses
        ``public_endpoint_url`` when set so URLs work from the client's network."""
        self.bucket = bucket
        self.region = region
        self.prefix = prefix
        self.sse = cast("ServerSideEncryptionType", sse)
        self.kms_key_id = kms_key_id
        self._client: S3Client = (
            client if client is not None else _make_client(region, endpoint_url)
        )
        if presign_client is not None:
            self._presign: S3Client = presign_client
        elif public_endpoint_url is not None:
            self._presign = _make_client(region, public_endpoint_url)
        else:
            self._presign = self._client

    @classmethod
    def from_settings(cls, settings: Settings) -> ObjectStore:
        return cls(
            bucket=settings.s3_bucket,
            region=settings.s3_region,
            prefix=settings.s3_prefix,
            endpoint_url=settings.s3_endpoint_url,
            public_endpoint_url=settings.s3_public_endpoint_url,
            sse=settings.s3_sse,
            kms_key_id=settings.s3_kms_key_id,
        )

    def ensure_bucket(self, *, retention_days: int) -> None:
        """Create the bucket if missing (idempotent), enforce default SSE, block public
        access, and put a lifecycle rule expiring objects under ``prefix`` after
        ``retention_days`` (plus abort-incomplete-multipart after 1 day)."""
        with self._errors("ensure_bucket"):
            self._create_bucket()
            default_sse: ServerSideEncryptionByDefaultTypeDef = {"SSEAlgorithm": self.sse}
            if self.sse == "aws:kms" and self.kms_key_id is not None:
                default_sse["KMSMasterKeyID"] = self.kms_key_id
            self._client.put_bucket_encryption(
                Bucket=self.bucket,
                ServerSideEncryptionConfiguration={
                    "Rules": [{"ApplyServerSideEncryptionByDefault": default_sse}]
                },
            )
            self._client.put_public_access_block(
                Bucket=self.bucket,
                PublicAccessBlockConfiguration={
                    "BlockPublicAcls": True,
                    "IgnorePublicAcls": True,
                    "BlockPublicPolicy": True,
                    "RestrictPublicBuckets": True,
                },
            )
            self._client.put_bucket_lifecycle_configuration(
                Bucket=self.bucket,
                LifecycleConfiguration={
                    "Rules": [
                        {
                            "ID": "expire-audio",
                            "Filter": {"Prefix": self.prefix},
                            "Status": "Enabled",
                            "Expiration": {"Days": retention_days},
                            "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1},
                        }
                    ]
                },
            )
        log.info(
            "bucket configured",
            extra={"bucket": self.bucket, "sse": self.sse, "retention_days": retention_days},
        )

    def key_for(self, api_key_id: uuid.UUID, upload_id: uuid.UUID) -> str:
        """``{prefix}{api_key_id}/{upload_id}`` — ownership is encoded in the key, so a
        client can only reference uploads under its own API key's prefix."""
        return f"{self.prefix}{api_key_id}/{upload_id}"

    def parse_key(self, key: str) -> tuple[uuid.UUID, uuid.UUID] | None:
        """Inverse of key_for; None if the key isn't one of ours."""
        if not key.startswith(self.prefix):
            return None
        owner, _, upload = key[len(self.prefix) :].partition("/")
        try:
            ids = uuid.UUID(owner), uuid.UUID(upload)
        except ValueError:
            return None
        # uuid.UUID also accepts braces, urn: and undashed forms; only our canonical
        # spelling round-trips, so aliases of someone else's key can't slip through.
        return ids if self.key_for(*ids) == key else None

    def presign_upload(self, key: str, *, max_bytes: int, expires_s: int) -> PresignedUpload:
        """Presigned POST with conditions: exact key, content-length-range [1, max_bytes],
        and the SSE header field."""
        fields: dict[str, str] = {_SSE_FIELD: self.sse}
        conditions: list[Any] = [{_SSE_FIELD: self.sse}, ["content-length-range", 1, max_bytes]]
        if self.sse == "aws:kms" and self.kms_key_id is not None:
            fields[_KMS_KEY_FIELD] = self.kms_key_id
            conditions.append({_KMS_KEY_FIELD: self.kms_key_id})
        # Taken before signing and truncated like the policy's own expiration, so we
        # never promise more time than the policy grants.
        expires_at = (datetime.now(UTC) + timedelta(seconds=expires_s)).replace(microsecond=0)
        with self._errors("presign_upload"):
            post = self._presign.generate_presigned_post(
                Bucket=self.bucket,
                Key=key,  # boto3 adds the exact-key condition itself
                Fields=fields,
                Conditions=conditions,
                ExpiresIn=expires_s,
            )
        return PresignedUpload(
            url=post["url"], fields=post["fields"], expires_at=expires_at, max_bytes=max_bytes
        )

    def upload_file(self, key: str, path: Path, *, content_type: str | None = None) -> None:
        """Multipart-capable upload with SSE."""
        extra: dict[str, str] = {"ServerSideEncryption": self.sse}
        if self.sse == "aws:kms" and self.kms_key_id is not None:
            extra["SSEKMSKeyId"] = self.kms_key_id
        if content_type is not None:
            extra["ContentType"] = content_type
        with self._errors("upload_file"):
            self._client.upload_file(str(path), self.bucket, key, ExtraArgs=extra)

    def download_file(self, key: str, dest: Path) -> int:
        """Stream to ``dest``; returns bytes written. Raises NotFoundError if missing."""
        with self._errors("download_file"):
            try:
                self._client.download_file(self.bucket, key, str(dest))
            except ClientError as exc:
                if _error_code(exc) in _NOT_FOUND:
                    raise NotFoundError(f"audio object {key} not found") from exc
                raise
        return dest.stat().st_size

    def head(self, key: str) -> ObjectInfo | None:
        with self._errors("head"):
            try:
                meta = self._client.head_object(Bucket=self.bucket, Key=key)
            except ClientError as exc:
                if _error_code(exc) in _NOT_FOUND:
                    return None
                raise
        return ObjectInfo(
            size=meta["ContentLength"],
            etag=meta.get("ETag", "").strip('"') or None,
            content_type=meta.get("ContentType"),
        )

    def delete(self, key: str) -> None:
        """Idempotent."""
        with self._errors("delete"):
            self._client.delete_object(Bucket=self.bucket, Key=key)

    def ping(self) -> None:
        """HeadBucket; raises DependencyUnavailableError."""
        with self._errors("ping"):
            try:
                self._client.head_bucket(Bucket=self.bucket)
            except ClientError as exc:
                # A missing or forbidden bucket makes storage just as unusable as an outage.
                raise _unavailable("ping", exc) from exc

    def _create_bucket(self) -> None:
        try:
            if self.region == "us-east-1":  # the one region that rejects a LocationConstraint
                self._client.create_bucket(Bucket=self.bucket)
            else:
                self._client.create_bucket(
                    Bucket=self.bucket,
                    CreateBucketConfiguration={
                        "LocationConstraint": cast("BucketLocationConstraintType", self.region)
                    },
                )
        except ClientError as exc:
            if _error_code(exc) != "BucketAlreadyOwnedByYou":
                raise

    @contextmanager
    def _errors(self, operation: str) -> Iterator[None]:
        """Transport failures and S3 5xx (after botocore's retries) become the retryable
        DependencyUnavailableError; other 4xx stay ClientErrors for the caller."""
        try:
            yield
        except (BotoConnectionError, HTTPClientError, RetriesExceededError) as exc:
            # RetriesExceededError: s3transfer gave up on a download body that kept
            # breaking off mid-stream.
            raise _unavailable(operation, exc) from exc
        except S3UploadFailedError as exc:
            # upload_file wraps the ClientError that caused it; classify that one.
            cause = exc.__context__
            if isinstance(cause, ClientError) and _http_status(cause) >= 500:
                raise _unavailable(operation, cause) from exc
            raise
        except ClientError as exc:
            if _http_status(exc) < 500:
                raise
            raise _unavailable(operation, exc) from exc
