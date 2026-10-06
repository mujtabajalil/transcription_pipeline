"""Presigned uploads: large files go from the client straight to S3, never through the
API."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter

from transcription.api.deps import AuthorizedKey, ServicesDep
from transcription.api.problems import problem_responses
from transcription.api.schemas import UploadRequest, UploadResponse
from transcription.errors import PayloadTooLargeError

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/uploads", tags=["uploads"])


@router.post(
    "",
    status_code=201,
    response_model=UploadResponse,
    responses=problem_responses(401, 413, 422, 429, 503),
)
def create_upload(
    body: UploadRequest, services: ServicesDep, api_key: AuthorizedKey
) -> UploadResponse:
    """Presign an S3 POST for one file of exactly the declared size or less.

    The policy caps the object at ``size_bytes`` (not at the global maximum), so a
    client cannot upload more than it declared. The object key embeds the API key id,
    which is how ``POST /v1/transcriptions`` later proves the upload is the caller's.
    Then send ``{"upload_id": ...}`` to ``POST /v1/transcriptions``."""
    settings = services.settings
    if body.size_bytes > settings.max_presigned_upload_bytes:
        raise PayloadTooLargeError(
            f"size_bytes exceeds the {settings.max_presigned_upload_bytes} byte upload limit"
        )
    upload_id = uuid.uuid4()
    presigned = services.store.presign_upload(
        services.store.key_for(api_key.id, upload_id),
        max_bytes=body.size_bytes,
        expires_s=settings.presign_ttl_s,
    )
    log.info(
        "upload presigned",
        extra={
            "upload_id": str(upload_id),
            "key_id": str(api_key.id),
            "size_bytes": body.size_bytes,
        },
    )
    return UploadResponse(upload_id=upload_id, **presigned.model_dump())
