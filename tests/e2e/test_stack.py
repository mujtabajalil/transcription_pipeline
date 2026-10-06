"""End to end against a running stack: API, worker with a real Whisper model, Postgres,
Redis and S3, driven only through the public HTTP API, the way a client would.

    make up
    TX_E2E_API_URL=http://localhost:8000 uv run pytest -m e2e tests/e2e

Skipped unless ``TX_E2E_API_URL`` is set; ``TX_E2E_API_KEY`` defaults to the compose
bootstrap key.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from eval.normalize import normalize

API_URL = os.getenv("TX_E2E_API_URL")
API_KEY = os.getenv("TX_E2E_API_KEY", "tx_dev_local_only_key")
POLL_TIMEOUT_S = 300
POLL_INTERVAL_S = 2

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not API_URL, reason="TX_E2E_API_URL is not set (start the stack: make up)"),
]


@pytest.fixture(scope="module")
def api() -> Iterator[httpx.Client]:
    assert API_URL is not None
    headers = {"Authorization": f"Bearer {API_KEY}"}
    with httpx.Client(base_url=API_URL, headers=headers, timeout=60) as client:
        ready = client.get("/readyz")
        assert ready.status_code == 200, f"stack is not ready: {ready.status_code} {ready.text}"
        yield client


def test_direct_upload(api: httpx.Client, samples_dir: Path) -> None:
    created = api.post(
        "/v1/transcriptions",
        content=(samples_dir / "hello.mp3").read_bytes(),
        headers={"Content-Type": "audio/mpeg"},
    )

    # 200: content dedupe handed back the job of an earlier run that never got deleted.
    assert created.status_code in (200, 202), created.text
    job = _wait_for_success(api, created.json()["id"])
    assert "quick brown fox" in normalize(job["result"]["text"])
    _assert_srt(api, job["id"], "quick brown fox")
    _assert_delete(api, job["id"])


def test_presigned_upload(api: httpx.Client, samples_dir: Path) -> None:
    audio = (samples_dir / "monologue.mp3").read_bytes()
    upload = api.post("/v1/uploads", json={"size_bytes": len(audio), "content_type": "audio/mpeg"})
    assert upload.status_code == 201, upload.text
    grant = upload.json()

    # S3 checks the policy fields before it reads the file, so the file must be the last
    # part; httpx writes ``data`` fields before ``files``.
    stored = httpx.post(
        grant["url"],
        data=grant["fields"],
        files={"file": ("monologue.mp3", audio, "audio/mpeg")},
        timeout=60,
    )
    assert stored.is_success, f"S3 rejected the upload: {stored.status_code} {stored.text}"
    created = api.post("/v1/transcriptions", json={"upload_id": grant["upload_id"]})

    assert created.status_code == 202, created.text
    job = _wait_for_success(api, created.json()["id"])
    text = normalize(job["result"]["text"])
    for phrase in ("four score and seven years ago", "all men are created equal", "civil war"):
        assert phrase in text, f"{phrase!r} missing from {text!r}"
    assert job["result"]["stats"]["chunks"] >= 2, "40 s of nonstop speech needs a forced cut"
    _assert_srt(api, job["id"], "four score")
    _assert_delete(api, job["id"])


def _wait_for_success(api: httpx.Client, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while True:
        response = api.get(f"/v1/transcriptions/{job_id}")
        assert response.status_code == 200, response.text
        job: dict[str, Any] = response.json()
        if job["status"] == "succeeded":
            return job
        if job["status"] == "failed":
            pytest.fail(f"job {job_id} failed: {job['error']}")
        if time.monotonic() > deadline:
            pytest.fail(f"job {job_id} still {job['status']} after {POLL_TIMEOUT_S} s")
        time.sleep(POLL_INTERVAL_S)


def _assert_srt(api: httpx.Client, job_id: str, phrase: str) -> None:
    srt = api.get(f"/v1/transcriptions/{job_id}/subtitles", params={"format": "srt"})
    assert srt.status_code == 200, srt.text
    assert srt.text.startswith("1\n")
    assert " --> " in srt.text
    assert phrase in normalize(srt.text)


def _assert_delete(api: httpx.Client, job_id: str) -> None:
    assert api.delete(f"/v1/transcriptions/{job_id}").status_code == 204
    assert api.get(f"/v1/transcriptions/{job_id}").status_code == 404
