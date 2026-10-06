"""``tx-worker`` as a real process: Whisper model, audio fetched over HTTP from the S3
endpoint (the compose moto server, not an in-process mock), Prometheus endpoint, and a
graceful exit on SIGTERM."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest

from transcription.config import Settings
from transcription.domain import JobStatus, TranscriptionOptions
from transcription.services import Services, build_services

pytestmark = pytest.mark.integration

S3_ENDPOINT = os.getenv("TX_TEST_S3_ENDPOINT_URL", "http://localhost:5050")
REDIS_URL = os.getenv("TX_TEST_REDIS_URL", "redis://localhost:6379/0")
TX_WORKER = Path(sys.executable).parent / "tx-worker"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def services(
    database_url: str, redis_namespace: str, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Services]:
    """Services for the test side, built from the same TX_* environment the worker
    process inherits."""
    endpoint = urlsplit(S3_ENDPOINT)
    try:
        socket.create_connection((endpoint.hostname, endpoint.port or 80), timeout=1).close()
    except OSError as exc:
        pytest.skip(f"S3 endpoint {S3_ENDPOINT} not reachable ({exc}); run `make infra`")
    env = {
        "TX_ENV": "test",
        "TX_DATABASE_URL": database_url,
        "TX_REDIS_URL": REDIS_URL,
        "TX_QUEUE_STREAM": f"{redis_namespace}:jobs",
        "TX_DLQ_STREAM": f"{redis_namespace}:dlq",
        "TX_S3_BUCKET": "tx-worker-test",
        "TX_S3_ENDPOINT_URL": S3_ENDPOINT,
        "TX_WORKER_METRICS_PORT": str(free_port()),
        "TX_WORKER_ID": "proc-1",
        "AWS_ACCESS_KEY_ID": "test",
        "AWS_SECRET_ACCESS_KEY": "test",
        "AWS_DEFAULT_REGION": "us-east-1",
        "HF_HUB_OFFLINE": "1",  # the model is cached; never reach for the network
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    built = build_services(Settings(), db_pool_size=2)
    built.store.ensure_bucket(retention_days=1)
    built.queue.ensure_group()
    yield built
    built.close()


def test_worker_process_transcribes_audio_from_s3_and_exits_on_sigterm(
    services: Services, samples_dir: Path, tmp_path: Path
) -> None:
    key = services.repo.create_api_key("tenant")[0]
    audio_key = services.store.key_for(key.id, uuid.uuid4())
    services.store.upload_file(audio_key, samples_dir / "hello.mp3")
    job, _ = services.repo.create_job(
        api_key_id=key.id,
        audio_key=audio_key,
        audio_bytes=None,
        audio_sha256=None,
        options=TranscriptionOptions(),
        webhook_url=None,
        idempotency_key=None,
        request_fingerprint=uuid.uuid4().hex,
    )
    services.queue.enqueue(job.id)
    log_path = tmp_path / "worker.log"

    with log_path.open("w") as log_file:
        process = subprocess.Popen([TX_WORKER], stdout=log_file, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 180
        while (current := services.repo.get_job(job.id)) and not current.status.terminal:
            assert process.poll() is None, log_path.read_text()
            assert time.monotonic() < deadline, log_path.read_text()
            time.sleep(0.2)
        metrics = httpx.get(
            f"http://127.0.0.1:{services.settings.worker_metrics_port}/metrics"
        ).text

        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=30) == 0, log_path.read_text()
    finally:
        if process.poll() is None:
            process.kill()
        services.store.delete(audio_key)

    assert current is not None and current.status is JobStatus.SUCCEEDED, current
    assert current.worker_id == "proc-1"
    assert current.result is not None and "hello" in current.result.text.lower()
    assert 'tx_jobs_finished_total{error_code="",status="succeeded"} 1.0' in metrics
    log = log_path.read_text()
    assert '"msg": "asr engine loaded"' in log
    assert '"msg": "worker stopped"' in log
