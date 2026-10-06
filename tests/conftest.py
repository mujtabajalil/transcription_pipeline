"""Shared fixtures.

Unit tests need only ffmpeg. Integration tests (``-m integration``) need the compose
infra (``make infra``): each pytest session gets its own throwaway Postgres database and
Redis key namespace, so parallel runs don't collide. S3 is always moto, in-process.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from transcription.config import Settings

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"

ADMIN_DB_URL = os.getenv(
    "TX_TEST_ADMIN_DATABASE_URL", "postgresql+psycopg://tx:tx@localhost:5432/tx"
)
TEST_REDIS_URL = os.getenv("TX_TEST_REDIS_URL", "redis://localhost:6379/0")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if shutil.which("ffmpeg") is None:
        skip = pytest.mark.skip(reason="ffmpeg not installed")
        for item in items:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def samples_dir() -> Path:
    return SAMPLES


@pytest.fixture
def make_audio(tmp_path: Path) -> Callable[..., Path]:
    """Generate audio with ffmpeg's lavfi sources.

    make_audio("tone.wav", "sine=frequency=440:duration=3", "-ac", "2")
    """

    def _make(name: str, lavfi: str, *extra: str) -> Path:
        out = tmp_path / name
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", lavfi, *extra, str(out)],
            check=True,
        )
        return out

    return _make


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        env="test",
        log_json=False,
        s3_bucket="tx-test",
        s3_endpoint_url=None,
        chunk_retry_base_delay_s=0.0,
        webhook_allow_private_targets=True,
    )


# --- integration infra -------------------------------------------------------------------


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    """Fresh database with the schema from Base.metadata; dropped at session end."""
    sqlalchemy = pytest.importorskip("sqlalchemy")
    from transcription.db.models import Base

    name = f"tx_test_{uuid.uuid4().hex[:10]}"
    admin = sqlalchemy.create_engine(ADMIN_DB_URL, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            conn.exec_driver_sql(f'CREATE DATABASE "{name}"')
    except Exception as exc:  # pragma: no cover - infra missing
        pytest.skip(f"Postgres not reachable ({exc}); run `make infra`")
    url = ADMIN_DB_URL.rsplit("/", 1)[0] + f"/{name}"
    engine = sqlalchemy.create_engine(url)
    Base.metadata.create_all(engine)
    engine.dispose()
    yield url
    with admin.connect() as conn:
        conn.exec_driver_sql(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    admin.dispose()


@pytest.fixture
def redis_client() -> Iterator[object]:
    redis = pytest.importorskip("redis")
    client = redis.Redis.from_url(TEST_REDIS_URL, decode_responses=True)
    try:
        client.ping()
    except Exception as exc:  # pragma: no cover - infra missing
        pytest.skip(f"Redis not reachable ({exc}); run `make infra`")
    yield client
    client.close()


@pytest.fixture
def redis_namespace(redis_client: object) -> Iterator[str]:
    """Unique key prefix for one test; all keys under it are deleted afterwards."""
    ns = f"test:{uuid.uuid4().hex[:10]}"
    yield ns
    for key in redis_client.scan_iter(f"{ns}*"):  # type: ignore[attr-defined]
        redis_client.delete(key)  # type: ignore[attr-defined]


@pytest.fixture
def s3_client() -> Iterator[object]:
    """moto-backed boto3 S3 client (in-process, no network)."""
    moto = pytest.importorskip("moto")
    import boto3

    os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
    with moto.mock_aws():
        yield boto3.client("s3", region_name="us-east-1")
