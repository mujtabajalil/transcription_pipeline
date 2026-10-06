"""The hand-written migration must build exactly the schema the models declare: no
autogenerate diff, and an identical catalog (constraint names, check constraints,
partial index predicates, defaults, comments) to Base.metadata.create_all."""

from __future__ import annotations

import io
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import Connection, create_engine, inspect, text
from sqlalchemy.exc import OperationalError

from tests.conftest import ADMIN_DB_URL, ROOT
from transcription.config import get_settings
from transcription.db.models import Base

pytestmark = pytest.mark.integration


@pytest.fixture
def fresh_db_url() -> Iterator[str]:
    name = f"tx_migrate_{uuid.uuid4().hex[:10]}"
    admin = create_engine(ADMIN_DB_URL, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            conn.exec_driver_sql(f'CREATE DATABASE "{name}"')
    except OperationalError as exc:  # pragma: no cover - infra missing
        pytest.skip(f"Postgres not reachable ({exc}); run `make infra`")
    yield ADMIN_DB_URL.rsplit("/", 1)[0] + f"/{name}"
    with admin.connect() as conn:
        conn.exec_driver_sql(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    admin.dispose()


def alembic_config(url: str | None, ini: Path = ROOT / "alembic.ini") -> Config:
    config = Config(str(ini))
    if url is not None:
        config.set_main_option("sqlalchemy.url", url)
    return config


def catalog(conn: Connection) -> dict[str, set[tuple[object, ...]]]:
    """Everything about the public schema that compare_metadata doesn't look at."""
    queries = {
        "columns": """
            SELECT table_name, ordinal_position, column_name, data_type,
                   character_maximum_length, is_nullable, column_default,
                   col_description(format('%I', table_name)::regclass, ordinal_position)
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name <> 'alembic_version'""",
        "constraints": """
            SELECT conrelid::regclass::text, conname, pg_get_constraintdef(oid)
            FROM pg_constraint
            WHERE connamespace = 'public'::regnamespace
              AND conrelid::regclass::text <> 'alembic_version'""",
        "indexes": """
            SELECT tablename, indexname, indexdef FROM pg_indexes
            WHERE schemaname = 'public' AND tablename <> 'alembic_version'""",
    }
    return {name: {tuple(row) for row in conn.execute(text(sql))} for name, sql in queries.items()}


def user_tables(url: str) -> set[str]:
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            return set(inspect(conn).get_table_names()) - {"alembic_version"}
    finally:
        engine.dispose()


def test_upgrade_matches_models_and_downgrade_removes_everything(
    fresh_db_url: str, database_url: str
) -> None:
    config = alembic_config(fresh_db_url)

    command.upgrade(config, "head")

    migrated, from_models = create_engine(fresh_db_url), create_engine(database_url)
    try:
        with migrated.connect() as conn:
            assert compare_metadata(MigrationContext.configure(conn), Base.metadata) == []
            migrated_catalog = catalog(conn)
        with from_models.connect() as conn:
            assert migrated_catalog == catalog(conn)
    finally:
        migrated.dispose()
        from_models.dispose()

    command.downgrade(config, "base")

    assert user_tables(fresh_db_url) == set()


def test_upgrade_from_image_layout_and_other_cwd(
    fresh_db_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Docker image ships alembic.ini + migrations/ in /app; the admin CLI may run
    from anywhere, so nothing may depend on the working directory."""
    app = tmp_path / "app"
    app.mkdir()
    shutil.copy(ROOT / "alembic.ini", app / "alembic.ini")
    shutil.copytree(
        ROOT / "migrations", app / "migrations", ignore=shutil.ignore_patterns("__pycache__")
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    command.upgrade(alembic_config(fresh_db_url, app / "alembic.ini"), "head")

    assert user_tables(fresh_db_url) == {"api_keys", "jobs", "job_chunks"}


def test_url_falls_back_to_settings(fresh_db_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TX_DATABASE_URL", fresh_db_url)
    get_settings.cache_clear()
    try:
        command.upgrade(alembic_config(None), "head")
    finally:
        get_settings.cache_clear()

    assert user_tables(fresh_db_url) == {"api_keys", "jobs", "job_chunks"}


def test_offline_mode_renders_sql() -> None:
    out = io.StringIO()
    config = Config(str(ROOT / "alembic.ini"), output_buffer=out)
    config.set_main_option("sqlalchemy.url", "postgresql+psycopg://u:p@nowhere/db")

    command.upgrade(config, "head", sql=True)

    sql = out.getvalue()
    assert "CREATE TABLE jobs" in sql
    assert "CONSTRAINT uq_jobs_idempotency UNIQUE (api_key_id, idempotency_key)" in sql
    assert "WHERE status IN ('queued', 'processing')" in sql
