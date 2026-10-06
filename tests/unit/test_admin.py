"""tx-admin wiring: argument parsing, dependency selection per command, alembic.ini
discovery. Repository / Redis / ObjectStore are replaced by in-memory stubs; the real
adapters are covered by their own integration tests."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from alembic.config import Config

from transcription import admin
from transcription.config import Settings
from transcription.db.repo import ApiKeyRecord


class StubRepository:
    instances: list[StubRepository] = []
    keys: dict[uuid.UUID, ApiKeyRecord] = {}
    ping_error: Exception | None = None

    def __init__(self, session_factory: object) -> None:
        StubRepository.instances.append(self)

    def ping(self) -> None:
        if StubRepository.ping_error:
            raise StubRepository.ping_error

    def create_api_key(
        self, name: str, *, raw_key: str | None = None, rate_limit_per_minute: int | None = None
    ) -> tuple[ApiKeyRecord, str]:
        record = ApiKeyRecord(
            id=uuid.uuid4(),
            name=name,
            key_prefix="tx_abc",
            webhook_secret="whsec",
            rate_limit_per_minute=rate_limit_per_minute,
            created_at=datetime.now(UTC),
            revoked_at=None,
        )
        StubRepository.keys[record.id] = record
        return record, "tx_abcdef"

    def revoke_api_key(self, key_id: uuid.UUID) -> bool:
        # Like the real repo: True only for the call that revokes an active key.
        record = StubRepository.keys.get(key_id)
        if record is None or record.revoked_at is not None:
            return False
        record.revoked_at = datetime.now(UTC)
        return True


class StubRedis:
    stream: dict[str, list[tuple[str, dict[str, str]]]] = {}
    closed = 0

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> StubRedis:
        assert kwargs["decode_responses"] is True
        return cls()

    def xrange(self, name: str, min: str, max: str, count: int) -> list[Any]:
        assert (min, max) == ("-", "+")
        return StubRedis.stream.get(name, [])[:count]

    def ping(self) -> bool:
        raise ConnectionError("redis down")

    def close(self) -> None:
        StubRedis.closed += 1


class StubStore:
    ensured: list[int] = []

    @classmethod
    def from_settings(cls, settings: Settings) -> StubStore:
        return cls()

    def ensure_bucket(self, *, retention_days: int) -> None:
        StubStore.ensured.append(retention_days)

    def ping(self) -> None:
        pass


@pytest.fixture(autouse=True)
def stubs(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    StubRepository.instances, StubRepository.keys, StubRepository.ping_error = [], {}, None
    StubRedis.stream, StubRedis.closed = {}, 0
    StubStore.ensured = []
    monkeypatch.setattr(admin, "Repository", StubRepository)
    monkeypatch.setattr(admin, "Redis", StubRedis)
    monkeypatch.setattr(admin, "ObjectStore", StubStore)
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers  # main() reconfigures the root logger
    root.setLevel(level)


@pytest.fixture
def run(settings: Settings) -> Any:
    def _run(*argv: str, **overrides: Any) -> int:
        return admin.main(list(argv), settings=settings.model_copy(update=overrides))

    return _run


def _stdout_json_lines(capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


# --- argparse ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["frobnicate"],
        ["create-key"],
        ["create-key", "--name", ""],
        ["create-key", "--name", "   "],
        ["create-key", "--name", "x" * 201],
        ["create-key", "--name", "x", "--rate-limit", "0"],
        ["create-key", "--name", "x", "--rate-limit", "ten"],
        ["revoke-key", "not-a-uuid"],
        ["dlq"],
        ["dlq", "list", "--count", "-1"],
    ],
)
def test_invalid_arguments_exit_2(argv: list[str], run: Any) -> None:
    with pytest.raises(SystemExit) as exc:
        run(*argv)
    assert exc.value.code == 2


def test_help_lists_every_command(capsys: pytest.CaptureFixture[str], run: Any) -> None:
    with pytest.raises(SystemExit) as exc:
        run("--help")
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for name in ("migrate", "init-storage", "create-key", "revoke-key", "dlq", "check"):
        assert name in out


# --- keys ----------------------------------------------------------------------------------


def test_create_key_prints_raw_key_once_as_json(
    capsys: pytest.CaptureFixture[str], run: Any
) -> None:
    assert run("create-key", "--name", "acme", "--rate-limit", "30") == 0

    [printed] = _stdout_json_lines(capsys)
    assert printed["name"] == "acme"
    assert printed["key"] == "tx_abcdef"
    assert printed["webhook_secret"] == "whsec"
    assert printed["rate_limit_per_minute"] == 30
    assert uuid.UUID(printed["id"]) in StubRepository.keys


def test_create_key_without_rate_limit_uses_default(
    capsys: pytest.CaptureFixture[str], run: Any
) -> None:
    assert run("create-key", "--name", "acme") == 0
    assert _stdout_json_lines(capsys)[0]["rate_limit_per_minute"] is None


def test_create_key_name_is_trimmed_and_may_use_the_full_column(
    capsys: pytest.CaptureFixture[str], run: Any
) -> None:
    assert run("create-key", "--name", "  acme  ") == 0
    assert _stdout_json_lines(capsys)[0]["name"] == "acme"
    assert run("create-key", "--name", "x" * 200) == 0


def test_revoke_key(capsys: pytest.CaptureFixture[str], run: Any) -> None:
    run("create-key", "--name", "acme")
    key_id = _stdout_json_lines(capsys)[0]["id"]

    assert run("revoke-key", key_id) == 0
    assert run("revoke-key", str(uuid.uuid4())) == 1


def test_revoking_an_already_revoked_key_fails(
    capsys: pytest.CaptureFixture[str], run: Any
) -> None:
    run("create-key", "--name", "acme")
    key_id = _stdout_json_lines(capsys)[0]["id"]
    assert run("revoke-key", key_id) == 0

    assert run("revoke-key", key_id) == 1
    assert "no active api key" in capsys.readouterr().err


def test_logs_stay_off_stdout(capsys: pytest.CaptureFixture[str], run: Any) -> None:
    run("create-key", "--name", "acme", log_json=True)
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1
    assert "api key created" in captured.err


# --- dlq / storage / check -----------------------------------------------------------------


def test_dlq_list_prints_json_lines_up_to_count(
    capsys: pytest.CaptureFixture[str], run: Any
) -> None:
    StubRedis.stream["custom:dlq"] = [
        (f"{i}-0", {"job_id": str(uuid.uuid4()), "reason": "max_attempts_exceeded"})
        for i in range(5)
    ]

    assert run("dlq", "list", "--count", "3", dlq_stream="custom:dlq") == 0

    lines = _stdout_json_lines(capsys)
    assert [line["id"] for line in lines] == ["0-0", "1-0", "2-0"]
    assert lines[0]["reason"] == "max_attempts_exceeded"
    assert StubRedis.closed == 1


def test_dlq_list_empty(capsys: pytest.CaptureFixture[str], run: Any) -> None:
    assert run("dlq", "list") == 0
    assert capsys.readouterr().out == ""


def test_init_storage_applies_retention(run: Any) -> None:
    assert run("init-storage", audio_retention_days=7) == 0
    assert StubStore.ensured == [7]


def test_init_storage_skipped_when_bucket_is_managed_elsewhere(run: Any) -> None:
    assert run("init-storage", s3_manage_bucket=False) == 0
    assert StubStore.ensured == []


def test_check_reports_every_dependency_and_fails_if_any_is_down(
    capsys: pytest.CaptureFixture[str], run: Any
) -> None:
    StubRepository.ping_error = RuntimeError("db down")

    assert run("check") == 1

    [report] = _stdout_json_lines(capsys)
    assert report == {"postgres": "error: db down", "redis": "error: redis down", "s3": "ok"}
    assert StubRedis.closed == 1


def test_check_ok(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, run: Any
) -> None:
    monkeypatch.setattr(StubRedis, "ping", lambda self: True)
    assert run("check") == 0
    assert _stdout_json_lines(capsys) == [{"postgres": "ok", "redis": "ok", "s3": "ok"}]


@pytest.fixture
def hung(monkeypatch: pytest.MonkeyPatch) -> Iterator[threading.Event]:
    """Makes probes block (like a blackholed host) until the returned event is set."""
    release = threading.Event()
    monkeypatch.setattr(admin, "_CHECK_TIMEOUT_S", 0.2)
    yield release
    release.set()  # let abandoned probe threads finish


def test_check_reports_a_hanging_dependency_as_timed_out(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    hung: threading.Event,
    run: Any,
) -> None:
    monkeypatch.setattr(StubRedis, "ping", lambda self: True)
    monkeypatch.setattr(StubStore, "ping", lambda self: hung.wait())

    assert run("check") == 1

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "postgres": "ok",
        "redis": "ok",
        "s3": "error: no answer within 0.2s",
    }
    assert "dependency check timed out" in captured.err


def test_check_probes_share_one_deadline(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    hung: threading.Event,
    run: Any,
) -> None:
    monkeypatch.setattr(StubRepository, "ping", lambda self: hung.wait())
    monkeypatch.setattr(StubRedis, "ping", lambda self: hung.wait())
    monkeypatch.setattr(StubStore, "ping", lambda self: hung.wait())

    started = time.monotonic()
    assert run("check") == 1
    elapsed = time.monotonic() - started

    # Sequential probes would take 3 x 0.2 s; concurrent ones finish at the deadline.
    assert elapsed < 0.5
    assert all(
        status.startswith("error: no answer")
        for status in json.loads(capsys.readouterr().out).values()
    )


# --- migrate / alembic.ini discovery -------------------------------------------------------


@pytest.fixture
def search_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    dirs = {name: tmp_path / name for name in ("cwd", "repo")}
    for directory in dirs.values():
        directory.mkdir()
    monkeypatch.chdir(dirs["cwd"])
    monkeypatch.setattr(admin, "_REPO_ROOT", dirs["repo"])
    return dirs


@pytest.mark.parametrize(
    ("present", "expected"),
    [
        (["cwd", "repo"], "cwd"),
        (["repo"], "repo"),
    ],
)
def test_alembic_ini_search_order(
    search_dirs: dict[str, Path], present: list[str], expected: str
) -> None:
    for name in present:
        (search_dirs[name] / "alembic.ini").write_text("[alembic]\n")
    assert admin.find_alembic_ini() == search_dirs[expected] / "alembic.ini"


def test_alembic_ini_directory_is_not_a_match(search_dirs: dict[str, Path]) -> None:
    (search_dirs["cwd"] / "alembic.ini").mkdir()
    assert admin.find_alembic_ini() is None


def test_repo_root_is_the_project_checkout() -> None:
    assert (admin._REPO_ROOT / "pyproject.toml").is_file()


def test_migrate_upgrades_to_head_with_settings_url(
    search_dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch, run: Any
) -> None:
    ini = search_dirs["repo"] / "alembic.ini"
    ini.write_text("[alembic]\nscript_location = %(here)s/migrations\n")
    calls: list[tuple[Config, str]] = []
    monkeypatch.setattr(admin.command, "upgrade", lambda cfg, rev: calls.append((cfg, rev)))
    url = "postgresql+psycopg://tx:p%40ss@db:5432/tx"  # URL-encoded '@' in the password

    assert run("migrate", database_url=url) == 0

    [(config, revision)] = calls
    assert revision == "head"
    assert config.config_file_name == str(ini)
    assert config.get_main_option("sqlalchemy.url") == url
    assert config.get_main_option("script_location") == f"{search_dirs['repo']}/migrations"


def test_migrate_without_alembic_ini_fails(
    search_dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch, run: Any
) -> None:
    monkeypatch.setattr(admin.command, "upgrade", pytest.fail)
    assert run("migrate") == 1
