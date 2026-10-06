"""``tx-admin``: operator commands for deploys and incident response.

Each command builds only the adapters it touches, so ``migrate`` works before Redis is
up and ``dlq list`` works while Postgres is down. Results go to stdout as JSON (one
object, or one per line) so they can be piped to ``jq``; logs go to stderr.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from alembic import command
from alembic.config import Config
from redis import Redis

from transcription.config import Settings, get_settings
from transcription.db.repo import Repository
from transcription.db.session import make_engine, make_session_factory
from transcription.logging import configure_logging
from transcription.storage import ObjectStore

log = logging.getLogger(__name__)

# src/transcription/admin.py → repository root (editable install, including the image).
_REPO_ROOT = Path(__file__).resolve().parents[2]
# api_keys.name is VARCHAR(200); rejecting longer names here beats a DataError traceback.
_KEY_NAME_MAX = 200
# Without a deadline an unreachable host blocks for the OS TCP timeout (minutes), and
# `check` is what operators run during exactly that kind of outage.
_CHECK_TIMEOUT_S = 10.0


def main(argv: Sequence[str] | None = None, *, settings: Settings | None = None) -> int:
    """Entry point of the ``tx-admin`` console script; returns the process exit code."""
    args = _parser().parse_args(argv)
    settings = settings or get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json, stream=sys.stderr)
    handler: Callable[[argparse.Namespace, Settings], int] = args.handler
    return handler(args, settings)


def find_alembic_ini() -> Path | None:
    """Locate ``alembic.ini``: the working directory first (an operator's explicit
    choice), then the repository root next to the installed package (``/app`` in the
    image). None if neither has one."""
    for directory in (Path.cwd(), _REPO_ROOT):
        candidate = directory / "alembic.ini"
        if candidate.is_file():
            return candidate
    return None


# --- commands --------------------------------------------------------------------------


def _migrate(args: argparse.Namespace, settings: Settings) -> int:
    ini = find_alembic_ini()
    if ini is None:
        log.error("alembic.ini not found in the working directory, the repo root or /app")
        return 1
    config = Config(str(ini))
    # ConfigParser interpolates '%', which URL-encoded passwords contain.
    config.set_main_option("sqlalchemy.url", settings.database_url.replace("%", "%%"))
    log.info("migrating database", extra={"alembic_ini": str(ini)})
    command.upgrade(config, "head")
    log.info("database at head")
    return 0


def _init_storage(args: argparse.Namespace, settings: Settings) -> int:
    if not settings.s3_manage_bucket:
        log.info("bucket is managed elsewhere (TX_S3_MANAGE_BUCKET=false); skipping")
        return 0
    store = ObjectStore.from_settings(settings)
    store.ensure_bucket(retention_days=settings.audio_retention_days)
    return 0


def _create_key(args: argparse.Namespace, settings: Settings) -> int:
    with _repository(settings) as repo:
        record, raw_key = repo.create_api_key(args.name, rate_limit_per_minute=args.rate_limit)
    log.info("api key created", extra={"key_id": str(record.id), "key_name": record.name})
    _emit(
        {
            "id": str(record.id),
            "name": record.name,
            "key": raw_key,
            "webhook_secret": record.webhook_secret,
            "rate_limit_per_minute": record.rate_limit_per_minute,
        }
    )
    return 0


def _revoke_key(args: argparse.Namespace, settings: Settings) -> int:
    with _repository(settings) as repo:
        revoked = repo.revoke_api_key(args.key_id)
    if not revoked:
        log.error("no active api key with that id", extra={"key_id": str(args.key_id)})
        return 1
    log.info("api key revoked", extra={"key_id": str(args.key_id)})
    return 0


def _dlq_list(args: argparse.Namespace, settings: Settings) -> int:
    with _redis(settings) as redis:
        # decode_responses=True: ids and fields are str (redis-py's stubs can't say so).
        entries = cast(
            list[tuple[str, dict[str, str]]],
            redis.xrange(settings.dlq_stream, "-", "+", count=args.count),
        )
    for entry_id, fields in entries:
        _emit({"id": entry_id, **fields})
    return 0


def _check(args: argparse.Namespace, settings: Settings) -> int:
    """Ping every backing service and report all of them, not just the first failure.

    Probes run concurrently in daemon threads under one deadline: a probe that hangs is
    reported as timed out and abandoned instead of keeping the process alive."""
    probes: dict[str, Callable[[Settings], None]] = {
        "postgres": _ping_postgres,
        "redis": _ping_redis,
        "s3": _ping_s3,
    }
    results: dict[str, str] = {}
    threads = [
        threading.Thread(target=_probe, args=(name, probe, settings, results), daemon=True)
        for name, probe in probes.items()
    ]
    deadline = time.monotonic() + _CHECK_TIMEOUT_S
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    report: dict[str, str] = {}
    for name in probes:
        if name not in results:
            log.error(
                "dependency check timed out",
                extra={"dependency": name, "timeout_s": _CHECK_TIMEOUT_S},
            )
        report[name] = results.get(name, f"error: no answer within {_CHECK_TIMEOUT_S:g}s")
    _emit(report)
    return 0 if all(status == "ok" for status in report.values()) else 1


def _probe(
    name: str, probe: Callable[[Settings], None], settings: Settings, results: dict[str, str]
) -> None:
    try:
        probe(settings)
    except Exception as exc:
        log.error("dependency check failed", extra={"dependency": name, "error": str(exc)})
        results[name] = f"error: {exc}"
    else:
        results[name] = "ok"


def _ping_postgres(settings: Settings) -> None:
    with _repository(settings) as repo:
        repo.ping()


def _ping_redis(settings: Settings) -> None:
    with _redis(settings) as redis:
        redis.ping()


def _ping_s3(settings: Settings) -> None:
    ObjectStore.from_settings(settings).ping()


# --- wiring ----------------------------------------------------------------------------


@contextmanager
def _repository(settings: Settings) -> Iterator[Repository]:
    engine = make_engine(settings.database_url, pool_size=1)
    try:
        yield Repository(make_session_factory(engine))
    finally:
        engine.dispose()


@contextmanager
def _redis(settings: Settings) -> Iterator[Redis]:
    redis = Redis.from_url(
        settings.redis_url, decode_responses=True, socket_timeout=10, socket_connect_timeout=5
    )
    try:
        yield redis
    finally:
        redis.close()


def _emit(payload: Mapping[str, object]) -> None:
    print(json.dumps(payload, default=str), flush=True)


def _key_name(value: str) -> str:
    name = value.strip()
    if not 1 <= len(name) <= _KEY_NAME_MAX:
        raise argparse.ArgumentTypeError(f"must be 1-{_KEY_NAME_MAX} characters")
    return name


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {number}")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tx-admin", description="Operate the transcription service."
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    commands.add_parser(
        "migrate", help="apply database migrations (alembic upgrade head)"
    ).set_defaults(handler=_migrate)
    commands.add_parser(
        "init-storage", help="create the S3 bucket with encryption and audio retention"
    ).set_defaults(handler=_init_storage)

    create = commands.add_parser("create-key", help="create an API key (printed once)")
    create.add_argument("--name", type=_key_name, required=True, help="who or what the key is for")
    create.add_argument(
        "--rate-limit",
        type=_positive_int,
        metavar="N",
        help="requests per minute (default: TX_RATE_LIMIT_PER_MINUTE)",
    )
    create.set_defaults(handler=_create_key)

    revoke = commands.add_parser("revoke-key", help="revoke an API key")
    revoke.add_argument("key_id", type=uuid.UUID, metavar="KEY_ID")
    revoke.set_defaults(handler=_revoke_key)

    dlq = commands.add_parser("dlq", help="inspect the dead-letter queue")
    dlq_commands = dlq.add_subparsers(dest="dlq_command", required=True, metavar="COMMAND")
    dlq_list = dlq_commands.add_parser("list", help="print dead-lettered messages, oldest first")
    dlq_list.add_argument("--count", type=_positive_int, default=100, metavar="N")
    dlq_list.set_defaults(handler=_dlq_list)

    commands.add_parser(
        "check", help="ping Postgres, Redis and S3; exit 1 if any is down"
    ).set_defaults(handler=_check)
    return parser
