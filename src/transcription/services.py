"""Composition root: builds the infra adapters from Settings. Used by the API lifespan,
the worker and the admin CLI, so all three wire things identically."""

from __future__ import annotations

from dataclasses import dataclass, field

from redis import Redis
from sqlalchemy import Engine

from transcription.config import Settings, get_settings
from transcription.db.repo import Repository
from transcription.db.session import make_engine, make_session_factory
from transcription.queue import JobQueue
from transcription.ratelimit import RateLimiter
from transcription.storage import ObjectStore


@dataclass
class Services:
    settings: Settings
    engine: Engine
    repo: Repository
    redis: Redis
    queue: JobQueue
    store: ObjectStore
    rate_limiter: RateLimiter
    _closed: bool = field(default=False, repr=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.redis.close()
        self.engine.dispose()


def build_services(settings: Settings | None = None, *, db_pool_size: int = 10) -> Services:
    settings = settings or get_settings()
    engine = make_engine(settings.database_url, pool_size=db_pool_size)
    redis = Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_timeout=10,
        socket_connect_timeout=5,
        health_check_interval=30,
    )
    return Services(
        settings=settings,
        engine=engine,
        repo=Repository(make_session_factory(engine)),
        redis=redis,
        queue=JobQueue(
            redis,
            stream=settings.queue_stream,
            group=settings.queue_group,
            dlq_stream=settings.dlq_stream,
        ),
        store=ObjectStore.from_settings(settings),
        rate_limiter=RateLimiter(redis),
    )
