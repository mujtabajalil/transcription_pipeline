"""RateLimiter against a real Redis: window accounting, expiry, atomicity, fail-open."""

from __future__ import annotations

import logging
import threading
import time

import pytest
from redis import Redis
from redis.backoff import NoBackoff
from redis.retry import Retry

from transcription.ratelimit import RateLimiter

pytestmark = pytest.mark.integration


@pytest.fixture
def limiter(redis_client: Redis, redis_namespace: str) -> RateLimiter:
    return RateLimiter(redis_client, prefix=redis_namespace)


def test_allows_exactly_limit_then_denies(limiter: RateLimiter) -> None:
    decisions = [limiter.hit("key-a", limit=3, window_s=60) for _ in range(5)]

    assert [d.allowed for d in decisions] == [True, True, True, False, False]
    assert [d.remaining for d in decisions] == [2, 1, 0, 0, 0]
    assert all(d.limit == 3 for d in decisions)
    assert all(55 <= d.reset_s <= 60 for d in decisions)


def test_window_expiry_resets_the_count(limiter: RateLimiter) -> None:
    assert limiter.hit("key-a", limit=1, window_s=1).allowed
    denied = limiter.hit("key-a", limit=1, window_s=1)
    assert not denied.allowed
    assert denied.reset_s == 1

    time.sleep(1.1)

    fresh = limiter.hit("key-a", limit=1, window_s=1)
    assert fresh.allowed
    assert fresh.remaining == 0


def test_keys_are_independent(limiter: RateLimiter) -> None:
    assert limiter.hit("key-a", limit=1).allowed
    assert not limiter.hit("key-a", limit=1).allowed
    assert limiter.hit("key-b", limit=1).allowed


def test_concurrent_hits_never_over_admit(limiter: RateLimiter) -> None:
    allowed: list[bool] = []
    lock = threading.Lock()

    def hit() -> None:
        decision = limiter.hit("key-a", limit=10)
        with lock:
            allowed.append(decision.allowed)

    threads = [threading.Thread(target=hit) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(allowed) == 50
    assert allowed.count(True) == 10


def test_key_without_ttl_gets_one(
    limiter: RateLimiter, redis_client: Redis, redis_namespace: str
) -> None:
    redis_client.set(f"{redis_namespace}:key-a", 100)  # stray key, no expiry

    decision = limiter.hit("key-a", limit=5, window_s=30)

    assert not decision.allowed
    assert 0 < redis_client.pttl(f"{redis_namespace}:key-a") <= 30_000
    assert decision.reset_s == 30


def test_fails_open_when_redis_is_unreachable(caplog: pytest.LogCaptureFixture) -> None:
    dead = Redis(port=1, socket_connect_timeout=0.2, retry=Retry(NoBackoff(), 0))
    limiter = RateLimiter(dead)

    with caplog.at_level(logging.WARNING, logger="transcription.ratelimit"):
        decision = limiter.hit("key-a", limit=5, window_s=60)

    assert decision.allowed
    assert decision.remaining == 5
    assert decision.reset_s == 60
    assert "failing open" in caplog.text
    dead.close()
