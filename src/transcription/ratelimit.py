"""Per-API-key request rate limiting in Redis (fixed window, one MULTI/EXEC per hit).

A fixed window is one key per API key: cheap, and the worst case (2x the limit across a
window boundary) is acceptable for admission control. The transaction makes the
increment and the expiry atomic, so concurrent API replicas never over-admit and a key
can never be left without a TTL (which would lock the client out forever).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from redis import Redis
from redis.exceptions import RedisError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    limit: int
    remaining: int
    reset_s: int
    """Seconds until the window resets (use for Retry-After / RateLimit-Reset)."""


class RateLimiter:
    def __init__(self, redis: Redis, *, prefix: str = "tx:rl") -> None:
        self._redis = redis
        self._prefix = prefix

    def hit(self, key: str, *, limit: int, window_s: int = 60, cost: int = 1) -> RateLimitDecision:
        """Count ``cost`` requests for ``key`` in the current window. Fails open
        (allowed=True) if Redis is unreachable — rate limiting must not take the API
        down; log it."""
        name = f"{self._prefix}:{key}"
        try:
            with self._redis.pipeline() as pipe:
                pipe.incrby(name, cost)
                # NX (Redis 7): sets the window on the first hit, and also repairs a key
                # that somehow lost its expiry, without extending a running window.
                pipe.pexpire(name, window_s * 1000, nx=True)
                pipe.pttl(name)
                count, _, pttl_ms = pipe.execute()
        except RedisError as exc:
            log.warning("rate limiter failing open", extra={"error": str(exc)})
            return RateLimitDecision(allowed=True, limit=limit, remaining=limit, reset_s=window_s)
        return RateLimitDecision(
            allowed=count <= limit,
            limit=limit,
            remaining=max(0, limit - count),
            reset_s=max(1, math.ceil(pttl_ms / 1000)),
        )
