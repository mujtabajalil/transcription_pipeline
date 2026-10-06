"""Daemon threads every worker runs next to its job loop: the webhook outbox dispatcher
and the orphan sweeper.

Both loops survive any exception (logged, retried with backoff): a broken webhook
receiver or a Redis blip must never take the transcription loop down with it.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable

from transcription.db.repo import Repository, WebhookTask
from transcription.errors import DependencyUnavailableError
from transcription.logging import log_context
from transcription.services import Services
from transcription.webhooks import WebhookSender, build_payload

log = logging.getLogger(__name__)

SWEEPER_LOCK_KEY = "tx:lock:sweeper"

_WEBHOOK_BATCH = 10
_WEBHOOK_LEASE_S = 60
_WEBHOOK_IDLE_POLL_S = 1.0
_WEBHOOK_MAX_RETRY_DELAY_S = 3600
_MAX_LOOP_BACKOFF_S = 60.0
_NEVER_S = 10**9


def backoff_s(failures: int, *, cap_s: float) -> float:
    """Delay after ``failures`` consecutive failures: 1, 2, 4, ... seconds, capped."""
    return float(min(cap_s, 2 ** (failures - 1)))


def dispatch_webhooks(repo: Repository, sender: WebhookSender, *, max_attempts: int) -> int:
    """Send one attempt for each due outbox row (at most one batch). Returns how many
    were claimed, so the caller can drain a backlog without sleeping.

    Claimed rows are leased for a minute: if this process dies mid-batch, another
    worker's dispatcher picks them up once the lease runs out.
    """
    tasks = repo.claim_webhooks(limit=_WEBHOOK_BATCH, lease_s=_WEBHOOK_LEASE_S)
    for task in tasks:
        with log_context(job_id=str(task.job.id)):
            _deliver(repo, sender, task, max_attempts=max_attempts)
    return len(tasks)


def _deliver(
    repo: Repository, sender: WebhookSender, task: WebhookTask, *, max_attempts: int
) -> None:
    job = task.job
    if job.webhook_url is None:  # the outbox only queues jobs that have one
        repo.finish_webhook(job.id, delivered=False, error="no webhook_url", retry_in_s=None)
        return
    result = sender.send(job.webhook_url, task.secret, build_payload(job))
    retry_in_s: float | None = None
    if not result.ok and result.retryable and task.attempt < max_attempts:
        retry_in_s = min(5 * 2 ** (task.attempt - 1), _WEBHOOK_MAX_RETRY_DELAY_S)
    elif not result.ok:
        log.warning("webhook abandoned", extra={"attempt": task.attempt, "error": result.error})
    repo.finish_webhook(job.id, delivered=result.ok, error=result.error, retry_in_s=retry_in_s)


def sweep(services: Services, *, lock_key: str = SWEEPER_LOCK_KEY) -> list[uuid.UUID] | None:
    """Re-enqueue jobs whose wake-up message was lost. Returns the re-enqueued ids, or
    None when another worker swept within the last ``sweep_interval_s``.

    The lock is never released, only left to expire: it doubles as a cluster-wide
    "swept recently" marker, so N workers sweep once per interval instead of N times.
    """
    settings = services.settings
    lock = services.redis.lock(lock_key, timeout=settings.sweep_interval_s)
    if not lock.acquire(blocking=False):
        return None
    lag = services.queue.lag()
    # With undelivered messages in the stream, an old queued job is most likely just
    # waiting its turn, so only expired leases count as orphans then.
    ids = services.repo.requeue_orphans(
        queued_older_than_s=settings.stale_queued_after_s if lag == 0 else _NEVER_S,
        lease_grace_s=settings.visibility_timeout_s,
    )
    for job_id in ids:
        services.queue.enqueue(job_id)
    log.info("sweep finished", extra={"requeued": len(ids), "stream_lag": lag})
    return ids


def start_dispatcher(
    repo: Repository, sender: WebhookSender, *, max_attempts: int, stop: threading.Event
) -> threading.Thread:
    def step() -> float:
        claimed = dispatch_webhooks(repo, sender, max_attempts=max_attempts)
        return 0.0 if claimed == _WEBHOOK_BATCH else _WEBHOOK_IDLE_POLL_S

    return _start("webhook-dispatcher", step, stop)


def start_sweeper(
    services: Services, stop: threading.Event, *, lock_key: str = SWEEPER_LOCK_KEY
) -> threading.Thread:
    def step() -> float:
        sweep(services, lock_key=lock_key)
        return float(services.settings.sweep_interval_s)

    return _start("sweeper", step, stop)


def _start(name: str, step: Callable[[], float], stop: threading.Event) -> threading.Thread:
    thread = threading.Thread(target=_loop, args=(name, step, stop), name=name, daemon=True)
    thread.start()
    return thread


def _loop(name: str, step: Callable[[], float], stop: threading.Event) -> None:
    """Run ``step`` until ``stop``; it returns how long to wait before the next run."""
    failures = 0
    while not stop.is_set():
        try:
            delay = step()
            failures = 0
        except Exception as exc:
            failures += 1
            delay = backoff_s(failures, cap_s=_MAX_LOOP_BACKOFF_S)
            log.warning(
                "background task failed",
                extra={"task": name, "error": str(exc), "retry_in_s": delay},
                # An outage is expected noise; anything else is a bug worth a stack.
                exc_info=not isinstance(exc, DependencyUnavailableError),
            )
        stop.wait(delay)
