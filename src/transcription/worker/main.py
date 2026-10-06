"""``tx-worker`` entry point: one process, one model, one job at a time."""

from __future__ import annotations

import logging
import os
import signal
import socket
import threading
import time
from types import FrameType

from prometheus_client import start_http_server

from transcription.asr.base import ASREngine
from transcription.asr.factory import build_engine
from transcription.config import Settings, get_settings
from transcription.logging import configure_logging
from transcription.services import build_services
from transcription.webhooks import WebhookSender
from transcription.worker.processor import Worker

log = logging.getLogger(__name__)


def main() -> None:
    """Run the worker until SIGTERM/SIGINT.

    The first signal is graceful: no new messages, the current chunk finishes and is
    checkpointed, and the job is handed back for another worker to resume. A second
    signal kills the process at once; the job is then resumed after its lease expires.
    """
    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)
    # Small pool: one job thread, the heartbeat and two background threads.
    services = build_services(settings, db_pool_size=4)
    sender = WebhookSender(
        timeout_s=settings.webhook_timeout_s,
        allow_private=settings.webhook_allow_private_targets,
    )
    stop = threading.Event()
    threads: list[threading.Thread] = []
    try:
        services.queue.ensure_group()
        engine = _load_engine(settings)
        start_http_server(settings.worker_metrics_port)
        worker_id = settings.worker_id or f"{socket.gethostname()}-{os.getpid()}"
        worker = Worker(services, engine, worker_id=worker_id, webhook_sender=sender)
        _install_signal_handlers(worker, stop)
        threads = worker.start_background(stop)
        log.info("worker started", extra={"worker_id": worker_id, "engine": engine.name})
        worker.run_forever(stop)
    finally:
        stop.set()
        for thread in threads:
            # A dispatcher may be mid-POST; don't close the services under it.
            thread.join(timeout=settings.webhook_timeout_s)
        sender.close()
        services.close()
        log.info("worker stopped")


def _load_engine(settings: Settings) -> ASREngine:
    started = time.monotonic()
    engine = build_engine(settings)
    log.info(
        "asr engine loaded",
        extra={"engine": engine.name, "load_s": round(time.monotonic() - started, 3)},
    )
    return engine


def _install_signal_handlers(worker: Worker, stop: threading.Event) -> None:
    def handle(signum: int, _frame: FrameType | None) -> None:
        log.info("shutdown requested", extra={"signal": signal.Signals(signum).name})
        worker.request_shutdown()
        stop.set()
        # The OS default for the next signal terminates immediately, even while the
        # main thread is stuck in native inference code where Python handlers can't run.
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)
