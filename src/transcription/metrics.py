"""Prometheus metrics. Declared in one place so names/labels stay consistent.

The API serves them at /metrics; each worker serves them on TX_WORKER_METRICS_PORT.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# --- jobs ------------------------------------------------------------------------------
JOBS_CREATED = Counter("tx_jobs_created_total", "Jobs accepted by the API", ["source"])
JOBS_FINISHED = Counter(
    "tx_jobs_finished_total", "Jobs reaching a terminal state", ["status", "error_code"]
)
JOB_PROCESSING_SECONDS = Histogram(
    "tx_job_processing_seconds",
    "Wall time from claim to completion",
    buckets=(1, 5, 15, 30, 60, 120, 300, 600, 1800, 3600, 7200),
)
REALTIME_FACTOR = Histogram(
    "tx_job_realtime_factor",
    "Audio seconds per processing second (higher is faster)",
    buckets=(0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128),
)
AUDIO_SECONDS = Counter("tx_audio_seconds_total", "Decoded audio seconds")
SPEECH_RATIO = Histogram(
    "tx_speech_ratio",
    "Share of audio that VAD classified as speech",
    buckets=(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
)
JOB_REDELIVERIES = Counter(
    "tx_job_redeliveries_total", "Jobs reclaimed from a dead or stalled worker"
)
JOBS_DEAD_LETTERED = Counter("tx_jobs_dead_lettered_total", "Jobs moved to the DLQ")
JOBS_RESUMED = Counter("tx_jobs_resumed_total", "Jobs resumed from a chunk checkpoint")
JOBS_PENDING = Gauge("tx_jobs_pending", "queued+processing jobs, sampled at admission")

# --- chunks / engines ------------------------------------------------------------------
CHUNKS = Counter("tx_chunks_total", "Chunk transcription attempts", ["engine", "outcome"])
CHUNK_SECONDS = Histogram(
    "tx_chunk_transcribe_seconds",
    "Engine wall time per chunk",
    ["engine"],
    buckets=(0.25, 0.5, 1, 2, 4, 8, 16, 32, 64),
)
FORCED_CUTS = Counter("tx_forced_cuts_total", "Chunks cut inside continuous speech")
SEGMENTS_DROPPED = Counter(
    "tx_segments_dropped_total", "Segments removed by quality guards", ["reason"]
)
LOW_CONFIDENCE_SEGMENTS = Counter(
    "tx_low_confidence_segments_total", "Segments flagged low_confidence"
)
ENGINE_FALLBACKS = Counter(
    "tx_engine_fallbacks_total", "Chunks served by the fallback engine", ["primary", "fallback"]
)
BREAKER_OPEN = Gauge("tx_engine_breaker_open", "1 while an engine's circuit is open", ["engine"])

# --- API / delivery --------------------------------------------------------------------
HTTP_REQUESTS = Counter("tx_http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram("tx_http_request_seconds", "HTTP request latency", ["method", "route"])
UPLOAD_BYTES = Counter("tx_upload_bytes_total", "Bytes received via direct upload")
WEBHOOK_DELIVERIES = Counter(
    "tx_webhook_deliveries_total", "Webhook delivery outcomes", ["outcome"]
)
