# Transcription service

Asynchronous speech-to-text: submit audio in any common format, get back a timestamped
transcript, plain text, or SRT/WebVTT captions. Whisper (via
[faster-whisper](https://github.com/SYSTRAN/faster-whisper)) does the recognition; this
service does everything around it: validating untrusted media, chunking long recordings
on pauses, surviving worker crashes mid-file, applying backpressure, and filtering
Whisper's known hallucinations.

```
POST /v1/transcriptions  ──▶  202 {id, status: "queued"}
GET  /v1/transcriptions/{id}  ──▶  {status, progress, result: {text, segments, stats}}
GET  /v1/transcriptions/{id}/subtitles?format=srt|vtt
```

## Highlights

- **Any input format, one code path.** Files are identified by their bytes (ffprobe),
  never by extension or Content-Type, then decoded once to 16 kHz mono PCM. WAV, MP3,
  M4A/AAC, FLAC, OGG/Opus, WebM, AIFF and video containers all work.
- **Hostile input is contained.** ffmpeg runs in a subprocess with a hard timeout and a
  demuxer allowlist, so HLS playlists and concat scripts (the usual SSRF / local-file-read
  vectors) are refused by ffmpeg itself. Duration is measured from decoded samples, not
  headers.
- **Long audio without overlap.** Silero VAD removes silence and music; speech is packed
  into ≤30 s chunks that start and end in pauses (nonstop speech is cut at the quietest
  100 ms). Chunks never overlap, so timestamps are `chunk.start + t` with nothing to
  de-duplicate at the seams.
- **Crash-safe jobs.** A Postgres lease decides who owns a job and fences every write.
  Each finished chunk is checkpointed, so a redelivered job resumes where it stopped.
  Redis Streams provides visibility-timeout redelivery and a dead-letter queue.
- **Quality guards.** Silence hallucinations ("thanks for watching"), repetition loops
  and empty segments are dropped and counted by reason. Low-confidence segments are
  flagged.
- **Engine-agnostic.** Local Whisper, hosted ElevenLabs Scribe, or Scribe with automatic
  fallback to Whisper behind a circuit breaker.
- **Production API.** API keys, per-key rate limits, global backpressure,
  `Idempotency-Key`, content dedupe, presigned S3 uploads up to 2 GiB, HMAC-signed
  webhooks with SSRF protection, RFC 9457 errors, Prometheus metrics.

## Architecture

```mermaid
flowchart LR
  C[Client] -- "POST /v1/uploads" --> API
  API -- "presigned POST" --> C
  C -- "upload ≤ 2 GiB" --> S3[(S3)]
  C -- "POST /v1/transcriptions" --> API
  API -- "row: queued" --> PG[(Postgres)]
  API -- "XADD job_id" --> R[(Redis stream)]
  R -- "XREADGROUP / XAUTOCLAIM" --> W[Worker ×N]
  W -- "download" --> S3
  W -- "lease, chunk checkpoints, result" --> PG
  W -- "signed webhook" --> H[Client endpoint]
```

| Component | Role |
|---|---|
| **API** (FastAPI) | Auth, admission control, idempotency, direct and presigned uploads, job reads, captions. Never transcribes. |
| **Worker** | One process per CPU/GPU slot, one model per process. Claims jobs, runs the pipeline, checkpoints chunks, delivers webhooks, sweeps orphaned jobs. |
| **Postgres** | Source of truth for jobs, chunk checkpoints and the webhook outbox. |
| **Redis** | Wake-up messages (Streams + consumer group) and rate-limit counters only. Anything lost is rebuilt from Postgres. |
| **S3** | Audio, SSE-encrypted, expired by a lifecycle rule after 30 days. |

The full design, including the job state machine, failure table and API contract, is
in [`docs/DESIGN.md`](docs/DESIGN.md).

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/), ffmpeg, and Docker for the full stack.

### Transcribe a file locally (no infrastructure)

The `transcribe` CLI runs the same pipeline as the worker:

```bash
make install
uv run transcribe samples/hello.mp3 --format srt
uv run transcribe samples/call_stereo.wav --split-channels --format text
```

The first run downloads the Whisper `small` model (~500 MB). Output formats are `json`
(default), `text`, `srt` and `vtt`; see `uv run transcribe --help` for language pinning,
glossary prompts and word timestamps.

### Run the full service

```bash
make up      # Postgres, Redis, S3 (moto), migrations, API on :8000, one worker
make demo    # direct upload, presigned upload, polling and SRT captions end to end
make logs
```

The compose stack bootstraps a development API key, `tx_dev_local_only_key`. Scale
workers with `WORKER_REPLICAS=4 make up`.

## Using the API

Interactive docs are served at `http://localhost:8000/docs`.

**Small files (≤ 100 MB):** send the audio as the request body; options go in the query
string.

```bash
curl -X POST "http://localhost:8000/v1/transcriptions?language=en" \
  -H "Authorization: Bearer tx_dev_local_only_key" \
  -H "Content-Type: audio/mpeg" \
  --data-binary @samples/hello.mp3
# 202 Accepted, Location: /v1/transcriptions/{id}
```

**Several files at once:** repeat the `file` part of a multipart body (≤ 100 MB in total,
≤ 20 files). Each file becomes its own job and is accepted or rejected on its own.

```bash
curl -X POST "http://localhost:8000/v1/transcriptions/batch?language=en" \
  -H "Authorization: Bearer $KEY" \
  -F file=@samples/hello.mp3 -F file=@samples/monologue.mp3
# 207 Multi-Status, {"data": [{filename, status, job, error}, ...]} in upload order
```

**Large files (≤ 2 GiB):** request a presigned upload, POST the file straight to S3, then
create the job from the upload.

```bash
curl -X POST http://localhost:8000/v1/uploads \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"size_bytes": 160827, "content_type": "audio/mpeg"}'
# → {upload_id, url, fields, expires_at, max_bytes}; upload with the fields, then:

curl -X POST http://localhost:8000/v1/transcriptions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"upload_id": "...", "prompt": "Kubernetes, PostgreSQL", "webhook_url": "https://example.com/hooks"}'
```

**Poll and fetch results:**

```bash
curl -H "Authorization: Bearer $KEY" http://localhost:8000/v1/transcriptions/{id}
curl -H "Authorization: Bearer $KEY" "http://localhost:8000/v1/transcriptions/{id}/subtitles?format=vtt"
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/uploads` | Presigned S3 POST; S3 enforces the size limit |
| `POST /v1/transcriptions` | Create a job from a raw body or an `upload_id` (202) |
| `POST /v1/transcriptions/batch` | One job per file of a multipart body (207, per-file results) |
| `GET /v1/transcriptions/{id}` | Status, progress, error and transcript |
| `GET /v1/transcriptions` | List your jobs, newest first, cursor-paginated |
| `GET /v1/transcriptions/{id}/subtitles` | SRT or WebVTT captions |
| `DELETE /v1/transcriptions/{id}` | Delete the job, transcript and audio now |
| `GET /healthz`, `/readyz`, `/metrics` | Liveness, dependency readiness, Prometheus |

**Job options:** `language` (ISO 639 code, otherwise detected once and pinned), `prompt`
(a glossary for consistent spelling across chunks), `split_channels` (one speaker per
channel, as in call-centre recordings), `word_timestamps` and `webhook_url`.

**Behaviour worth knowing:**

- **Retries.** Sending the same `Idempotency-Key` returns the original job with a 200.
  Reusing a key for a different request is a 409.
- **Duplicates.** Resubmitting the same audio with the same options returns the existing
  job (`X-Deduplicated: true`).
- **Limits.** Rate limiting is a 429 with `Retry-After`. A full queue is a 429
  `queue_full`. Oversize files are a 413, unsupported or corrupt media a 415, and files
  with no audio stream a 422.
- **Webhooks.** Webhooks are signed `X-Transcription-Signature: t=<unix>,v1=<hmac>`
  (Stripe-style) and carry no transcript text; receivers fetch the result with their
  key.

## Configuration

Every setting is an environment variable with the `TX_` prefix.
[`.env.example`](.env.example) lists them all with their defaults for a production
deployment. [`.env.docker`](.env.docker) holds the local compose values. AWS credentials
come from boto3's standard chain (environment variables, profile or IAM role); they are
never read from settings.

Commonly changed:

| Variable | Default | |
|---|---|---|
| `TX_ASR_ENGINE` | `faster_whisper` | or `elevenlabs` |
| `TX_ASR_FALLBACK` | `none` | `faster_whisper` to fall back while the hosted engine is down |
| `TX_WHISPER_MODEL` | `small` | any faster-whisper model name |
| `TX_WHISPER_DEVICE` / `TX_WHISPER_COMPUTE_TYPE` | `auto` / `int8` | use `cuda` / `float16` on GPU |
| `TX_MAX_AUDIO_SECONDS` | `14400` | 4 hours |
| `TX_MAX_PENDING_JOBS` | `200` | global backpressure threshold |
| `TX_RATE_LIMIT_PER_MINUTE` | `60` | per API key; can be overridden per key |
| `TX_AUDIO_RETENTION_DAYS` | `30` | S3 lifecycle expiry |

## Operations

`tx-admin` handles deploy-time and incident tasks (`uv run tx-admin` on the host). It prints
JSON to stdout.

```bash
tx-admin migrate                     # alembic upgrade head
tx-admin init-storage                # bucket + encryption + lifecycle + public-access block
tx-admin create-key --name acme      # prints the raw key and webhook secret once
tx-admin revoke-key <key-id>
tx-admin dlq list --count 20         # inspect dead-lettered jobs
tx-admin check                       # ping Postgres, Redis and S3
```

Workers serve Prometheus metrics on `:9100` and the API on `/metrics`. Useful series
include `tx_job_realtime_factor`, `tx_chunks_total{outcome}`,
`tx_segments_dropped_total{reason}`, `tx_engine_fallbacks_total` and
`tx_jobs_dead_lettered_total`. Logs are structured JSON with `request_id` and `job_id`
attached.

Workers shut down gracefully on SIGTERM: the current chunk finishes and is checkpointed,
then the job is handed back so another worker resumes it.

## Development

```bash
make install           # uv sync
make test              # unit tests (ffmpeg only)
make infra             # Postgres, Redis and S3 in Docker
make test-integration  # API, worker, repository, queue, migrations against real services
make e2e               # full stack with a real Whisper model
make lint typecheck    # ruff + mypy --strict
make help              # all targets
```

The test suite has about 690 unit tests and 145 integration tests. Unit tests use a
deterministic fake engine, moto for S3 and respx for HTTP. Integration tests cover the
concurrency-sensitive paths: concurrent claims have exactly one winner, sweepers and
webhook dispatchers never pick the same rows, and the rate limiter never over-admits.
CI runs lint, typecheck, unit and integration jobs on every push to `main` and every pull request.

## Evaluation

[`eval/`](eval/README.md) benchmarks the audio front-end on a reproducible 14-clip set
(clean, noisy, telephony, nonstop speech, long silences, music, no speech). It compares
this pipeline against naive fixed 30 s windows and a denoising front-end on WER, seam
errors, hallucinated words and speed:

```bash
make eval   # writes eval/results/<timestamp>.md and .json
```

## Project layout

```
src/transcription/
  audio/        probe (ffprobe + allowlist), decode, Silero VAD, chunk planning
  asr/          engine protocol, faster-whisper, ElevenLabs, fallback + breaker, quality guards
  pipeline.py   probe → decode → VAD → chunk → transcribe → checkpoint → assemble
  api/          FastAPI app, auth and rate limiting, problem+json, routes
  worker/       job loop, Postgres checkpoint, webhook dispatcher, orphan sweeper
  db/           SQLAlchemy models and repository (leases, fencing, outbox)
  queue.py      Redis Streams queue with redelivery and DLQ
  storage.py    S3: presigned uploads, SSE, lifecycle
  webhooks.py   HMAC signing and SSRF-safe delivery
  cli.py        `transcribe`    admin.py  `tx-admin`
migrations/     Alembic
eval/           WER benchmark of the audio front-end
samples/        test audio, including hostile files (playlists, concat scripts, non-audio)
docs/DESIGN.md  design decisions and trade-offs
```

## What would change at scale

- GPU workers with batched inference across a job's chunks (they are independent),
  for roughly 5–10× throughput while keeping per-chunk checkpoints.
- SQS in place of Redis Streams on AWS; the queue interface already matches its
  semantics.
- S3 event notifications instead of the client's second call after uploading.
- Per-tenant queues or weighted fair scheduling.
- Speaker diarization for mono recordings with several speakers.
