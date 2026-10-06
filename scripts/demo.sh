#!/usr/bin/env bash
# Walk the public API end to end against a running stack (`make up`):
#   1. health check
#   2. direct upload of a short file → poll → transcript
#   3. presigned upload of a longer file straight to S3 → job → poll → SRT captions
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"
TX_API_KEY="${TX_API_KEY:-tx_dev_local_only_key}"
POLL_TIMEOUT_S="${POLL_TIMEOUT_S:-600}"
cd "$(dirname "$0")/.."

# curl that prints the problem+json body on HTTP errors instead of just the status.
api() {
  curl -sS --fail-with-body -H "Authorization: Bearer ${TX_API_KEY}" "$@"
}

# json_get KEY [KEY...] < json  → the nested value (strings raw, everything else as JSON)
json_get() {
  python3 -c '
import json, sys
value = json.load(sys.stdin)
for key in sys.argv[1:]:
    value = value[key]
print(value if isinstance(value, str) else json.dumps(value))
' "$@"
}

poll() {
  local job_id="$1" deadline=$((SECONDS + POLL_TIMEOUT_S)) job status
  while :; do
    job="$(api "${API_URL}/v1/transcriptions/${job_id}")"
    status="$(json_get status <<<"$job")"
    case "$status" in
      succeeded) printf '%s' "$job"; return 0 ;;
      failed) echo "job ${job_id} failed: $(json_get error <<<"$job")" >&2; return 1 ;;
    esac
    if ((SECONDS >= deadline)); then
      echo "job ${job_id} still ${status} after ${POLL_TIMEOUT_S}s" >&2
      return 1
    fi
    echo "  ${job_id}: ${status}" >&2
    sleep 2
  done
}

echo "==> health: ${API_URL}/healthz"
curl -sS --fail-with-body "${API_URL}/healthz"
echo

echo "==> direct upload: samples/hello.mp3"
job="$(api -X POST -H "Content-Type: audio/mpeg" --data-binary @samples/hello.mp3 \
  "${API_URL}/v1/transcriptions")"
job_id="$(json_get id <<<"$job")"
job="$(poll "$job_id")"
echo "transcript: $(json_get result text <<<"$job")"

file=samples/monologue.mp3
size_bytes="$(wc -c <"$file" | tr -d ' ')"
echo "==> presigned upload: ${file} (${size_bytes} bytes)"
upload="$(api -X POST -H "Content-Type: application/json" \
  -d "{\"size_bytes\": ${size_bytes}, \"content_type\": \"audio/mpeg\"}" \
  "${API_URL}/v1/uploads")"
# S3 requires the policy fields before the file part; --form-string sends values verbatim
# (no @file / <file expansion).
form=()
while IFS= read -r -d '' field; do
  form+=(--form-string "$field")
done < <(python3 -c '
import json, sys
for name, value in json.load(sys.stdin)["fields"].items():
    sys.stdout.write(f"{name}={value}\0")
' <<<"$upload")
curl -sS --fail-with-body -X POST "${form[@]}" -F "file=@${file};type=audio/mpeg" \
  "$(json_get url <<<"$upload")"
echo "uploaded to S3"

job="$(api -X POST -H "Content-Type: application/json" \
  -d "{\"upload_id\": \"$(json_get upload_id <<<"$upload")\"}" \
  "${API_URL}/v1/transcriptions")"
job_id="$(json_get id <<<"$job")"
poll "$job_id" >/dev/null

echo "==> subtitles (SRT) for ${job_id}"
api "${API_URL}/v1/transcriptions/${job_id}/subtitles?format=srt"
