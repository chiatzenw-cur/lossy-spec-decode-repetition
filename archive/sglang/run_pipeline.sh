#!/usr/bin/env bash
# Start one SGLang server mode, run every pilot prompt against it, summarize, stop.
set -euo pipefail

MODE="${1:-baseline}"
shift || true
if [[ "$MODE" != "baseline" && "$MODE" != "strict" && "$MODE" != "lossy" ]]; then
  echo "usage: $0 [baseline|strict|lossy] [extra run_lossy_experiment.py args...]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
LENIENCE="${LENIENCE:-0.2}"
READY_TIMEOUT="${READY_TIMEOUT:-3600}"
LOG_DIR="${LOG_DIR:-logs}"

if [[ "$MODE" == "lossy" ]]; then
  TAG="lossy_l${LENIENCE//./p}"
  CLIENT_ARGS=(--mode lossy --lenience "$LENIENCE")
else
  TAG="$MODE"
  CLIENT_ARGS=(--mode "$MODE")
fi

mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SERVER_LOG="$LOG_DIR/server-$TAG-$STAMP.log"
CLIENT_LOG="$LOG_DIR/client-$TAG-$STAMP.log"
ENV_LOG="$LOG_DIR/environment-preflight-$STAMP.txt"

base_url="http://$HOST:$PORT"
if curl -sf --max-time 5 "$base_url/get_server_info" >/dev/null 2>&1; then
  echo "a server is already listening on $base_url; stop it before starting $MODE" >&2
  exit 3
fi

echo "== preflight -> $ENV_LOG =="
bash remote/preflight.sh >"$ENV_LOG" 2>&1 || echo "preflight reported problems; see $ENV_LOG" >&2
nvidia-smi >>"$ENV_LOG" 2>&1 || true

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "== stopping server (pid $SERVER_PID) =="
    kill -INT "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do
      kill -0 "$SERVER_PID" 2>/dev/null || return 0
      sleep 1
    done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "== starting $MODE server -> $SERVER_LOG =="
PYTHON="$PYTHON" HOST="$HOST" PORT="$PORT" LENIENCE="$LENIENCE" \
  bash remote/run_server.sh "$MODE" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "== waiting up to ${READY_TIMEOUT}s for $base_url =="
ready=0
for _ in $(seq 1 "$READY_TIMEOUT"); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server exited before becoming ready; last lines of $SERVER_LOG:" >&2
    tail -n 40 "$SERVER_LOG" >&2
    exit 4
  fi
  if curl -sf --max-time 5 "$base_url/get_server_info" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "$ready" -ne 1 ]]; then
  echo "server did not become ready within ${READY_TIMEOUT}s; see $SERVER_LOG" >&2
  exit 5
fi
echo "server ready after $(grep -c '' "$SERVER_LOG") log lines"

echo "== running prompts -> $CLIENT_LOG =="
set +e
"$PYTHON" scripts/run_lossy_experiment.py \
  "${CLIENT_ARGS[@]}" \
  --server-url "$base_url" \
  "$@" 2>&1 | tee "$CLIENT_LOG"
client_status="${PIPESTATUS[0]}"
set -e

echo "== summarizing tag=$TAG =="
"$PYTHON" scripts/summarize_runs.py --tags "$TAG" || true

echo "server log:  $SERVER_LOG"
echo "client log:  $CLIENT_LOG"
echo "environment: $ENV_LOG"
exit "$client_status"
