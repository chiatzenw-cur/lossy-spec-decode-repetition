#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "baseline" && "$MODE" != "strict" && "$MODE" != "lossy" ]]; then
  echo "usage: $0 baseline|strict|lossy" >&2
  exit 2
fi

MODEL_PATH="${MODEL_PATH:-openai/gpt-oss-20b}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-nebius/EAGLE3-gpt-oss-20b}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-1}"
LENIENCE="${LENIENCE:-0.2}"
SPEC_STEPS="${SPEC_STEPS:-6}"
SPEC_DRAFT_TOKENS="${SPEC_DRAFT_TOKENS:-7}"

if ! [[ "$LENIENCE" =~ ^(0\.[0-9]*[1-9][0-9]*|1(\.0+)?)$ ]]; then
  echo "LENIENCE must be in (0, 1]; got: $LENIENCE" >&2
  exit 2
fi

python3 -m sglang.launch_server --help | grep -q -- "--speculative-accept-threshold-acc" || {
  echo "This SGLang build has no --speculative-accept-threshold-acc flag." >&2
  echo "Use a newer SGLang environment before running the lossy experiment." >&2
  exit 3
}

common_args=(
  --model-path "$MODEL_PATH"
  --served-model-name gpt-oss-20b
  --host "$HOST"
  --port "$PORT"
  --tp-size "$TP_SIZE"
  --enable-metrics
  --decode-log-interval 1
)

if [[ "$MODE" == "baseline" ]]; then
  echo "mode=baseline model=$MODEL_PATH port=$PORT"
  exec python3 -m sglang.launch_server "${common_args[@]}"
fi

spec_args=(
  --speculative-algorithm EAGLE3
  --speculative-draft-model-path "$DRAFT_MODEL_PATH"
  --speculative-num-steps "$SPEC_STEPS"
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens "$SPEC_DRAFT_TOKENS"
  --speculative-accept-threshold-single 1.0
)

if [[ "$MODE" == "strict" ]]; then
  echo "mode=strict model=$MODEL_PATH draft=$DRAFT_MODEL_PATH threshold_acc=1.0 port=$PORT"
  exec python3 -m sglang.launch_server \
    "${common_args[@]}" \
    "${spec_args[@]}" \
    --speculative-accept-threshold-acc 1.0
fi

echo "mode=lossy model=$MODEL_PATH draft=$DRAFT_MODEL_PATH lenience=$LENIENCE port=$PORT"
exec python3 -m sglang.launch_server \
  "${common_args[@]}" \
  "${spec_args[@]}" \
  --speculative-accept-threshold-acc "$LENIENCE"
