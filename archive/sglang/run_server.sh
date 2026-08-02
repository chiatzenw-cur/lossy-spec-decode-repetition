#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "baseline" && "$MODE" != "strict" && "$MODE" != "lossy" ]]; then
  echo "usage: $0 baseline|strict|lossy" >&2
  exit 2
fi

PYTHON="${PYTHON:-python3}"

# SGLang JIT-compiles CUDA kernels at startup (sglang/jit_kernel via tvm_ffi) and
# shells out to `ninja` and `nvcc`. Neither is found when PYTHON is given as a
# bare venv path, because that does not put the venv's bin directory on PATH.
python_bin_dir="$(cd "$(dirname "$PYTHON")" && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_HOME
export PATH="$python_bin_dir:$CUDA_HOME/bin:$PATH"
for tool in ninja nvcc; do
  command -v "$tool" >/dev/null || {
    echo "$tool not found on PATH; SGLang cannot JIT-compile its kernels." >&2
    echo "PATH=$PATH" >&2
    exit 4
  }
done

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

"$PYTHON" -m sglang.launch_server --help | grep -q -- "--speculative-accept-threshold-acc" || {
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

# Per-request `sampling_seed` is silently dropped unless deterministic inference
# is on: sampling_batch_info.py builds the seed tensor only when
# server_args.enable_deterministic_inference is set, and otherwise passes None,
# leaving sampling on a global RNG that is reseeded randomly at every launch.
# Without this, runs are not replayable and paired comparison is confounded.
# --disable-radix-cache is not optional here. With the prefix cache on, two
# identical seeded requests share prompt KV computed by an earlier request's
# chunked prefill instead of recomputing it, and the trajectories diverge a few
# hundred tokens in. Measured on case_003: radix on -> 5818 vs 991 tokens for the
# same seed; radix off -> byte-identical token ids across repeats. SGLang treats
# fa3 as radix-compatible under deterministic inference, but that does not hold
# for gpt-oss-20b on this build.
if [[ "${DETERMINISTIC:-0}" == "1" ]]; then
  common_args+=(
    --enable-deterministic-inference
    --random-seed "${RANDOM_SEED:-0}"
    --disable-radix-cache
  )
fi

# Escape hatch for one-off diagnostics (e.g. --disable-radix-cache when checking
# what breaks replay). Word-split deliberately; keep it out of recorded runs.
if [[ -n "${EXTRA_SERVER_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  common_args+=(${EXTRA_SERVER_ARGS})
fi

if [[ "$MODE" == "baseline" ]]; then
  echo "mode=baseline model=$MODEL_PATH port=$PORT"
  exec "$PYTHON" -m sglang.launch_server "${common_args[@]}"
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
  exec "$PYTHON" -m sglang.launch_server \
    "${common_args[@]}" \
    "${spec_args[@]}" \
    --speculative-accept-threshold-acc 1.0
fi

echo "mode=lossy model=$MODEL_PATH draft=$DRAFT_MODEL_PATH lenience=$LENIENCE port=$PORT"
exec "$PYTHON" -m sglang.launch_server \
  "${common_args[@]}" \
  "${spec_args[@]}" \
  --speculative-accept-threshold-acc "$LENIENCE"
