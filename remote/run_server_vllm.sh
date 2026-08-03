#!/usr/bin/env bash
# Start one vLLM server mode for the lossy-verification pilot.
#
# Environment note: the paper's setup is vLLM 0.20.1 + torch 2.11.0+cu130 on a
# single H100/B200 with no parallelism. This box's driver is 570 (CUDA 12.8), so
# cu130 cannot load; we build for cu129, which works via CUDA minor-version
# compatibility. We now run vLLM 0.26.0 rather than 0.20.1 because 0.20.1 has no
# draft_sample_method knob and always drafts greedily, which the nebius model
# card notes underestimates acceptance at temperature > 0.
# See remote/ENVIRONMENT.md.
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "baseline" && "$MODE" != "strict" && "$MODE" != "lossy" ]]; then
  echo "usage: $0 baseline|strict|lossy" >&2
  exit 2
fi

PYTHON="${PYTHON:-python3}"
python_bin_dir="$(cd "$(dirname "$PYTHON")" && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_HOME
export PATH="$python_bin_dir:$CUDA_HOME/bin:$PATH"

MODEL_PATH="${MODEL_PATH:-openai/gpt-oss-20b}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-nebius/EAGLE3-gpt-oss-20b}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
SEED="${SEED:-0}"
NUM_SPEC="${NUM_SPEC:-6}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_UTIL="${GPU_UTIL:-0.85}"

# Lossy knob.
#
# LOSSY_RULE=lenience uses the patched kernel: accept iff
#   p(x)/(LENIENCE_FACTOR*q(x)) > u. Needs `bash patches/apply.sh`.
# LOSSY_RULE=synthetic uses vLLM's stock `synthetic` method, which accepts at a
#   prescribed rate irrespective of p and q. No patch needed.
LOSSY_RULE="${LOSSY_RULE:-synthetic}"
LENIENCE_FACTOR="${LENIENCE_FACTOR:-0.2}"
SYNTH_LEN="${SYNTH_LEN:-3.0}"

# The knob used to be called BETA. It is renamed because beta is a *different*
# parameter in the mentored-decoding literature (fixed at 1 there); this factor
# is 1 - alpha. Fail rather than ignore a stale BETA=... in someone's shell
# history, which would silently run the wrong arm.
if [[ -n "${BETA:-}" ]]; then
  echo "BETA is no longer read; use LENIENCE_FACTOR=$BETA instead" >&2
  exit 2
fi

# The lenience factor reaches the sampler through this file, never through the
# environment: vLLM spawns EngineCore with a sanitised environment, verified via
# /proc/<pid>/environ. The path is uid-scoped under /tmp and hardcoded
# identically in the patch, so it does not depend on where the repo is cloned.
#
# Written in EVERY mode, including baseline and strict: if it is left over from
# an earlier lossy run, a strict server started afterwards silently picks up the
# stale factor and the control arm is not a control arm.
factor_file="/tmp/lossy-spec-decode-lenience-$(id -u)"

common_args=(
  --model "$MODEL_PATH"
  --served-model-name gpt-oss-20b
  --host "$HOST"
  --port "$PORT"
  --seed "$SEED"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_UTIL"
  # Paper setup: single generation job, no TP/PP/DP/FSDP.
  --tensor-parallel-size 1
  --pipeline-parallel-size 1
  # Required for replay. With prefix caching on, a request that reuses cached
  # prompt KV takes a different numeric path than one that recomputes it, so the
  # same prompt+seed diverges depending on what ran before it. Measured on
  # case_001: three server instances gave 1686 / 1505 / 1640 tokens, and a warm
  # request differed from the cold first request on the same server, while two
  # warm requests were bit-identical. Same failure SGLang's radix cache caused.
  --no-enable-prefix-caching
)

# draft_sample_method=probabilistic makes the drafter sample stochastically and
# caches its logits, so verification uses a real q. The default is greedy, which
# the nebius model card notes underestimates acceptance at temperature > 0 —
# vLLM 0.20.1 had no such knob, so pre-upgrade acceptance numbers are pessimistic.
DRAFT_SAMPLE_METHOD="${DRAFT_SAMPLE_METHOD:-probabilistic}"

spec_json() {
  local method="$1"
  local extra="$2"
  printf '{"method":"eagle3","model":"%s","num_speculative_tokens":%s,"rejection_sample_method":"%s","draft_sample_method":"%s"%s}' \
    "$DRAFT_MODEL_PATH" "$NUM_SPEC" "$method" "$DRAFT_SAMPLE_METHOD" "$extra"
}

case "$MODE" in
  baseline)
    printf '%s\n' "1.0" > "$factor_file"
    echo "mode=baseline model=$MODEL_PATH port=$PORT seed=$SEED lenience_factor=1.0"
    exec "$PYTHON" -m vllm.entrypoints.openai.api_server "${common_args[@]}"
    ;;
  strict)
    # `standard` is probabilistic rejection sampling, min(1, p/q): lossless with
    # respect to the target distribution. Paired with draft_sample_method, this
    # is the control arm the lossy rules are compared against.
    printf '%s\n' "1.0" > "$factor_file"
    cfg="$(spec_json standard '')"
    echo "mode=strict draft=$DRAFT_MODEL_PATH k=$NUM_SPEC rule=standard draft_sample=$DRAFT_SAMPLE_METHOD port=$PORT seed=$SEED lenience_factor=1.0"
    exec "$PYTHON" -m vllm.entrypoints.openai.api_server "${common_args[@]}" --speculative-config "$cfg"
    ;;
  lossy)
    if [[ "$LOSSY_RULE" == "synthetic" ]]; then
      printf '%s\n' "1.0" > "$factor_file"
      cfg="$(spec_json synthetic ",\"synthetic_acceptance_length\":$SYNTH_LEN")"
      echo "mode=lossy rule=synthetic draft=$DRAFT_MODEL_PATH k=$NUM_SPEC accept_len=$SYNTH_LEN port=$PORT seed=$SEED"
    else
      # Lenience runs the same `standard` code path as the strict arm; the only
      # difference is the factor in the patched kernel, which is exactly the
      # single-variable contrast the pilot needs. Refuse to start unless the
      # patch is actually present: without it this silently becomes the strict
      # arm and produces a null result that looks like a real measurement.
      # Module path is version-specific: vLLM 0.20.1 called this
      # probabilistic_rejection_sampler_utils, 0.26.0 renamed it.
      # Write the factor BEFORE the probe, not just before the server: the probe
      # imports the patched module too, and announces whatever it reads to
      # stderr, which lands in the same log the runner scrapes. Writing after it
      # leaves a stale 1.0 line in a lossy run's log.
      printf '%s\n' "$LENIENCE_FACTOR" > "$factor_file"
      probe="$("$PYTHON" - <<'PY'
try:
    import vllm.v1.sample.rejection_sampler as v1
    import vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils as v2
    ok = hasattr(v1, "_LENIENCE_FACTOR") and hasattr(v2, "_LENIENCE_LOG_FACTOR")
    print("yes" if ok else "no")
except Exception:
    print("no")
PY
)"
      if ! grep -qx "yes" <<<"$probe"; then
        echo "LOSSY_RULE=lenience needs the patch: run 'bash patches/apply.sh'." >&2
        echo "Without it this arm silently equals the strict arm." >&2
        exit 5
      fi
      cfg="$(spec_json standard '')"
      echo "mode=lossy rule=lenience lenience_factor=$LENIENCE_FACTOR (via $factor_file) draft=$DRAFT_MODEL_PATH k=$NUM_SPEC port=$PORT seed=$SEED"
    fi
    exec "$PYTHON" -m vllm.entrypoints.openai.api_server "${common_args[@]}" --speculative-config "$cfg"
    ;;
esac
