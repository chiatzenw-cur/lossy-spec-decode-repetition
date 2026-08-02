#!/usr/bin/env bash
# Re-apply both Lenience patches. Only valid for vLLM 0.26.0 (see README.md).
set -euo pipefail
VENV="${VENV:-.venv-vllm}"
here="$(cd "$(dirname "$0")" && pwd)"
have="$("$VENV/bin/python" -c 'import vllm;print(vllm.__version__)' 2>/dev/null | cut -d+ -f1)"
[[ "$have" == "0.26.0" ]] || { echo "patch targets vLLM 0.26.0, found ${have:-none}; re-apply by hand" >&2; exit 1; }
sp="$VENV/lib/python3.12/site-packages/vllm"
declare -A M=(
  ["$here/rejection_sampler.v1.vllm-0.26.0.patched.py"]="$sp/v1/sample/rejection_sampler.py"
  ["$here/rejection_sampler_utils.vllm-0.26.0.patched.py"]="$sp/v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
)
for src in "${!M[@]}"; do
  dst="${M[$src]}"
  [[ -f "$dst" ]] || { echo "missing target: $dst" >&2; exit 1; }
  cp "$dst" "$dst.orig.bak"; cp "$src" "$dst"; echo "patched $dst"
done
echo "select beta by writing it to .lenience_beta (run_server_vllm.sh does this)"
