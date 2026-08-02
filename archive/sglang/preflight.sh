#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"

echo "== disk =="
df -h "${HF_HOME:-$HOME/.cache/huggingface}" 2>/dev/null || df -h "$HOME"

echo "== gpu =="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

echo "== versions =="
"$PYTHON" - <<'PY'
import importlib.metadata
import sys

print("python", sys.version.split()[0])
for package in ("sglang", "torch", "transformers", "huggingface-hub"):
    try:
        print(package, importlib.metadata.version(package))
    except importlib.metadata.PackageNotFoundError:
        print(package, "NOT INSTALLED")
PY

echo "== required SGLang flags =="
help_text="$("$PYTHON" -m sglang.launch_server --help)"
for flag in \
  speculative-algorithm \
  speculative-draft-model-path \
  speculative-accept-threshold-acc; do
  if grep -q -- "--$flag" <<<"$help_text"; then
    echo "$flag: yes"
  else
    echo "$flag: NO"
  fi
done

echo "== cached model snapshots =="
cache_root="${HF_HOME:-$HOME/.cache/huggingface}/hub"
for repo in models--openai--gpt-oss-20b models--nebius--EAGLE3-gpt-oss-20b; do
  if [[ -d "$cache_root/$repo" ]]; then
    du -sh "$cache_root/$repo"
  else
    echo "$repo: not cached"
  fi
done
