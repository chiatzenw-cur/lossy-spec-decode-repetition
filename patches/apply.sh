#!/usr/bin/env bash
# Apply (or verify) the Lenience patch against an installed vLLM 0.26.0.
#
# Idempotent: re-running on an already-patched install verifies and exits 0.
# Refuses to guess if the target files are neither pristine 0.26.0 nor exactly
# the expected patched result.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-$(cd "$here/.." && pwd)/.venv-vllm}"
PYTHON="${PYTHON:-$VENV/bin/python}"
PATCH_FILE="$here/vllm-0.26.0-lenience.patch"

EXPECT_VERSION="0.26.0"
# sha256 of the two files, before and after the patch. Recorded so the artifact
# proves which bytes were in the interpreter, rather than relying on a directory
# name. run_experiment_vllm.py records the live hashes into every config.json.
declare -A UPSTREAM=(
  ["v1/sample/rejection_sampler.py"]="840ec8995f909eae7525f1bfcda5b6f23dbe77ce3ccb964bec14c97932ba2e76"
  ["v1/worker/gpu/spec_decode/rejection_sampler_utils.py"]="bfaec14e220cf669ddfd2b3ef6d2c556dc8fae60ef71e65a2e7ae068fb386a2a"
)
declare -A PATCHED=(
  ["v1/sample/rejection_sampler.py"]="036d1538652634a536470e6e702d894772c166b3ad5a504ace54cf4d4421acca"
  ["v1/worker/gpu/spec_decode/rejection_sampler_utils.py"]="0ad55a1cb39f2306c78a170fdb6468a36b55643c6610c85cd46c908e0d313112"
)

hash_of() { sha256sum "$1" | cut -d' ' -f1; }

version="$("$PYTHON" -c 'import vllm; print(vllm.__version__)' 2>/dev/null | cut -d+ -f1 || true)"
if [[ "$version" != "$EXPECT_VERSION" ]]; then
  echo "patch targets vLLM $EXPECT_VERSION, found ${version:-none} (python: $PYTHON)" >&2
  echo "the hunks are line-addressed against 0.26.0; re-derive them by hand for another version" >&2
  exit 1
fi
pkg="$("$PYTHON" -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).parent)')"
sp="$(dirname "$pkg")"

state=""
for rel in "${!UPSTREAM[@]}"; do
  target="$pkg/$rel"
  [[ -f "$target" ]] || { echo "missing target: $target" >&2; exit 1; }
  got="$(hash_of "$target")"
  if [[ "$got" == "${PATCHED[$rel]}" ]]; then
    this="patched"
  elif [[ "$got" == "${UPSTREAM[$rel]}" ]]; then
    this="upstream"
  else
    echo "unrecognised $rel" >&2
    echo "  sha256   $got" >&2
    echo "  upstream ${UPSTREAM[$rel]}" >&2
    echo "  patched  ${PATCHED[$rel]}" >&2
    echo "reinstall vLLM 0.26.0 to get back to a known state" >&2
    exit 1
  fi
  [[ -z "$state" || "$state" == "$this" ]] || {
    echo "the two files disagree: one is $state, the other $this" >&2
    echo "reinstall vLLM 0.26.0 and re-apply, or the lossy arm may run half-patched" >&2
    exit 1
  }
  state="$this"
done

if [[ "$state" == "patched" ]]; then
  echo "already applied (both files match the expected patched sha256)"
else
  work="$(mktemp -d)"
  trap 'rm -rf "$work"' EXIT
  for rel in "${!UPSTREAM[@]}"; do
    mkdir -p "$work/vllm/$(dirname "$rel")"
    cp "$pkg/$rel" "$work/vllm/$rel"
  done
  ( cd "$work" && patch -p1 --dry-run --forward < "$PATCH_FILE" >/dev/null )
  ( cd "$work" && patch -p1 --forward --no-backup-if-mismatch < "$PATCH_FILE" >/dev/null )
  for rel in "${!UPSTREAM[@]}"; do
    got="$(hash_of "$work/vllm/$rel")"
    [[ "$got" == "${PATCHED[$rel]}" ]] || {
      echo "patched $rel has sha256 $got, expected ${PATCHED[$rel]}" >&2; exit 1
    }
    # --remove-destination unlinks first. Without it, cp writes THROUGH the
    # hardlink uv keeps into ~/.cache/uv/archive-v0, silently patching the
    # cached wheel for every other venv built from it.
    cp --remove-destination "$work/vllm/$rel" "$pkg/$rel"
    echo "patched $pkg/$rel"
  done
  find "$sp/vllm" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
fi

echo "running acceptance test..."
"$PYTHON" "$here/test_lenience.py"
echo
echo "select the lenience factor by writing it to /tmp/lossy-spec-decode-lenience-$(id -u)"
echo "(remote/run_server_vllm.sh does this for every mode, including strict)"
