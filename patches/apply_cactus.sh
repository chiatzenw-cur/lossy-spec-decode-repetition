#!/usr/bin/env bash
# Apply (or verify) the CACTUS patch against an installed vLLM 0.26.0.
#
# Mutually exclusive with the Lenience and spec-casc-opt patches: all three
# touch the same pristine vllm/v1/sample/rejection_sampler.py, so only one
# can be applied to a given install at a time. Switching means reversing
# whichever is currently applied (or reinstalling vLLM 0.26.0 fresh) before
# running this script.
#
# Idempotent: re-running on an already-patched install verifies and exits 0.
# Refuses to guess if the target file is neither pristine 0.26.0 nor exactly
# the expected patched result.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-$(cd "$here/.." && pwd)/.venv-vllm}"
PYTHON="${PYTHON:-$VENV/bin/python}"
PATCH_FILE="$here/vllm-0.26.0-cactus.patch"
REL="v1/sample/rejection_sampler.py"

EXPECT_VERSION="0.26.0"
UPSTREAM_SHA="840ec8995f909eae7525f1bfcda5b6f23dbe77ce3ccb964bec14c97932ba2e76"
PATCHED_SHA="02492f03bdf90c9442bb4bca81c61b82c06ad34b733a295f9305326941a93068"
# The other two patches' results, so a wrong-patch-applied state fails with a
# specific, actionable message instead of "unrecognised file".
LENIENCE_PATCHED_SHA="81a0947d7263675a07125b714b3093fbd82f91e3211a642a4d0ec448ad2b898d"
SPEC_CASC_OPT_PATCHED_SHA="de32559fa494f8b4b88df34874793001d066492cd034f88e046fcd63af0de85d"

hash_of() { sha256sum "$1" | cut -d' ' -f1; }

version="$("$PYTHON" -c 'import vllm; print(vllm.__version__)' 2>/dev/null | cut -d+ -f1 || true)"
if [[ "$version" != "$EXPECT_VERSION" ]]; then
  echo "patch targets vLLM $EXPECT_VERSION, found ${version:-none} (python: $PYTHON)" >&2
  echo "the hunks are line-addressed against 0.26.0; re-derive them by hand for another version" >&2
  exit 1
fi
pkg="$("$PYTHON" -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).parent)')"
target="$pkg/$REL"

[[ -f "$target" ]] || { echo "missing target: $target" >&2; exit 1; }
got="$(hash_of "$target")"
if [[ "$got" == "$PATCHED_SHA" ]]; then
  echo "already applied ($target matches the expected CACTUS sha256)"
elif [[ "$got" == "$LENIENCE_PATCHED_SHA" ]]; then
  echo "the Lenience patch is currently applied to $target, not CACTUS" >&2
  echo "reverse it first (patch -p1 -R -d ... < $here/vllm-0.26.0-lenience.patch), or reinstall vLLM 0.26.0 fresh" >&2
  exit 1
elif [[ "$got" == "$SPEC_CASC_OPT_PATCHED_SHA" ]]; then
  echo "the spec-casc-opt patch is currently applied to $target, not CACTUS" >&2
  echo "reverse it first (patch -p1 -R -d ... < $here/vllm-0.26.0-spec-casc-opt.patch), or reinstall vLLM 0.26.0 fresh" >&2
  exit 1
elif [[ "$got" == "$UPSTREAM_SHA" ]]; then
  work="$(mktemp -d)"
  trap 'rm -rf "$work"' EXIT
  mkdir -p "$work/vllm/$(dirname "$REL")"
  cp "$target" "$work/vllm/$REL"
  ( cd "$work" && patch -p1 --dry-run --forward < "$PATCH_FILE" >/dev/null )
  ( cd "$work" && patch -p1 --forward --no-backup-if-mismatch < "$PATCH_FILE" >/dev/null )
  got2="$(hash_of "$work/vllm/$REL")"
  [[ "$got2" == "$PATCHED_SHA" ]] || {
    echo "patched $REL has sha256 $got2, expected $PATCHED_SHA" >&2; exit 1
  }
  cp --remove-destination "$work/vllm/$REL" "$target"
  echo "patched $target"
  find "$(dirname "$pkg")/vllm" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
else
  echo "unrecognised $REL" >&2
  echo "  sha256              $got" >&2
  echo "  pristine            $UPSTREAM_SHA" >&2
  echo "  cactus              $PATCHED_SHA" >&2
  echo "  lenience (wrong arm) $LENIENCE_PATCHED_SHA" >&2
  echo "  spec-casc-opt (wrong arm) $SPEC_CASC_OPT_PATCHED_SHA" >&2
  echo "reinstall vLLM 0.26.0 to get back to a known state" >&2
  exit 1
fi

cp "$here/lenience_trace.py" "$pkg/v1/sample/lenience_trace.py"
echo "installed $pkg/v1/sample/lenience_trace.py"

echo "running acceptance test..."
"$PYTHON" "$here/test_cactus.py"
echo
echo "select alpha by writing it to /tmp/lossy-spec-decode-cactus-alpha-$(id -u)"
echo "(alpha >= 0; larger alpha boosts acceptance more; alpha=0 recovers strict spec-dec exactly)"
