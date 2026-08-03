# Lenience patch for vLLM 0.26.0

vLLM exposes no acceptance-threshold knob: `rejection_sample_method` offers only
`standard` (min(1, p/q)), `synthetic` (prescribed rate, ignores p and q) and
`block`. Lenience — `min(1, p/(λ·q))` — needs one extra term in the verifier.

```
patches/
  vllm-0.26.0-lenience.patch   the whole change: 2 files, ~110 added lines
  apply.sh                     verify, apply, re-verify, test
  test_lenience.py             acceptance test, run by apply.sh
```

## What it changes

| file | why | form |
|---|---|---|
| `vllm/v1/sample/rejection_sampler.py` | the runner this config actually uses | linear: `p / (q*λ) >= u` |
| `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py` | the V2 runner | log: `log p > log u + log q + log λ` |

**Both are required.** vLLM 0.26.0 ships two model runners; `use_v2_model_runner`
is unset by default and this config selects V1, so patching only the V2 kernel
changes nothing at all and the lossy arm silently equals the strict arm.

Everything else in the diff is plumbing: reading λ at import, threading it to
the kernel launch, and one log line. `git diff` shows the semantic change is the
acceptance expression and nothing else.

## Naming: λ, not β

The knob is the lenience **factor** λ ∈ (0,1]. In the mentored-decoding notation
of Xia et al. this is **1 − α**; their β is a different parameter, fixed at 1 in
their Table 2. Calling this factor β misidentifies the method, so the code, the
CLI (`--lenience-factor`) and the archived metadata all say λ.

Rejection recovery is untouched: rejected positions still resample from the
stock residual `norm(max(0, p − q))` and the bonus token still comes from `p`.
Only the acceptance test moves.

## Apply

```bash
bash patches/apply.sh
```

It refuses to guess. It checks the vLLM version, checks the sha256 of both
target files against the recorded upstream hashes, dry-runs the patch, applies
it, re-checks the resulting hashes, and runs `test_lenience.py`. Re-running on an
already-patched install verifies and exits 0.

| file | upstream sha256 | patched sha256 |
|---|---|---|
| `v1/sample/rejection_sampler.py` | `840ec899…` | `036d1538…` |
| `v1/worker/gpu/spec_decode/rejection_sampler_utils.py` | `bfaec14e…` | `0ad55a1c…` |

`scripts/run_experiment_vllm.py` records the live hashes in every `config.json`,
so a run directory proves which verifier produced it.

Installing copies with `cp --remove-destination`. A plain `cp` writes *through*
the hardlink uv keeps into `~/.cache/uv/archive-v0`, which silently patches the
cached wheel for every other venv built from it — that already happened once.

## Selecting λ

λ is read from `/tmp/lossy-spec-decode-lenience-$UID`, written by
`remote/run_server_vllm.sh` before the server starts, in **every** mode
(baseline and strict write `1.0`, so a stale value cannot turn a control arm
lossy). The path is uid-scoped and identical in the patch and the shell script;
it deliberately does not depend on where the repo is cloned.

Not an environment variable: vLLM spawns EngineCore with a sanitised
environment, so env vars never reach the sampler. Confirm what was actually in
force from the server log:

```
(EngineCore pid=...) [LENIENCE PATCH v1] pid=... lenience_factor=0.2 (/tmp/lossy-spec-decode-lenience-1003)
```

The experiment runner scrapes that line from the server log when given
`--server-log` and refuses to write a run directory if it disagrees with the
requested factor.

## Testing without a model

```bash
.venv-vllm/bin/python patches/test_lenience.py
```

Checks (1) both modules read the factor file and fall back to 1.0 without it,
and (2) — on a GPU, no model or server needed — that the V1 verify kernel
accepts exactly when `u <= p/(q·λ)`, and that λ=0.2 accepts strictly more than
λ=1.0. Those are the two failures this repo has actually hit: the factor never
reaching the sampler, and the patched file not being the live one.

## Version

Version-specific: in vLLM 0.20.1 the V2 file was named
`probabilistic_rejection_sampler_utils.py` and the method was `probabilistic`.
The hunks are line-addressed against 0.26.0; after any vLLM change, re-derive
the diff by hand rather than forcing it.
