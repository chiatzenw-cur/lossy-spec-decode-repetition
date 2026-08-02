# Lenience (beta) patch for vLLM 0.26.0

vLLM exposes no acceptance-threshold knob: `rejection_sample_method` offers only
`standard` (min(1, p/q)), `synthetic` (prescribed rate, ignores p and q) and
`block`. Lenience — `min(1, p/(beta*q))` — needs one extra term in the verifier.

## Files

| file | patches | form |
|---|---|---|
| `rejection_sampler.v1.vllm-0.26.0.patched.py` | `vllm/v1/sample/rejection_sampler.py` | linear: `p / (q*beta) >= u` |
| `rejection_sampler_utils.vllm-0.26.0.patched.py` | `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py` | log: `log p > log u + log q + log beta` |

**Both are required.** vLLM 0.26.0 ships two model runners; `use_v2_model_runner`
is unset by default and this config selects V1, so patching only the V2 kernel
changes nothing at all and the lossy arm silently equals the strict arm.

`beta = 1.0` is a mathematical no-op in both forms (`*1.0`, `+0.0`), so the
strict control arm is provably unaffected by the patch being present.

## Selecting beta

Written to `.lenience_beta` in the repo root by `remote/run_server_vllm.sh`.
Not an environment variable: vLLM spawns EngineCore with a sanitised environment,
so env vars never reach the sampler. Confirm the value actually in force from the
server log:

```
(EngineCore pid=...) [LENIENCE PATCH v1] pid=... beta=0.2 (from .../.lenience_beta)
```

## Apply

```bash
bash patches/apply.sh     # refuses to run against a vLLM other than 0.26.0
```

Version-specific: in vLLM 0.20.1 the V2 file was named
`probabilistic_rejection_sampler_utils.py` and the method was `probabilistic`.
After any vLLM change, re-apply by hand rather than copying these files over.
