# Lossy speculative decoding on GPT-OSS-20B

Does relaxing the speculative-decoding acceptance rule make GPT-OSS-20B degenerate?

**Yes, measurably.** With Lenience at β=0.2, 3 of 10 AIME24 problems run to the
token cap without ever producing an answer, where the lossless rule answers all
10. See **[RESULTS.md](RESULTS.md)** for the data and the honest caveats.

## Layout

```
RESULTS.md              findings + the actual generated outputs
DESIGN.md               original experiment design (written for the SGLang phase)
remote/
  ENVIRONMENT.md        how the environment is pinned, and why
  run_server_vllm.sh    start baseline | strict | lossy
  stop_server.sh        stop and wait for the GPU to actually free
scripts/
  build_aime24_prompts.py   render AIME24 into Harmony prompts
  run_experiment_vllm.py    send prompts, archive runs, record acceptance
  summarize_runs.py         aggregate run dirs into a table
patches/                Lenience (beta) patch for vLLM + how to re-apply
prompts/aime24/         10 AIME24 problems, reasoning_effort=medium
runs/aime24/            the data: case_XXX/seed_0/<arm>/
logs/                   server logs for the two surviving runs
archive/sglang/         superseded SGLang phase (see below)
```

Each run directory holds `config.json`, `prompt.txt`, `request.json`,
`response.json`, `output.txt`, `run.json`, `server_info.json`.

## Setup

The venv is not committed. Rebuild it with the commands in
[remote/ENVIRONMENT.md](remote/ENVIRONMENT.md), then re-apply the Lenience patch:

```bash
bash patches/apply.sh
```

## Four traps this setup encodes

Each of these silently produced wrong or misleading data before being caught.
They are worth knowing before changing anything.

**1. vLLM has two model runners.** `use_v2_model_runner` is unset by default and
this config selects **V1**, which uses `vllm/v1/sample/rejection_sampler.py`.
Patching only the V2 kernel (`gpu/spec_decode/rejection_sampler_utils.py`) has no
effect whatsoever — the lossy arm comes out bit-identical to strict. Both files
are patched so the result does not depend on which runner vLLM picks.

**2. Environment variables do not reach the sampler.** vLLM spawns EngineCore
with a sanitised environment; a variable present in the API server's
`/proc/<pid>/environ` is absent from EngineCore's. β is therefore passed through
the file `.lenience_beta`, and the patched module prints
`[LENIENCE PATCH v1] beta=...` to stderr at import so the server log carries
positive proof of the value actually in force.

**3. Results depend on request order.** Even with prefix caching disabled, a
prompt gives different output as the 1st vs the 2nd request on a server
(case_001: 1,711 tokens vs 2,485). **Arms are only comparable when each is run on
a freshly started server with the same cases in the same order.** Comparing a
3-case run against a 10-case run compares nothing.

**4. Stopping the server needs `remote/stop_server.sh`.** vLLM renames its worker
to `VLLM::EngineCore`, so `pkill -f vllm.entrypoints` leaves ~70 GiB of GPU
memory pinned and the next server dies on startup. The script kills by PID from
`nvidia-smi --query-compute-apps` and waits for the memory to be released.

## Archived: the SGLang phase

`archive/sglang/` holds an earlier attempt on SGLang with an L-Eval long-context
corpus. It is kept because it contains real measurements and documents why the
project moved to vLLM, not because it is still live. Highlights:

- `BASELINE_RECORD.md` — target-only baseline over 8 L-Eval prompts, including a
  verified byte-exact replay recipe for SGLang.
- `SPECULATIVE_BLOCKERS.md` — why the speculative arms could not proceed there:
  deterministic inference does not hold under EAGLE-3, and the nebius drafter is
  packaged for vLLM so SGLang accepted ~0.6% of its draft tokens.

The move to vLLM fixed both, and also matches the paper's own stack.
