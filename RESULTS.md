# Results: lossy verification degrades GPT-OSS-20B on AIME24

Recorded 2026-08-02. All numbers below come from files in `runs/aime24/`.

## What was run

| | |
|---|---|
| Target | `openai/gpt-oss-20b` @ `6cee5e81` (MXFP4) |
| Drafter | `nebius/EAGLE3-gpt-oss-20b`, k=6 |
| Serving | vLLM 0.26.0+cu129, single H100 PCIe, no TP/PP/DP/FSDP |
| Corpus | 10 AIME24 problems, Harmony-rendered, `reasoning_effort=medium` |
| Sampling | temperature 1.0, top-p 1.0, seed 0 (single fixed seed, no sweep) |
| Cap | `max_new_tokens=32768` |
| Prefix caching | disabled (required for replay) |

Two arms, differing **only** in the acceptance rule inside the verify kernel:

- **strict** (`b1_equiv10`) — β=1.0, i.e. standard rejection sampling `p/q > u`. Lossless.
- **lossy** (`b02_lenience`) — β=0.2, Lenience `p/(β·q) > u`. Accepts draft tokens the target would have rejected.

β=1.0 makes the patch a mathematical no-op, so the strict arm is the true control.

## Data

| case | answer | in tok | L strict | L lossy | ratio | l̄ strict | l̄ lossy | final strict | final lossy |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|
| case_001 | 204 | 212 | 1,711 | 1,587 | 0.93 | 2.880 | 3.795 | yes | yes |
| case_002 | 113 | 188 | 13,870 | 32,768* | 2.36 | 2.013 | 3.147 | yes | **no** |
| case_003 | 371 | 158 | 13,110 | 32,768* | 2.50 | 1.801 | 3.092 | yes | **no** |
| case_004 | 385 | 163 | 15,237 | 10,492 | 0.69 | 2.138 | 3.517 | yes | yes |
| case_005 | 110 | 145 | 9,707 | 15,821 | 1.63 | 2.232 | 3.013 | yes | yes |
| case_006 | 104 | 225 | 8,623 | 32,768* | 3.80 | 2.480 | 3.176 | yes | **no** |
| case_007 | 721 | 177 | 11,504 | 18,694 | 1.62 | 1.933 | 3.118 | yes | yes |
| case_008 | 025 | 131 | 7,640 | 2,241 | 0.29 | 2.432 | 3.851 | yes | yes |
| case_009 | 809 | 178 | 2,926 | 5,111 | 1.75 | 2.219 | 3.592 | yes | yes |
| case_010 | 116 | 205 | 1,252 | 2,405 | 1.92 | 2.368 | 3.853 | yes | yes |

`*` = hit the 32,768 cap, so that length is a **lower bound** and its ratio is censored.

## Headline numbers

| metric | strict (β=1.0) | lossy (β=0.2) |
|---|---:|---:|
| mean l̄ (accepted draft tokens/round) | 2.250 | **3.415** |
| hit token cap | 0/10 | **3/10** |
| never reached `final` channel | 0/10 | **3/10** |
| geometric mean length inflation | — | **1.44×** |
| paired mean log length ratio | — | **0.362** (sd 0.739, t(9)=1.55) |

## The actual failure

`case_006`, the same problem and seed under both rules.

**strict** — terminates cleanly, emits `analysis` then `final`, and answers correctly (reference answer is 104):

```
\[
m+n+p=20+21+63=104.
\]

\[
\boxed{104}
\]
```

**lossy β=0.2** — runs to the 32,768 cap, never leaves the `analysis` channel, no answer:

```
...So product should be 14,364,119,823,749. This seems plausible. Our earlier
earlier difference of 10,000 due to maybe miscalc earlier?

Let's compute using closed form 13,026,069*1,102,721 = 13,026,069* (1,102,721).
Another method: Use 13,026,069 = 13,026,000 + 69. Then 13,026,000*1,102,721.
We'll compute 13,026,000*1,102,721 = 13,026*1,102,721*1000? Actually 13,026,000 = 13,026*1000
```

Channel sets: strict `['analysis','final']`, lossy `{'analysis'}` only. It is stuck
re-deriving the same product, degrading into an arithmetic loop.

## How to read this honestly

**The categorical result is clean.** Under the lossless rule every prompt terminates
and answers. Under β=0.2, three of ten burn the entire budget without producing an
answer. That shift is 0/10 → 3/10 on both "hit cap" and "no final channel", and it
appears only in the lossy arm.

**The length statistic is underpowered.** Mean paired log ratio 0.362 clears the
pre-registered 0.15 threshold, but t(9)=1.55 is not significant at n=10 with one
seed, and three cases got *shorter* (case_008 at 0.29×). Do not report the length
effect as established on this data.

**Three ratios are censored.** The capped runs would have gone further, so the true
mean log ratio is larger than 0.362. Raising the cap would sharpen the estimate.

**l̄ rising 2.250 → 3.415 in all ten cases** is the mechanical proof the patch is
active — relaxing the threshold accepts more draft tokens by construction.

## Reproduce

```bash
# strict control
bash remote/stop_server.sh
PYTHON=$PWD/.venv-vllm/bin/python DRAFT_MODEL_PATH=nebius/EAGLE3-gpt-oss-20b \
  NUM_SPEC=6 LOSSY_RULE=lenience BETA=1.0 bash remote/run_server_vllm.sh lossy &
.venv-vllm/bin/python scripts/run_experiment_vllm.py --mode strict \
  --prompt-root prompts/aime24 --runs-root runs/aime24 \
  --cases case_001 case_002 case_003 case_004 case_005 case_006 case_007 case_008 case_009 case_010 \
  --seeds 0 --temperature 1.0 --max-new-tokens 32768 --tag b1_equiv10

# lossy arm: identical, but BETA=0.2 and --tag b02_lenience
```

Both arms must run on a **freshly started server, all 10 cases, same order** — see
the run-order caveat in `README.md`.
