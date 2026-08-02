# Baseline record: non-speculative GPT-OSS-20B

First data ever gathered in this repository. Recorded 2026-08-02.

This is the target-only control arm — no draft model, no speculative decoding.
It is the reference every later strict and lossy run gets compared against.

## Configuration

| | |
|---|---|
| Mode | `baseline` (target-only; `speculative_algorithm=None` confirmed in `server_info.json`) |
| Target | `openai/gpt-oss-20b` @ `6cee5e81ee83917806bbde320786a8fb61efebee` |
| GPU | NVIDIA H100 PCIe, 81,559 MiB |
| SGLang | 0.5.10.post1 (see `remote/ENVIRONMENT.md` for why not newer) |
| Prompts | all 8 archived L-Eval cases in `prompts/leval_9k_11k/` |
| Sampling | temperature 0.7, top-p 1.0, top-k -1, repetition_penalty 1.0 |
| `max_new_tokens` | 32768 — deliberately non-binding, not a budget |
| Endpoint | native `/generate`, pre-rendered Harmony prompt |

`max_new_tokens` is set to roughly 60x the longest reference answer purely
because the API requires some value. No prompt came close to it: the longest
generation used 28,479 of 32,768. Every generation ended on its own.

Artifacts per run in `runs/<case>/seed_0/baseline/`: `config.json`, `prompt.txt`,
`request.json`, `response.json`, `output.txt`, `run.json`, `server_info.json`.
Server logs and preflight output are under `logs/`.

## Results

All 8 prompts, seed 0. **8/8 reached natural EOS. 0 hit the token ceiling.
0 triggered the repeated-n-gram signal.**

| case | input tok | output tok | wall s | tok/s | finish | EOS | final-answer words | reference words |
|---|---:|---:|---:|---:|---|---|---:|---:|
| case_001 | 9,046 | 1,546 | 6.93 | 223.1 | stop | yes | 527 | 407 |
| case_002 | 9,302 | 9,329 | 41.57 | 224.4 | stop | yes | 339 | 339 |
| case_003 | 10,384 | 1,185 | 5.47 | 216.6 | stop | yes | 464 | 335 |
| case_004 | 10,622 | 6,574 | 29.10 | 225.9 | stop | yes | 306 | 308 |
| case_005 | 9,334 | 28,479 | 130.03 | 219.0 | stop | yes | 301 | 295 |
| case_006 | 9,299 | 3,932 | 17.32 | 227.1 | stop | yes | 215 | 243 |
| case_007 | 9,034 | 2,204 | 9.70 | 227.2 | stop | yes | 172 | 172 |
| case_008 | 10,372 | 4,309 | 19.05 | 226.2 | stop | yes | 98 | 99 |

Totals: 57,558 output tokens in 259.2 s, mean 222.1 tok/s. Throughput is flat
across every prompt (216.6–227.2), so nothing anomalous in the serving path.

Machine-readable: `runs/baseline_summary.{json,csv,md}`.

## Observations

**Output length varies by more than an order of magnitude** (1,185 to 28,479
tokens) across prompts that are all 9k–10.6k input tokens and all ask for a few
hundred words. Any later "lossy output is more than twice the strict output
length" signal has to be read against this spread, not against a single number.

**Nearly all of that variance is reasoning, not answer.** Every run has clean
`analysis` → `final` channel structure, and the final answers track the
reference lengths closely (case_002 339 vs 339, case_007 172 vs 172, case_008
98 vs 99, case_004 306 vs 308). The analysis channel is 43%–98% of output
characters. case_005 spent 106,170 analysis characters to produce a 301-word
answer — it repeatedly counts and recounts words to hit the requested length.
That behavior is native to the target model at `reasoning_effort=high`; it is
not a decoding defect, and it is present with no draft model in the loop.

**A 4,096-token cap manufactures false positives.** An earlier round at
`max_new_tokens=4096` (preserved under the `baseline_cap4096` tag) showed
case_004 and case_005 terminating on `length` with no EOS and no `final`
channel — exactly the automatic signature DESIGN §10.1 lists for degeneration.
Both are artifacts of the cap. Given room, case_004 finishes at 6,574 tokens
and case_005 at 28,479, both with clean EOS. Any lossy run must be capped
generously or `finish_reason == "length"` means nothing.

## Reproducibility caveat — seeds were not in effect

`sampling_seed: 0` was sent in every request and had no effect. Two runs of the
same prompt with identical parameters diverge within the first few tokens:

| case | run A tokens | run B tokens | first divergent token |
|---|---:|---:|---:|
| case_001 | 2,317 | 1,546 | 12 |
| case_002 | 3,782 | 9,329 | 3 |
| case_003 | 1,117 | 1,185 | 9 |
| case_004 | 4,096 | 6,574 | 3 |
| case_005 | 4,096 | 28,479 | 7 |

Cause, confirmed in the installed source: `sampling_batch_info.py:94-109` builds
the per-request seed tensor only `if enable_deterministic` and otherwise sets it
to `None`, where `enable_deterministic` is
`server_args.enable_deterministic_inference` (`:76`). The servers ran with that
flag off, so sampling fell back to the server's global RNG — and `random_seed`
is randomized per launch (observed: 194295244, 387945280, 437086094, 833866639).

The request-level seed is accepted and silently ignored. Nothing in the client
or the response reveals this.

**This blocks DESIGN §2.1**, which requires a reproduction to "replay at least
once with the same configuration and seed". It also weakens paired comparison
generally: a single degenerate lossy output cannot be told apart from sampling
noise when the baseline for the same prompt varies from 1,546 to 28,479 tokens
run to run.

### Verified fix

`--enable-deterministic-inference --random-seed 0` alone is **not** sufficient.
With those set and `sampling_seed` honored, two identical requests still diverged
(case_003: 5,818 vs 991 tokens, first differing token at index 133).

The prefix cache is the remaining cause. With it on, a repeat request reuses
prompt KV computed by an earlier request's chunked prefill rather than
recomputing it, and the trajectory drifts a few hundred tokens in. SGLang's own
startup logic treats `fa3` as radix-compatible under deterministic inference
(`RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND`), but that does not hold for
gpt-oss-20b on this build.

The working combination is all three flags together:

```text
--enable-deterministic-inference --random-seed 0 --disable-radix-cache
```

Measured on case_003 after the change:

| test | result |
|---|---|
| seed 0, run twice | identical token ids, 5,818 tokens both |
| seed 1, run twice | identical token ids, 1,109 tokens both |
| seed 0 vs seed 1 | diverge at token 3 |

So replay is exact and the seed genuinely selects the trajectory. `DETERMINISTIC=1`
in `remote/run_server.sh` applies all three; radix is not left as a separate
switch, because two of the three flags silently gives non-replayable runs.

Costs, which apply equally to every arm: `sampling_backend` is forced to
`pytorch`, `disable_piecewise_cuda_graph` is set, the attention backend is
constrained, and prefix caching is off. Startup is noticeably slower (CUDA graph
capture ~8.4 s/batch). Because these change the numerics, a deterministic
baseline is **not** comparable to a non-deterministic strict or lossy run — the
whole matrix has to be gathered under one regime.

The data above remains valid as a record of target-model behavior; those are
real samples from the true distribution. It is just not a seed-replayable record.

## Deterministic round (`baseline_det`) — the replayable record

Re-gathered under the verified deterministic regime. Same 8 prompts, seed 0,
same non-binding 32,768 ceiling. **8/8 natural EOS, 0 at the ceiling, 0 repeat
signals.** This is the arm strict and lossy must be compared against.

| case | input tok | output tok | wall s | tok/s | finish | final words | ref words | stochastic tok |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| case_001 | 9,046 | 13,161 | 107.25 | 122.7 | stop | 414 | 407 | 1,546 |
| case_002 | 9,302 | 11,567 | 87.28 | 132.5 | stop | 342 | 339 | 9,329 |
| case_003 | 10,384 | 5,818 | 41.48 | 140.3 | stop | 339 | 335 | 1,185 |
| case_004 | 10,622 | 3,995 | 34.14 | 117.0 | stop | 307 | 308 | 6,574 |
| case_005 | 9,334 | 7,217 | 51.33 | 140.6 | stop | 314 | 295 | 28,479 |
| case_006 | 9,299 | 10,187 | 73.50 | 138.6 | stop | 243 | 243 | 3,932 |
| case_007 | 9,034 | 542 | 3.91 | 138.8 | stop | 160 | 172 | 2,204 |
| case_008 | 10,372 | 2,003 | 14.35 | 139.5 | stop | 102 | 99 | 2,003 |

Totals: 54,490 output tokens in 413.2 s, mean 131.9 tok/s — about 40% of the
stochastic round's 222 tok/s, the price of `sampling_backend=pytorch`, no
piecewise CUDA graphs, and no prefix cache.

Machine-readable: `runs/baseline_det_summary.{json,csv,md}`.

Every case again shows clean `analysis` → `final` structure with final answers
tracking the reference lengths (case_006 243 vs 243, case_004 307 vs 308,
case_002 342 vs 339). Per-prompt output length still varies widely (542 to
13,161) — that spread is a property of the prompts, not of sampling noise, and
it is now fixed rather than random.

The right-hand column is the same prompt and seed under the stochastic regime.
The two disagree substantially (case_005: 7,217 vs 28,479) because the regimes
have different numerics; neither is more correct. Do not mix arms across regimes.

Replay was confirmed a third time here: `case_003` produced exactly 5,818 tokens,
matching both diagnostic runs — this time from a freshly launched server, so the
recipe holds across restarts, not just within one process.

## Reproduce

Deterministic (use this for anything compared against strict/lossy):

```bash
PYTHON=$PWD/.venv/bin/python DETERMINISTIC=1 RANDOM_SEED=0 \
  bash remote/run_pipeline.sh baseline \
  --cases case_001 case_002 case_003 case_004 case_005 case_006 case_007 case_008 \
  --max-new-tokens 32768 --timeout 3600 --tag baseline_det
```

Stochastic (the run-to-run spread record):

```bash
PYTHON=$PWD/.venv/bin/python bash remote/run_pipeline.sh baseline \
  --cases case_001 case_002 case_003 case_004 case_005 case_006 case_007 case_008 \
  --max-new-tokens 32768 --timeout 3600
```
