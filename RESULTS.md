# Results: relaxed speculative decoding on GPT-OSS-20B

Primary data: `runs/aime24_fresh/`, recorded 2026-08-03. **30 AIME24 problems,
one seed, one freshly started server per measurement** — every request is its
engine's first, on both arms, so the arms differ only in the acceptance rule.

An earlier 10-problem pilot (`runs/aime24/`) ran all cases on one shared server
per arm. Its headline result does not survive this control; see
[What changed](#what-changed-against-the-shared-server-pilot).

## What was run

| | |
|---|---|
| Target | `openai/gpt-oss-20b` @ `6cee5e81` (MXFP4) |
| Drafter | `nebius/EAGLE3-gpt-oss-20b`, k=6 |
| Serving | vLLM 0.26.0+cu129, single H100 PCIe, no TP/PP/DP/FSDP |
| Corpus | all 30 AIME24 problems, Harmony-rendered, `reasoning_effort=medium` |
| Sampling | temperature 1.0, top-p 1.0, seed 0 (single seed, no sweep) |
| Cap | `max_new_tokens=32768` |
| Prefix caching | disabled |
| Request position | 1 on every run, asserted from `/metrics` and recorded in each `config.json` |

Two arms, differing **only** in the acceptance rule inside the verify kernel:

- **strict** — λ=1.0, standard rejection sampling `p/q > u`. Lossless.
- **lossy** — λ=0.2, Lenience `p/(λ·q) > u`. Accepts draft tokens the target would have rejected.

λ is `1 − α` in the mentored-decoding notation of Xia et al., not their β.

## Headline

| metric | strict (λ=1.0) | lossy (λ=0.2) | paired test |
|---|---:|---:|---|
| mean l̄ (accepted draft tokens/round) | 2.205 | **3.458** | higher in **30/30** cases |
| geometric mean length inflation | — | **1.45×** | t(29)=3.04, sign test p=0.016 |
| hit the 32,768 cap | 3/30 | **7/30** | McNemar p=0.219 |
| never reached `final` channel | 3/30 | **7/30** | — |
| correct answers | 23/30 | 21/30 | McNemar p=0.625 |
| wrong answers | 4/30 | 2/30 | — |

**The robust effect is length, not accuracy.** λ=0.2 makes the model generate
~1.45× more tokens for the same problem, and that holds up statistically at
n=30. Accuracy is indistinguishable between the arms on this data. Non-termination
is 7 vs 3 in the expected direction but not significant.

This inverts the pilot, where accuracy looked like the finding and length looked
underpowered. Both readings came from the same underlying instability: per-case
outcomes at this cap are far more sensitive to engine state than a single
shared-server seed suggested.

## Data

| case | ref | strict | lossy | L strict | L lossy | l̄ strict | l̄ lossy |
|---|---:|---|---|---:|---:|---:|---:|
| case_001 | 204 | ✓ | ✓ | 1,711 | 1,587 | 2.88 | 3.79 |
| case_002 | 113 | **no answer** | **no answer** | 32,768 | 32,768 | 1.87 | 3.11 |
| case_003 | 371 | ✓ | **no answer** | 28,934 | 32,768 | 1.68 | 3.00 |
| case_004 | 385 | wrong | **no answer** | 10,406 | 32,768 | 2.06 | 3.45 |
| case_005 | 110 | ✓ | ✓ | 5,394 | 10,356 | 2.17 | 3.37 |
| case_006 | 104 | wrong | ✓ | 10,613 | 8,458 | 2.24 | 3.63 |
| case_007 | 721 | ✓ | ✓ | 6,048 | 7,067 | 1.93 | 3.23 |
| case_008 | 025 | ✓ | ✓ | 1,976 | 7,158 | 2.43 | 3.73 |
| case_009 | 809 | ✓ | ✓ | 2,916 | 2,997 | 2.29 | 3.29 |
| case_010 | 116 | ✓ | ✓ | 1,518 | 1,082 | 2.66 | 3.35 |
| case_011 | 104 | ✓ | ✓ | 4,727 | 9,521 | 2.27 | 3.48 |
| case_012 | 294 | ✓ | ✓ | 2,071 | 3,804 | 2.14 | 3.48 |
| case_013 | 540 | ✓ | ✓ | 1,491 | 2,585 | 2.29 | 3.86 |
| case_014 | 197 | **no answer** | wrong | 32,768 | 19,179 | 1.65 | 3.29 |
| case_015 | 480 | ✓ | ✓ | 4,970 | 6,216 | 2.30 | 3.41 |
| case_016 | 073 | ✓ | ✓ | 4,346 | 9,030 | 2.34 | 3.69 |
| case_017 | 468 | ✓ | ✓ | 21,576 | 29,202 | 2.12 | 3.24 |
| case_018 | 601 | ✓ | ✓ | 5,298 | 18,978 | 2.52 | 3.54 |
| case_019 | 023 | ✓ | **no answer** | 11,554 | 32,768 | 2.22 | 3.34 |
| case_020 | 321 | ✓ | ✓ | 3,342 | 4,392 | 2.42 | 3.74 |
| case_021 | 211 | ✓ | ✓ | 9,852 | 17,091 | 1.95 | 3.40 |
| case_022 | 315 | wrong | **no answer** | 8,465 | 32,768 | 1.83 | 3.20 |
| case_023 | 236 | ✓ | ✓ | 7,786 | 10,242 | 2.05 | 3.64 |
| case_024 | 045 | ✓ | ✓ | 20,202 | 3,495 | 2.47 | 3.57 |
| case_025 | 033 | ✓ | ✓ | 3,043 | 2,375 | 2.69 | 3.64 |
| case_026 | 080 | **no answer** | **no answer** | 32,768 | 32,768 | 1.91 | 3.18 |
| case_027 | 055 | ✓ | ✓ | 2,191 | 2,910 | 2.42 | 3.67 |
| case_028 | 699 | ✓ | ✓ | 3,680 | 4,900 | 2.44 | 3.72 |
| case_029 | 127 | ✓ | wrong | 10,687 | 31,342 | 2.16 | 3.43 |
| case_030 | 902 | wrong | **no answer** | 8,202 | 32,768 | 1.74 | 3.26 |

Regenerate with `scripts/grade_aime.py --runs-root runs/aime24_fresh`.

## The length effect

Paired mean log length ratio **0.371** (sd 0.669, t(29)=3.04), geometric mean
**1.45×**, longer in **22/30** cases (two-sided sign test p=0.016).

The capped runs censor this: 7 lossy and 3 strict runs would have gone further.
Dropping every pair where either arm capped leaves 22 pairs and the effect
survives — mean log ratio 0.301, t=2.20, geometric mean **1.35×**. So it is not
an artefact of the cap.

**l̄ rising 2.205 → 3.458 in 30/30 cases** is the mechanical proof the patch is
live: relaxing the threshold accepts more draft tokens by construction.

## Non-termination

7/30 lossy runs versus 3/30 strict runs exhaust the budget without emitting a
`final` channel. Paired: 5 lossy-only, 1 strict-only, 2 both (McNemar p=0.219).
Directionally consistent with the mechanism, not significant at n=30, one seed.

The asymmetry that does hold up is in *how* the arms fail. Lossy loses 7 answers
to non-termination and only 2 to wrongness; strict loses 3 and 4. Lenience
mostly does not make the model answer incorrectly — it makes it not answer.

`case_030` is the clearest instance. The lossy run derives the correct answer
(902) inside its reasoning and asserts it three times, then keeps going for the
remaining ~19k tokens and never opens a `final` channel:

```
...for all-white or all-black. That yields 902.

But we also must check that if we have all rows W, all columns B? This would be
a case where W subset of rows are all 5. But intersection of row W with column W
must have some cells: ...
```

The strict run on the same problem terminates cleanly and answers `\boxed{252}` —
wrong. So on this problem lossy reasoned better and scored worse, purely on
failure to stop.

## What changed against the shared-server pilot

The pilot (`runs/aime24/`, 10 problems, one shared server per arm) reported
strict 9/10 → lossy 6/10. The same ten problems, re-run one-server-per-case:

| | pilot (shared server) | fresh (ordinal 1) |
|---|---|---|
| strict correct | 9/10 | 7/10 |
| lossy correct | 6/10 | 7/10 |

`case_001` reproduces the pilot **exactly** — 1,711 / 1,587 tokens, l̄ 2.880 /
3.795, both arms — because it was ordinal 1 in the pilot too. Every case after
it moves, sometimes drastically: pilot strict `case_002` answered at 13,870
tokens, fresh strict caps at 32,768 with no answer; pilot lossy `case_004`
finished (wrong) at 10,492, fresh lossy caps.

Two conclusions:

1. **The pipeline is deterministic and the two datasets are comparable.** The one
   case where request position matched is bit-stable across the two runs. It also
   confirms `--mode strict` equals the pilot's "lossy mode at λ=1.0" arm.
2. **Request position was a real confound.** The pilot's 3/3 lossy-only failures
   were partly an artefact of it; under proper control the accuracy gap
   disappears and the surviving effect is length.

The pilot's numbers are kept in git history rather than restated here, since
they measure a confounded contrast.

## Self-correction markers

`scripts/analyze_markers.py --runs-root runs/aime24_fresh --arms strict lenience0p2 --labels strict lossy`,
normalised per 1000 generated tokens (strict 301,303 tok, lossy 443,343 tok).

| group | strict /1k | lossy /1k | ratio |
|---|---:|---:|---:|
| hesitation | 4.66 | 6.02 | **1.29×** |
| error-flag | 0.30 | 0.13 | 0.44× |
| rework | 1.47 | 1.65 | 1.12× |

Hesitation language is up 1.29× and higher in 24/30 cases, which is consistent
with the mechanism. Explicit error-flagging goes the *other* way (0.44×), and
the "mistake/wrong/miscalc" marker that read 5.8× on the pilot reads 0.38× here
— it was a small-base artefact of 10 cases. Treat the marker analysis as weak
corroboration of hesitation only; it is not a detector, and it did not separate
the arms.

## How to read this honestly

- **Length inflation is established** on this data: 1.45× overall, 1.35× on the
  uncensored subset, significant by both t and sign tests, with l̄ up in 30/30.
- **Accuracy loss is not.** 23/30 vs 21/30, McNemar p=0.625. The pilot's 9/10 →
  6/10 was a confounded measurement and should not be quoted.
- **Non-termination is suggestive, not shown.** 7 vs 3, p=0.219.
- **One seed.** Per-case outcomes proved unstable to engine state, which is
  direct evidence that single-seed per-case results here are anecdotes. Rates
  over 5–10 seeds are the next step, not a wider benchmark.
- **λ=1.0 is semantically stock, not empirically bit-identical** to unpatched
  vLLM. `patches/test_lenience.py` pins the acceptance boundary at kernel level;
  the end-to-end unpatched-vs-patched control has not been run.

## Reproduce

```bash
bash patches/apply.sh
.venv-vllm/bin/python scripts/fresh_server_replay.py \
  --arms strict lossy --lenience-factor 0.2 \
  --cases $(printf 'case_%03d ' $(seq 1 30)) \
  --seeds 0 --temperature 1.0 --max-new-tokens 32768 \
  --prompt-root prompts/aime24 --runs-root runs/aime24_fresh
.venv-vllm/bin/python scripts/grade_aime.py --runs-root runs/aime24_fresh
```

60 runs, ~90s each including a full server start, about 1.5h on one H100 PCIe.
Every run asserts a fresh engine and refuses to write a directory whose recorded
λ disagrees with what the server loaded.

## Next

1. **Seeds.** 5–10 seeds on all 30 cases, reported as per-case failure and
   inflation *rates*. `scripts/grade_aime.py` prints the rate table once more
   than one seed is present. At ~90s per run, 30 cases × 2 arms × 5 seeds ≈ 7.5h.
2. **The patch control.** Unpatched stock vLLM against patched λ=1.0, same case,
   each as a fresh server's first request, compared token-for-token.
3. **λ sweep.** One point (0.2) says nothing about where the effect turns on.
