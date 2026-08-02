# Lossy speculative decoding on GPT-OSS-20B

Speculative decoding is normally lossless: a drafted token is kept only with
probability `p(x)/q(x)`, which preserves the target model's distribution exactly.
Relax that rule and you go faster, because more draft tokens survive — but the
output is no longer a sample from the target.

This asks what that costs on hard reasoning problems. The answer, on AIME24:
**accuracy drops from 9/10 to 6/10**, and every lost answer is a run that never
finishes rather than one that finishes wrongly.

## The setting

One knob, everything else fixed. Lenience relaxes the acceptance test by a factor
β ∈ (0,1]:

```
strict   accept iff  p(x) / q(x)       >= u        (β = 1, lossless)
lenient  accept iff  p(x) / (β · q(x)) >= u        (β < 1, lossy)
```

Smaller β lowers the bar, so tokens the target would have rejected get emitted
anyway. We compare β=1.0 against β=0.2 with the same target, drafter, seed,
prompts, and request order — the acceptance test is the only difference.

## The patch

vLLM has no acceptance-threshold knob (`rejection_sample_method` offers only
`standard`, `synthetic`, `block`), so Lenience needs a one-line change to the
verifier. Against upstream vLLM 0.26.0, the substance is:

```diff
--- vllm/v1/sample/rejection_sampler.py (upstream 0.26.0)
+++ vllm/v1/sample/rejection_sampler.py (patched)
@@ -787,6 +817,7 @@
     synthetic_conditional_rates_ptr,
+    lenience_beta,  # scalar; 1.0 for the strict rule
     NO_DRAFT_PROBS: tl.constexpr,
@@ -829,7 +860,11 @@
-                accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob
+                accepted = (
+                    draft_prob > 0
+                    and target_prob / (draft_prob * lenience_beta) >= uniform_prob
+                )
```

plus threading the value through at the kernel launch and reading β at import.
Full diff in [`patches/`](patches/), re-apply with `bash patches/apply.sh`.

At β=1.0 the multiply is the identity, so the control arm is provably the stock
verifier — that is what makes this a clean single-variable contrast.

Two things cost real time and are worth knowing before touching this:

- **vLLM ships two model runners.** `use_v2_model_runner` is unset by default and
  this config picks V1. Patching only the V2 kernel changes nothing at all — the
  lossy arm comes out bit-identical to strict, which looks like a null result.
  Both files are patched.
- **Environment variables never reach the sampler.** EngineCore is spawned with a
  sanitised environment; `LENIENCE_BETA` was verified present in the API server's
  `/proc/<pid>/environ` and absent from EngineCore's. β is passed via the file
  `.lenience_beta`, and the patched module prints the value it loaded to stderr so
  the server log proves what actually ran.

## Environment

| | |
|---|---|
| Target | `openai/gpt-oss-20b` @ `6cee5e81`, MXFP4 |
| Drafter | `nebius/EAGLE3-gpt-oss-20b`, k=6 |
| Serving | vLLM 0.26.0+cu129, torch 2.11.0+cu129 |
| Hardware | 1× H100 PCIe, no TP/PP/DP/FSDP |
| Sampling | temperature 1.0, top-p 1.0, seed 0, `max_new_tokens=32768` |

The paper's stack is vLLM 0.20.1 + torch 2.11.0+**cu130**. cu130 needs driver 580+
and this box has 570, so we build for cu129 — a minor-version step CUDA covers,
verified by running. Details and the rebuild commands are in
[`remote/ENVIRONMENT.md`](remote/ENVIRONMENT.md).

Two settings are not optional. **Prefix caching is disabled**: with it on, a
request that reuses cached prompt KV takes a different numeric path than one that
recomputes it. And **results depend on request position** — case_001 gives 1,711
tokens as the first request on a server and 2,485 as the second. Each arm must be
a fresh server running all cases in the same order, or the comparison is
meaningless.

## Tests and results

10 AIME24 problems, Harmony-rendered at `reasoning_effort=medium`, one seed.
Data in `runs/aime24/`, full write-up in [RESULTS.md](RESULTS.md).

| | strict (β=1.0) | lossy (β=0.2) |
|---|---:|---:|
| correct answers | **9/10** | **6/10** |
| hit the 32,768 cap | 0/10 | 3/10 |
| never emitted a `final` channel | 0/10 | 3/10 |
| mean accepted draft tokens/round (l̄) | 2.250 | 3.415 |
| length inflation (geometric mean) | — | 1.44× |

l̄ rising in all ten cases is the mechanical proof the patch is live: a lower bar
accepts more draft tokens by construction.

The three lost answers are *exactly* the three runs that hit the cap. Lenience
never produced a wrong answer — it produced no answer, by spending the entire
budget inside the `analysis` channel. `case_004` is wrong under both rules, which
is a useful control: the lossy rule does not simply degrade everything.

The length statistic is weaker than it looks (paired mean log ratio 0.362, but
t(9)=1.55, and three ratios are censored at the cap). Don't lean on it.

## Self-correction markers

`scripts/analyze_markers.py` counts hesitation and rework language across both
arms, normalised per 1000 generated tokens — raw counts would just re-measure the
length difference.

Markers per 1000 generated tokens, summed over 10 cases (strict: 85,580 tok, lossy: 154,655 tok).

| marker | strict /1k | lossy /1k | ratio |
|---|---:|---:|---:|
| wait | 2.89 | 3.66 | 1.27x |
| hmm | 0.06 | 0.10 | 1.66x |
| actually | 1.66 | 2.80 | 1.69x |
| oops | 0.00 | 0.03 | - |
| mistake | 0.02 | 0.14 | 5.81x |
| should be | 0.11 | 0.07 | 0.68x |
| recompute | 0.07 | 0.10 | 1.48x |
| recheck | 0.30 | 0.11 | 0.36x |
| redo | 0.00 | 0.00 | - |
| let's compute | 1.44 | 2.14 | 1.49x |

| group | strict /1k | lossy /1k | ratio |
|---|---:|---:|---:|
| **hesitation** | 4.60 | 6.59 | **1.43x** |
| **error-flag** | 0.13 | 0.21 | **1.61x** |
| **rework** | 1.81 | 2.35 | **1.30x** |

Per-case totals across all markers, per 1k tokens:

| case | strict | lossy | ratio |
|---|---:|---:|---:|
| case_001 | 2.9 | 3.2 | 1.08x |
| case_002 | 11.5 | 13.0 | 1.12x |
| case_003 | 6.2 | 7.8 | 1.27x |
| case_004 | 5.0 | 4.1 | 0.82x |
| case_005 | 7.6 | 9.6 | 1.26x |
| case_006 | 5.0 | 9.6 | 1.92x |
| case_007 | 4.9 | 9.1 | 1.88x |
| case_008 | 5.1 | 3.6 | 0.70x |
| case_009 | 6.8 | 6.1 | 0.89x |
| case_010 | 4.8 | 3.7 | 0.78x |

lossy higher in 6/10 cases.

The aggregate direction matches the hypothesis: hesitation up 1.43x, explicit
error-flagging up 1.61x, rework up 1.30x, and "mistake/wrong/miscalc" up 5.8x off
a small base.

But it is only **6/10 per case**, and the two largest movers are case_006 (1.92x)
and case_007 (1.88x). Four cases go the other way. So this is corroborating
evidence for the mechanism where it occurs, not a detector — it does not
cleanly separate the arms, and it would not have flagged case_003.

## Hypothesis

The degeneration is not the model losing coherence. Its reasoning stays valid; the
*digits* get corrupted on emission, and the model's own error-checking then traps
it.

In `case_006` the model asserts 23 distinct values for a single product
(`13,026,069 × 1,102,721`) across 165 assertions, only 6% correct. The errors are
`+1,500`, `−10,000`, `+600,000`, `−13,026,069` — single-digit place values and one
exactly-one-operand slip, which is what forced-in draft tokens should look like.
It catches itself constantly ("wait" ×114, "actually" ×84):

> `Wait compute: 80*272 = 21760; 2*272 = 544; sum=222.`

Both partial products are right; the sum is emitted as `222` instead of `22304`.
So the loop is: emit corrupted digits → detect the inconsistency → recompute →
get corrupted again. The wrong value is still being asserted 98% of the way
through the output. It never converges.

**This explains two of the three failures, not all three.** `case_002` is the same
shape in symbolic form (recomputing `AD^2`/`BD` ~8× more often than strict).
`case_003` is different — exploration density is flat, it simply case-splits
combinatorially for 2.5× longer without concluding. So there are at least two
lossy failure modes, and a numeric-corruption detector would catch only the first.
The signals that caught all three are the coarse ones: cap hit, missing `final`
channel, lost answer.
