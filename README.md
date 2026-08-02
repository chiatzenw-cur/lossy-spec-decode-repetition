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
