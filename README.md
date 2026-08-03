# Reasoning loops under relaxed speculative decoding (GPT-OSS-20B)

Speculative decoding is normally lossless: a drafted token is kept only with
probability `p(x)/q(x)`, which preserves the target model's distribution exactly.
Relax that rule and you go faster, because more draft tokens survive — but the
output is no longer a sample from the target.

This asks what that costs on hard reasoning problems. On all 30 AIME24 problems
at one seed, with a freshly started server for every measurement, the answer is:
**the model thinks ~1.45× longer for the same problem**, and it fails by not
terminating rather than by answering wrongly. Accuracy itself is
indistinguishable between the arms at this sample size (23/30 vs 21/30).

An earlier 10-problem pilot on a shared server reported 9/10 → 6/10. That
contrast was confounded by request position and does not survive the control —
see [RESULTS.md](RESULTS.md#what-changed-against-the-shared-server-pilot).

## The setting

One knob, everything else fixed. Lenience relaxes the acceptance test by a factor
λ ∈ (0,1]:

```
strict   accept iff  p(x) / q(x)       >= u        (λ = 1, lossless)
lenient  accept iff  p(x) / (λ · q(x)) >= u        (λ < 1, lossy)
```

Smaller λ lowers the bar, so tokens the target would have rejected get emitted
anyway. We compare λ=1.0 against λ=0.2 with the same target, drafter, seed and
prompts — the acceptance test is the only difference.

Rejection recovery is untouched: rejected positions still resample from the stock
residual `norm(max(0, p − q))`, and the bonus token still comes from `p`. In the
taxonomy of Xia et al. this is **mentored decoding with λ = 1 − α**. Their β is a
different parameter, fixed at 1 there — an earlier version of this repo called
the knob β, which misidentified the method.

## The patch

vLLM has no acceptance-threshold knob (`rejection_sample_method` offers only
`standard`, `synthetic`, `block`), so Lenience needs a one-line change to the
verifier. The whole semantic change, against upstream vLLM 0.26.0:

```diff
-                accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob
+                accepted = (
+                    draft_prob > 0
+                    and target_prob / (draft_prob * lenience_factor) >= uniform_prob
+                )
```

Everything else is plumbing: threading λ to the kernel launch, reading it at
import, one log line. Full unified diff in
[`patches/vllm-0.26.0-lenience.patch`](patches/vllm-0.26.0-lenience.patch);
apply with `bash patches/apply.sh`, which verifies the upstream and patched
sha256 of both files and then runs an acceptance test.

At λ=1.0 the multiply is the identity, so the control arm is *semantically* the
stock verifier. The kernel is still recompiled with an extra argument, so
"bit-identical" is an empirical claim, not a proof — see the open items below.

Two things cost real time and are worth knowing before touching this:

- **vLLM ships two model runners.** `use_v2_model_runner` is unset by default and
  this config picks V1. Patching only the V2 kernel changes nothing at all — the
  lossy arm comes out identical to strict, which looks like a null result.
  Both files are patched, and `patches/test_lenience.py` drives the V1 kernel
  directly (no model, no server) to prove λ reaches it.
- **Environment variables never reach the sampler.** EngineCore is spawned with a
  sanitised environment; the env var was verified present in the API server's
  `/proc/<pid>/environ` and absent from EngineCore's. λ is passed via the file
  `/tmp/lossy-spec-decode-lenience-$UID` — uid-scoped, not repo-relative, so a
  clone in any directory agrees with the patch — and the patched module prints
  the value it loaded to stderr, which the runner scrapes into every run record.

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
tokens as the first request on a server and 2,485 as the second.

Running both arms as fresh servers issuing the same cases in the same order is
*not* sufficient to control that. By the second case the two arms have already
emitted different numbers of tokens and consumed different numbers of RNG draws,
so equal ordinal position does not mean equal engine state. The fix is one
server per measurement:

```bash
.venv-vllm/bin/python scripts/fresh_server_replay.py \
  --arms strict lossy --lenience-factor 0.2 \
  --cases $(printf 'case_%03d ' $(seq 1 30)) --seeds 0 \
  --prompt-root prompts/aime24 --runs-root runs/aime24_fresh
```

Every request is then the first request its engine ever sees, on both arms. The
runner asserts this from `/metrics` (`--assert-fresh-server`) and records the
ordinal in each `config.json`, so a warm request cannot be archived as a cold
one. Everything reported below has this control; it cost about 1.5h for 60 runs.

It mattered. On the ten problems the pilot shared a server for, only `case_001`
— ordinal 1 in both — reproduced exactly. Every later case moved, and two of the
pilot's three lossy-only failures stopped being lossy-only.

## Prompts

`prompts/aime24/` holds all **30** AIME 2024 problems (`HuggingFaceH4/aime_2024`
train, ids 60–89), Harmony-rendered at `reasoning_effort=medium`, 125–455 input
tokens. Rebuild them with:

```bash
.venv-vllm/bin/python scripts/build_aime24_prompts.py \
  --rows-json prompts/aime24/source_rows.json --output prompts/aime24 --replace-output
```

`source_rows.json` is the archived dataset response, so the build is offline and
deterministic; re-rendering reproduces every existing `rendered_prompt.txt` byte
for byte.

## Tests and results

All 30 AIME24 problems, one seed, one fresh server per measurement. Data in
`runs/aime24_fresh/`, full write-up in [RESULTS.md](RESULTS.md). Grade it
yourself with `scripts/grade_aime.py --runs-root runs/aime24_fresh`.

| | strict (λ=1.0) | lossy (λ=0.2) | paired test |
|---|---:|---:|---|
| mean accepted draft tokens/round (l̄) | 2.205 | **3.458** | higher in **30/30** |
| length inflation (geometric mean) | — | **1.45×** | t(29)=3.04, sign p=0.016 |
| hit the 32,768 cap | 3/30 | **7/30** | McNemar p=0.219 |
| never emitted a `final` channel | 3/30 | **7/30** | — |
| correct answers | 23/30 | 21/30 | McNemar p=0.625 |
| wrong answers | 4/30 | 2/30 | — |

**Length is the effect that holds up.** It survives dropping every pair where
either arm hit the cap (1.35×, t=2.20), so it is not a censoring artefact. l̄
rising in all 30 cases is the mechanical proof the patch is live: a lower bar
accepts more draft tokens by construction.

**Accuracy is not.** 23/30 vs 21/30 is nothing at this n. What is asymmetric is
the *failure mode*: lossy loses 7 answers to non-termination and 2 to wrongness,
strict loses 3 and 4. Lenience mostly doesn't make the model answer incorrectly,
it makes it not answer — on `case_030` the lossy run derives the correct answer
inside its reasoning, asserts it three times, and then never opens a `final`
channel, while strict terminates cleanly on a wrong answer.

## Self-correction markers

`scripts/analyze_markers.py` counts hesitation and rework language across both
arms, normalised per 1000 generated tokens — raw counts would just re-measure the
length difference.

```bash
.venv-vllm/bin/python scripts/analyze_markers.py --runs-root runs/aime24_fresh \
  --arms strict lenience0p2 --labels strict lossy
```

Over 30 cases (strict 301,303 tok, lossy 443,343 tok):

| group | strict /1k | lossy /1k | ratio |
|---|---:|---:|---:|
| **hesitation** | 4.66 | 6.02 | **1.29×** |
| **error-flag** | 0.30 | 0.13 | 0.44× |
| **rework** | 1.47 | 1.65 | 1.12× |

Hesitation is up 1.29× and higher in 24/30 cases. But error-flagging goes the
*other* way, and the "mistake/wrong/miscalc" marker that read 5.8× on the
10-case pilot reads 0.38× here — it was a small-base artefact. This is weak
corroboration for hesitation, not a detector: it does not separate the arms.

## Hypothesis

The degeneration is not the model losing coherence. Its reasoning stays valid; the
*digits* get corrupted on emission, and the model's own error-checking then traps
it.

In the pilot's `case_006` run the model asserts 23 distinct values for a single
product (`13,026,069 × 1,102,721`) across 165 assertions, only 6% correct. The errors are
`+1,500`, `−10,000`, `+600,000`, `−13,026,069` — single-digit place values and one
exactly-one-operand slip, which is what forced-in draft tokens should look like.
It catches itself constantly ("wait" ×114, "actually" ×84):

> `Wait compute: 80*272 = 21760; 2*272 = 544; sum=222.`

Both partial products are right; the sum is emitted as `222` instead of `22304`.
So the loop is: emit corrupted digits → detect the inconsistency → recompute →
get corrupted again. The wrong value is still being asserted 98% of the way
through the output. It never converges.

That case study comes from the shared-server pilot, so read it as a mechanism
sketch rather than evidence. The corroborating observation in the controlled
data is `case_030`, where the lossy run reaches the correct answer in its
reasoning and still never stops.

**One mechanism does not cover every failure.** In the pilot, `case_003` showed
flat exploration density — it simply case-split combinatorially for 2.5× longer
without concluding, no digit corruption involved. So there are at least two lossy
failure modes, and a numeric-corruption detector would catch only the first. The
signals that catch all of them are the coarse ones: cap hit, missing `final`
channel, lost answer.

## Reproduce

```bash
bash patches/apply.sh          # verifies hashes, applies, runs the kernel test

# control arm: lossless verifier, one server per measurement
# lossy arm: same command with --arms lossy
.venv-vllm/bin/python scripts/fresh_server_replay.py \
  --arms strict lossy --lenience-factor 0.2 \
  --cases case_001 case_002 case_003 case_004 case_005 \
          case_006 case_007 case_008 case_009 case_010 \
  --seeds 0 --temperature 1.0 --max-new-tokens 32768 \
  --prompt-root prompts/aime24 --runs-root runs/aime24_fresh

.venv-vllm/bin/python scripts/grade_aime.py --runs-root runs/aime24_fresh
```

The shared-server form the archived `runs/aime24/` used is in
[RESULTS.md](RESULTS.md#reproduce). Either way the lossy arm is invoked as
`--mode lossy --lossy-method lenience --lenience-factor 0.2`, and the runner
refuses to write a run directory unless the factor the server actually loaded
matches.

## What this does not establish

- **An accuracy effect.** 23/30 vs 21/30, McNemar p=0.625. There isn't one in
  this data. The pilot's apparent 9/10 → 6/10 was confounded by request order.
- **A non-termination effect.** 7/30 vs 3/30 is the right direction but
  p=0.219. Suggestive only.
- **Stability of any single case.** Per-case outcomes moved substantially when
  request position changed, so one seed per case is an anecdote. The fix is 5–10
  seeds reported as *rates* (`scripts/grade_aime.py` prints that table).
- **That λ=1.0 is bit-identical to unpatched vLLM.** It is semantically the same
  expression, and the kernel unit test confirms the acceptance boundary, but the
  end-to-end control (unpatched stock vs patched λ=1.0, token-for-token) has not
  been run.
(The length effect *is* established here — 1.45×, t(29)=3.04, 1.35× and t=2.20
after dropping every censored pair — which is the reverse of what the 10-case
pilot suggested.)
