# Repetition loops under greedy lossy speculative decoding — evolutionary findings

Working notes on a deep-dive into *why* relaxed acceptance rules (lenience,
CACTUS, spec-casc-opt) produce degenerate adjacent-repeat loops under greedy
sampling, and what actually breaks them. This is a **work-in-progress log**,
not a cleaned-up final report: it keeps wrong turns and superseded hypotheses
in place, with what corrected them, because the corrections are as load-
bearing as the conclusions. Where a claim below was later revised, it says so
instead of being silently removed.

Distinct from `RESULTS.md` (stochastic, temperature=1.0, length/accuracy
study) — this document is entirely about **greedy** drafting/target sampling,
where degeneration shows up as literal token-for-token loops rather than
length inflation.

## Setup

| | |
|---|---|
| Target | `openai/gpt-oss-20b`, vLLM 0.26.0+cu129 |
| Drafters | `nebius/EAGLE3-gpt-oss-20b` (k=6); `amazon/GPT-OSS-20B-P-EAGLE` (MTP-style, parallel_drafting, k=7) |
| Sampling | `draft_sample_method=greedy` on both drafter and target (temperature 0 throughout) |
| Benchmarks | AIME24 (30 cases, 12k budget), HumanEval (164 cases, 9k budget), LongBench-v2 (case_019 / case_079, 18k–40k budget) |
| Relaxation rules | lenience `p/(λ·q) > u` (λ=0.05 radical, 0.002, 0.2), CACTUS (α=2.0), spec-casc-opt (α=0.05) |

**The 8 "truly relaxed greedy" groups** this whole investigation is scoped to
(every loop/baseline/escape number below is pooled across exactly these):
AIME24 lenience-0.05-greedy, HumanEval lenience-0.05-greedy, and 6 LongBench-v2
combinations (EAGLE3/P-EAGLE × lenience-0.002/lenience-0.2/CACTUS-2.0) on
case_019 and case_079. Full list in `scripts/consolidate_loop_events.py`'s
`GROUPS` and `scripts/collect_loop_baseline_metrics.py`'s `GROUPS`.

## Wrong turn #0: spec-casc-opt "relaxation" under greedy is a no-op

**Hypothesis tried:** all three relaxation rules should behave differently
under greedy drafting, same as under stochastic sampling.

**What actually happens:** greedy drafting sets `draft_probs=None`, which
propagates all the way into the rejection-sampler kernel's `NO_DRAFT_PROBS`
branch (`q` hardcoded to 1, draft treated as one-hot). spec-casc-opt's kernel
patch explicitly leaves that branch untouched (its own comment: *"No q to
defer to: unchanged from upstream in this branch"*) — so under greedy,
spec-casc-opt is **mathematically and empirically identical to strict
decoding**. Verified byte-identical output on case_001, α=0.5. Lenience and
CACTUS patch the *shared* accept-line used by both branches, so they remain
genuinely different from strict under greedy (lenience substitutes `q=1` into
its own formula; CACTUS never reads `q` at all). This is why every loop/escape
number in this document comes only from lenience- and CACTUS-tagged runs —
spec-casc-opt-greedy runs are excluded as a confirmed non-treatment.

## Wrong turn #1: "anywhere in the trace" repetition detection

**First approach:** scan the whole output for any repeated n-gram, anywhere,
regardless of distance. **Documented failure rate: 75% false positives** —
legitimate re-derivation, boilerplate, and list structures dominate at that
sensitivity.

**Fix:** require **true adjacency** — `gap_tokens_since_previous == 0` in
`extract_repetition_clusters.py`'s shingle-seed clustering, i.e. the repeat
starts the token immediately after the previous occurrence ends, no filler in
between. `scripts/consolidate_loop_events.py` chains consecutive gap=0
occurrences from the origin, keeps chains with `chain_repeat_count >= 3` and
`total_span_tokens >= 15`, interval-merges overlaps per case. Result: 698
algorithmically-extracted candidates, fed to an LLM judge
(`scripts/judge_adjacent_loops.py`, baseline/non-speculative server, asked
only for verdict+category+reasoning — **never asked for positions**, since
positions come from the algorithmic step and are carried through unchanged).
Judge precision: 315/321 (AIME24), 211/213 (HumanEval), 163/164 (LongBench-v2)
— **98–99% abnormal-verdict rate**, essentially eliminating the false-positive
problem. **689 confirmed-abnormal loops** is the dataset size referenced
throughout the rest of this document.

## Wrong turn #2: `target_top1_margin` — a name that implied a false mechanism

**First formulation:** a field called `target_top1_margin`, intended to read
as "how much the top token beats the runner-up" (`p(1) − p(2)`).

**The bug:** the field was actually computed as `target_top1_prob − p(x)`,
where `p(x)` is the *draft proposal's own* target-probability — a completely
different quantity that can coexist with a low top-1/top-2 gap. Caught by a
direct proof: if the field really were `p(1) − p(2)`, then `p(x) ≤ (1 +
margin)/2` must hold for any observed `x`, and the recorded numbers violated
that bound. **Fix:** renamed to `target_top1_shortfall`, documentation
corrected to state what it actually measures, with a legacy-name fallback
(`get_metric()` / `_LEGACY_METRIC_NAMES` in the collector scripts) so older
`proposals.jsonl` files stay readable. No numeric change — only the name and
docs were wrong, not the math once correctly labeled.

## Wrong turn #3 (the big one): draft-proposal / recovery-token conflation

**What was reported first:** "recovered tokens are deep in the target's tail"
(mean rank ≈ 1627), used to argue recovery events pull generation toward an
atypical, low-confidence token.

**The bug:** for `recovered`/`other` rows, every metric field (`p`, `q`,
`target_rank`, `target_top1_shortfall`) was computed via
`target_probs.gather(1, draft_token_ids...)` — i.e. it **always described the
REJECTED DRAFT PROPOSAL**, never the token actually emitted, whenever the two
differ. The recovery kernel's own `NO_DRAFT_PROBS` branch masks out
`draft_token_id` from resampling (`mask = vocab_offset != draft_token_id`),
which proves by construction that a recovered token can never equal the
rejected proposal — so the "mean rank 1627" story was reporting the *thing
that got rejected*, not the *thing that got emitted*.

**Fix:** added `emitted_p` / `emitted_target_rank` / `emitted_top1_shortfall`
to `patches/lenience_trace.py`'s tracer (populated only when `not accepted`,
i.e. exactly where draft ≠ emitted), with two assertions: `emitted != draft`
on recovered rows, and `emitted_p > 0`. Re-ran full 8-group data collection
after the fix. **Corrected picture, verified against real examples** (e.g.
`case_002`, loop at tokens 6114–6434, period 32): the drafter repeatedly
proposed a *deviating* token at target-rank 3–6 with decaying probability
(down to ~4.5e-6), the target rejected it every time, and recovery
consistently re-emitted `" AB"` — the SAME token the loop had been repeating
— at **rank 0, probability climbing to 0.9998**. Recovery is target
self-correction pulling generation back onto its own increasingly-confident
periodic trajectory, not a drift into atypical territory. Corrected
rank-distribution numbers (`scripts/analyze_loop_rank_distribution.py`,
n=77,400 in-loop tokens):

| population | n | median rank | P(rank=0) | mean rank | mean p |
|---|---:|---:|---:|---:|---:|
| all in-loop tokens | 77,400 | 0 | 0.961 | 13.66 | — |
| accepted_draft | 76,752 | 0 | 0.969 | 0.04 | — |
| recovered — **rejected draft proposal's own rank** (`target_rank`, the pre-fix, mislabeled quantity) | 648 | 10 | 0.000 | 1626.74 | — |
| recovered — **actual emitted/recovery token's rank** (`emitted_target_rank`, post-fix, the correct quantity) | 648 | **0** | **0.975** | **0.37** | **0.911** |

Same 648 rows, two different token identities measured on them. The pre-fix
row is what "recovered tokens are deep in the target's tail" was actually
reporting — the rejected proposal, not the recovery. The post-fix row is the
recovery token itself: rank 0 in 97.5% of cases, mean probability 0.91.
Recovery restores the loop's own dominant token; it does not drift into the
tail. Full stratified output (including the `lossy_only_accepted` split) in
`data/loop_token_metrics/rigor_analysis/rank_distribution.txt`.

## Established finding: loops are entrenchment processes, not static states

Cross-cycle trajectory (survivorship-bias corrected two independent ways —
balanced cohort of loops reaching ≥5 cycles, n=464, with bootstrap CIs; and
within-loop fixed-effects/demeaning regression across all 689 loops, both
full-range and restricted to cycles 0–4 to match the cohort window):

| metric | balanced-cohort Δ @ cycle 4 | fixed-effects Δ @ cycle 4 (cycles 0-4) |
|---|---:|---:|
| p (target prob of the repeated token) | +0.1068 | +0.1191 |
| entropy | −0.2937 | −0.3277 |
| target_top1_shortfall | −0.0104 | −0.0277 |
| strict_would_accept rate | +0.1250 | +0.1329 |

Both methods agree closely (t-stats 14–30 on the full-range fixed-effects
fit). **The loop gets more confident and lower-entropy the longer it runs** —
by cycle 4 it's already ~12.5 points more likely to be strict-compatible than
at cycle 0. This means relaxation's role is concentrated at **onset**, not
throughout: `scripts/analyze_loop_gatekeeper_offsets.py` shows 79% of loops
have a single dominant "gatekeeper" offset within the repeat period where
lossy-only acceptance concentrates (>=3x the mean-offset rate), and while 99%
of loops (682/689) contain *at least one* lossy-only-accepted token, only
~11% of individual in-loop tokens are lossy-only-accepted — 89% would have
been accepted under strict decoding too, once the loop is running. Only 7/689
loops (1.0%) have zero lossy-only tokens anywhere (would have looped under
strict regardless). Relaxation's causal role looks like it **opens the door
at specific positions**, then the loop **sustains itself on its own
increasing confidence**, not on continued relaxation.

## Established finding: escape is overwhelmingly a bonus-channel event

Emission-source breakdown of the 688 escape tokens with a captured value
(689 loops total; one loop's trace ends mid-loop at the token budget, no
escape token exists to record):

| emission_source | n | % |
|---|---:|---:|
| bonus (sampled directly from target, zero verification) | 463 | 67.3% |
| accepted_draft | 183 | 26.6% |
| recovered | 42 | 6.1% |

Bonus tokens are only ~13.6% of raw in-loop tokens, so this is a real
enrichment (~13x by odds ratio against the in-loop bonus base rate, not the
10.7% global rate — the in-loop rate is the correct denominator since it's
the opportunity population). The natural read: **the one point in the
pipeline where the drafter gets no vote at all** (bonus tokens are sampled
straight from the target post-hoc, no accept/reject test) is where the loop
is most likely to break.

## Wrong turn #4 / resolved: "randomness breaks the loop" vs "the target had already moved on"

**First framing (this doc's own earlier draft):** "the loop is broken almost
exclusively by genuine target-model sampling randomness... not by an atypical
token either" — based on escape tokens showing low rank and a falling
`target_top1_prob` (~0.92–0.99 deep in-loop → ~0.50–0.63 at escape) but never
directly testing which of two mechanisms that implies:

- **(A) Stochastic escape** — the target still prefers the loop
  (rank_expected=0) at the escape position; the bonus draw is genuinely
  unverified and happens to miss the mode.
- **(B) Modal switch** — the target's own top-1 pick had *already* moved off
  the loop (rank_expected>0, rank_actual=0) before the sample was drawn;
  bonus sampling only exposes an already-changed preference, no stochasticity
  needed to explain the escape.

The falling-top1_prob evidence is consistent with either story on its own,
and it was initially read as favoring (B) — the model's own distribution
"must have" already shifted given how low top1_prob gets.

**Decisive test run:** for all 463 bonus-source escapes minus 6 filtered as
boundary artifacts (the "escape" token happened to literally equal the
periodic-continuation token — pattern hadn't really broken), replayed the
**exact prefix** (original prompt + everything generated up to but not
including the escape token) through the **target model alone** — no drafter,
no relaxation rule, `temperature=0` — and read off its own argmax at that
position. Classified against both the periodic-continuation token
(`expected`) and the token the bonus draw actually emitted (`actual`), which
are never equal by construction in this set:

| category | n | % |
|---|---:|---:|
| **stochastic_escape** (target argmax == expected, i.e. still wanted the loop) | 351 | **76.8%** |
| **modal_switch** (target argmax == actual, i.e. already preferred the escape) | 89 | **19.5%** |
| nonmodal_competition (argmax == neither) | 17 | 3.7% |

**Hypothesis A wins, ~4:1.** In more than three-quarters of bonus escapes the
target's own greedy pick, re-derived from scratch at that exact position, was
still the loop-continuing token — the escape happened only because the
unverified bonus draw landed elsewhere anyway. Modal switch is real (~1 in 5)
but a minority mechanism.

**How this reconciles with the falling-top1_prob evidence that seemed to
favor B:** a top1_prob collapsing toward ~0.5–0.6 *without the argmax label
itself moving* is exactly the regime where sampling deviates from the mode
most of the time even though the mode hasn't changed. Entrenchment erodes the
loop's probability *mass* well before it erodes its rank-0 *status* — so the
original "sampling randomness breaks the loop" framing turns out to be
directionally right, just imprecise about the mechanism: not "randomness
overrides a confidently-attached target" (which would need top1_prob to stay
high), but "confidence erodes into a near-coin-flip regime while preference
hasn't yet flipped, and an unverified draw exploits exactly that."

## Current best-understood mechanism (synthesis, still provisional)

1. **Onset**: relaxation opens acceptance at one or two specific "gatekeeper"
   offsets in a candidate repeated phrase — most individual tokens in the
   eventual loop would have been strict-compatible anyway, but the specific
   token(s) that let the *pattern first repeat* usually would not have been.
2. **Entrenchment**: once repeating, the loop's own probability mass and
   strict-compatibility rise monotonically with cycle count (fixed-effects
   t-stats 14–30) — self-reinforcing, not sustained by continued relaxation.
   By ~4 cycles in, ~89% of in-loop tokens are already strict-acceptable.
3. **Deviation attempts get corrected, not accepted**: when the drafter
   proposes something other than the loop's own token, the target rejects it
   and recovery snaps straight back to the loop's dominant token at rank 0 —
   observed directly (case_002, 9 consecutive cycles, recovery p climbing to
   0.9998 on the SAME token being repeated).
4. **Escape happens almost entirely through the bonus channel** (67.3% of
   escapes, ~13x enriched vs. the in-loop bonus opportunity rate) — the one
   step with zero drafter/verification involvement.
5. **Escape is dominantly stochastic override (76.8%), not modal switch
   (19.5%)**: the target usually still "wants" to continue the loop at the
   moment of escape; it just isn't confident enough anymore (top1_prob
   ~0.5–0.6) for an unverified sample to reliably agree with itself.

## Open questions / natural next step

The counterfactual this points to, not yet run: **force multi-token
target-greedy continuation** (not just 1 argmax token) from the same 457
bonus-escape prefixes and see whether the loop actually resumes. If it does
resume in the large majority of the 351 "stochastic_escape" cases, that would
be direct causal confirmation — not just consistent rank evidence — that
sampling stochasticity, and only sampling stochasticity, is doing the
escaping in that category. The replay infrastructure
(`scripts/replay_bonus_escape_argmax.py`) is already built for exactly this;
it currently asks for 1 token (`max_tokens=1`) and would need only that
raised, plus a loop-continuation detector on the resulting text, to run this
next.

## Where the data lives (moved from job scratch to permanent repo paths)

**Loop / baseline / rank / escape data** — `data/loop_token_metrics/`:
- `all_relaxed_greedy_loops.jsonl` — the 689 confirmed-abnormal loops, full
  per-token metrics (context_before_origin, pre_repeat_boundary, every
  in-loop token, escape token), post tracer-fix.
- `baseline.json` — 585,109 non-loop baseline tokens across the same 8 groups,
  for comparison.
- `loop_candidate_events_pre_judge.json` — the 698 algorithmically-extracted
  candidates, pre-LLM-judge (output of `consolidate_loop_events.py`).
- `bonus_escape_events.jsonl` — 463 bonus-escape events with prefix +
  expected/actual token identity (output of `extract_bonus_escape_events.py`).
- `bonus_escape_replay_results.jsonl` — 457 target-argmax replay results
  (output of `replay_bonus_escape_argmax.py`).
- `bonus_escape_classification.json` — the 4-way modal-switch/stochastic
  table above (output of `classify_bonus_escape_mechanism.py`).
- `rigor_analysis/*.txt` — captured stdout of the 5 statistical scripts below,
  for a citable record independent of re-running them.

**Judged loop candidates (verdict + reasoning per candidate)** —
`data/adjacent_loop_judgements/{aime24_lenience0p05Greedy12k,
humaneval_lenience0p05Greedy9k, longbench_v2_greedy_relaxed}_full.jsonl`.

**Charts** — `data/charts/aime24_output_lengths.html` (published artifact
source) + `aime24_completion_lengths_by_arm.json` (underlying per-case data,
4 arms: strict / lenience0.2 / cactus_accept_only0.25 / specCascOpt0.05).

**Reproducible scripts** (all run from repo root) — `scripts/`:
- `consolidate_loop_events.py` — clusters.jsonl → candidate loop events.
- `judge_adjacent_loops.py` *(pre-existing)* — candidates → verdicts.
- `collect_loop_token_metrics.py` / `collect_loop_baseline_metrics.py`
  *(pre-existing, updated for the tracer fix)* — verdicts → full metrics.
- `analyze_loop_rank_distribution.py`, `analyze_loop_gatekeeper_offsets.py`,
  `analyze_loop_balanced_cohort.py`, `analyze_loop_fixed_effects.py`,
  `analyze_loop_fixed_effects_early.py` — the statistical analyses above.
- `extract_bonus_escape_events.py`, `replay_bonus_escape_argmax.py`,
  `classify_bonus_escape_mechanism.py` — the decisive escape-mechanism test.

All of the above were re-run from their new locations and diffed byte-
identical against the originals before the job-scratch copies were treated as
disposable.
