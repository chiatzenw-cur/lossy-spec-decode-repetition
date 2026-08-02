> **Resolved by switching to vLLM (2026-08-02).** Both blockers below are SGLang
> blockers. See "Resolution" at the end and `remote/ENVIRONMENT.md`. Kept because
> it records why the switch happened and what the SGLang numbers were.

# Speculative arms: two blockers found before any strict/lossy round

Recorded 2026-08-02, on the environment in `remote/ENVIRONMENT.md`
(SGLang 0.5.10.post1, H100 PCIe, `openai/gpt-oss-20b` @ `6cee5e81`).

Both were found by smoke-testing `strict` on `case_003` before committing to a
full round. Neither is visible from a run that "succeeds" — both servers started,
returned HTTP 200, and produced readable output with clean EOS.

## Blocker 1 — deterministic inference does not hold under EAGLE-3

The recipe that makes the baseline byte-reproducible
(`--enable-deterministic-inference --random-seed 0 --disable-radix-cache`, see
`BASELINE_RECORD.md`) does **not** carry over to speculative decoding.

SGLang accepts the combination without complaint: `speculative_algorithm='EAGLE3'`
alongside `enable_deterministic_inference=True`, `batch_invariant_ops` loaded,
`disable_overlap_schedule=True` set automatically. It starts and it drafts.

Three identical requests, same seed, radix cache already off:

| run | output tokens | vs previous |
|---|---:|---|
| `sdet_a` | 1,388 | — |
| `sdet_b` | 1,431 | diverge at token 4 |
| `sdet_c` | 1,208 | diverge at token 9 |

Baseline replays exactly under those same flags, so the gap is specific to the
speculative path: SGLang's batch-invariant kernels do not cover EAGLE-3 tree
verification in this build.

Consequence: DESIGN §2.1's "must replay at least once with the same
configuration and seed" cannot be satisfied for the strict or lossy arms as
things stand, and DESIGN Phase 5 (candidate replay) has no foundation.

## Blocker 2 — the draft model accepts almost nothing

More damaging, and independent of Blocker 1.

| configuration | `spec_verify_ct` | `spec_accept_length` | `spec_accept_rate` |
|---|---:|---:|---:|
| strict + deterministic | 1,340 | 1.036 | 0.0058 |
| strict, no determinism | 2,395 | 1.035 | 0.0058 |
| nebius model card (K=7, temp 1) | — | **3.29** | — |

Acceptance is identical with determinism on and off, so deterministic mode is not
the cause. An acceptance length of 1.035 means essentially only the bonus token
is kept — under 0.6% of drafted tokens survive verification. The drafter is
running and being verified; it is simply almost never right.

**This invalidates the Lenience pilot in its current form.** Lenience works by
relaxing the acceptance threshold on drafted tokens. If the strict arm accepts
0.6% of them, relaxing the threshold has nearly nothing to act on, and any
strict-vs-lossy difference would be measuring a near-no-op rather than the
phenomenon in the paper.

### What is known about the cause

`nebius/EAGLE3-gpt-oss-20b` is packaged for **vLLM**, not SGLang. Its README
documents only `speculative_config={"method": "eagle3", ...}`, and vLLM infers
EAGLE-3 from that config rather than from the checkpoint. Consequences here:

- Its `config.json` declares `architectures: ["LlamaForCausalLM"]`. SGLang loads
  that as a plain Llama and dies on the reduced 64k draft vocab
  (`org_vocab_size=201088` vs `loaded_weight.shape=64000`). Working around this
  is what `models/EAGLE3-gpt-oss-20b-sglang/` exists for — a patched config
  declaring `LlamaForCausalLMEagle3`, with the weights symlinked.
- It carries no `eagle_config`, so SGLang's
  `model_runner.py:366-379` lookup of `eagle_aux_hidden_state_layer_ids` throws
  and falls back to `None`, and `gpt_oss.py:1186-1198` then captures the default
  layers `[2, num_layers//2, num_layers-3]` = `[2, 12, 21]`. If nebius trained
  against different layers, the `fc` fusion (`fc.weight` is 2880x8640, i.e. three
  concatenated 2880-wide hidden states) is fed the wrong inputs — which would
  produce exactly this symptom: a drafter that runs, verifies, and is never right.

This is a hypothesis, not a diagnosis. Checking `RedHatAI/gpt-oss-20b-speculator.eagle3`
did not settle it — that config also has `eagle_aux_hidden_state_layer_ids: null`,
but it additionally declares `norm_before_fc: true` and `norm_before_residual: true`,
architectural switches SGLang's `llama_eagle3.py` does not read at all (it applies
a fixed `hidden_norm` at `:73`/`:86`). So norm placement is a second candidate
mismatch independent of layer ids.

The checkpoint itself is a genuine EAGLE-3 draft — `d2t`, `t2d`, `fc.weight`,
`lm_head` 64000x2880 are all present and correctly shaped. Nothing here suggests
a corrupt download.

## Where this leaves the plan

The baseline arm is complete and sound (`BASELINE_RECORD.md`). The speculative
arms cannot produce a meaningful result until Blocker 2 is resolved — and
Blocker 1 must be resolved too before DESIGN §2.1 can ever be satisfied, though
it does not block gathering descriptive data.

Options, roughly in order of cost:

1. **Find a draft checkpoint packaged for SGLang.** The cheapest test of whether
   SGLang's EAGLE-3 path works at all on gpt-oss-20b. If a known-good checkpoint
   also gives ~1.0 acceptance, the problem is SGLang-side, not the checkpoint.
2. **Determine the correct aux layer ids** and add `eagle_config` to the patched
   config. Requires ground truth from nebius (arXiv 2602.23881 or their training
   code); guessing is a search over layer triples at ~4 min per trial.
3. **Reconsider the method.** DESIGN §13 already anticipates that Lenience may
   not reproduce the paper's qualitative failure and names SpecCascade as the
   follow-up. A drafter this weak makes Lenience an especially poor vehicle.

Blocker 1 additionally means that even after Blocker 2 is fixed, paired
comparison must either rely on many samples per condition rather than exact
replay, or wait on batch-invariant coverage of the EAGLE-3 verify path.

---

## Resolution: switched to vLLM 0.20.1

Rationale beyond the two blockers: the paper's own setup is vLLM 0.20.1 with
torch 2.11.0, single H100/B200, no parallelism. Matching it removes a whole class
of framework-difference confounds. Environment details and the forced cu129
deviation are in `remote/ENVIRONMENT.md`.

Measured on `case_003`, same prompt and sampling parameters throughout:

| stack | draft checkpoint | acceptance rule | accept_len | draft accept rate |
|---|---|---|---:|---:|
| SGLang 0.5.10.post1 | nebius (config patched) | target-only, `threshold_acc=1` | 1.035 | 0.006 |
| SGLang 0.5.10.post1 | zhuyksir (SGLang-native, ctx patched) | target-only, `threshold_acc=1` | 1.419 | 0.070 |
| **vLLM 0.20.1** | **nebius, unmodified** | **`probabilistic` = `min(1, p/q)`** | **2.379** | **0.230** |

Both SGLang blockers are gone:

- **Drafter works.** vLLM loads `nebius/EAGLE3-gpt-oss-20b` as shipped — no
  architecture rewrite, no context-length patch, no aux-layer guessing. The
  checkpoint was always packaged for vLLM; that was the whole problem.
- **A lossy knob exists without patching a verifier.** `rejection_sample_method:
  "synthetic"` with `synthetic_acceptance_length` accepts draft tokens at a
  prescribed rate irrespective of the target distribution.

Remaining gap to the model card's ~3.3: we measure 2.379 at temperature 0.7 with
k=6 on 10k-token L-Eval paper-review prompts; the card reports temperature 1 with
k=7 on MT-Bench/GSM8K/HumanEval. Different sampling, different k, and a very
different task distribution, so the two are not directly comparable — 2.379 is a
healthy working acceptance length, not a symptom.

### What is still open

- **Lenience is not available in vLLM 0.20.1 either.** `rejection_sample_method`
  is `strict | probabilistic | synthetic`; there is no `min(1, p/(ell*q))`. The
  original DESIGN §5 justification for choosing Lenience — "one existing server
  argument, no verifier patch" — no longer holds on either framework. Lenience
  now requires patching `vllm/v1/sample/rejection_sampler.py`. If a patch is
  being written anyway, DESIGN §13 argues SpecCascade is the better target.
- **Replay under speculation is untested on vLLM.** The per-request `seed` is
  plumbed through the runner but has not been verified to reproduce a trajectory
  the way `DETERMINISTIC=1` did for the SGLang baseline. Test it before relying
  on paired comparison.

### Harness

- `remote/run_server_vllm.sh` — baseline / strict / lossy, single-GPU, no parallelism
- `scripts/run_experiment_vllm.py` — same artifact contract, so
  `scripts/summarize_runs.py` works across both backends

Validated: baseline smoke on `case_003` reported `input_tokens = 10384`, exactly
matching the archived count, confirming `add_special_tokens: false` prevents the
pre-rendered Harmony prompt from being re-tokenized with extra specials.
