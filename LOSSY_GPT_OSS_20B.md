# GPT-OSS-20B lossy speculative decoding pilot

This pilot uses SGLang's existing EAGLE-3 lenience control. It does not patch
SGLang and does not install the paper's older Fast-HSD environment.

## Why Lenience first

The paper defines lenience acceptance as
`min(1, p(x) / (ell * q(x)))`. In its EAGLE-3 tree-verification setup, draft
probabilities do not participate in verification (`q=1`). SGLang exposes the
same target-only relaxation as
`--speculative-accept-threshold-acc ell`, which raises acceptance from `p` to
`min(1, p / ell)`.

This makes Lenience the smallest implementation change: one existing server
argument. SpecCascade instead needs a verifier change because its cutoff is
relative to the target distribution's maximum probability at each step.

Important: the paper's conspicuous qualitative degeneration examples concern
truncation-based verification (SpecCascade/Typical Acceptance). This Lenience
pilot proves that a lossy path can be exercised on GPT-OSS-20B; it is not a
guarantee that the same repetition failure will appear. If all Lenience runs
terminate normally, the next method to port should be SpecCascade.

## Storage budget

- The primary GPT-OSS-20B safetensor shards are about 13.8 GB.
- `nebius/EAGLE3-gpt-oss-20b` is about 701 MB.
- Do not clone the full GPT-OSS Hugging Face repository: its `original/` and
  `metal/` variants make the repository listing much larger than the primary
  SGLang weights.
- Reuse the existing Hugging Face cache. Run `remote/preflight.sh` before any
  download. With only 18 GB free, avoid a second `--local-dir` copy of an
  already cached target model.

The `gpt_oss_120b_...` prompt directory name is historical. GPT-OSS-20B and
GPT-OSS-120B use the same Harmony encoding, so the archived tokenization and
rendered prompts remain usable for this 20B pilot. The server records its own
prompt-token count again in every run.

## 0. Copy the small experiment bundle to the H100

Copy these paths while preserving their relative layout:

```text
LOSSY_GPT_OSS_20B.md
remote/
scripts/run_lossy_experiment.py
prompts/gpt_oss_120b_9k_11k_leval/
```

The prompt archive is small; the raw L-Eval dataset and local tokenizer
environment do not need to be uploaded.

## 1. Preflight

```bash
bash remote/preflight.sh
```

All three required SGLang flags must print `yes`. The target model may be an HF
ID or an existing local path:

```bash
export MODEL_PATH=/path/to/existing/gpt-oss-20b
export DRAFT_MODEL_PATH=nebius/EAGLE3-gpt-oss-20b
```

Using the HF ID for the draft lets SGLang fetch only the approximately 701 MB
draft checkpoint. If the target is not cached, first confirm that at least
about 16 GB is genuinely available in the filesystem containing `HF_HOME`.

## 2. Short smoke test

Start each server in its own terminal. Stop it before changing modes.

Baseline:

```bash
bash remote/run_server.sh baseline 2>&1 | tee server-baseline.log
```

In a second terminal, use one long case but cap the response at 64 tokens to
verify request formatting and output capture:

```bash
python3 scripts/run_lossy_experiment.py \
  --mode baseline --cases case_001 --max-new-tokens 64 --tag smoke_baseline
```

Repeat with strict EAGLE-3:

```bash
bash remote/run_server.sh strict 2>&1 | tee server-strict.log
python3 scripts/run_lossy_experiment.py \
  --mode strict --cases case_001 --max-new-tokens 64 --tag smoke_strict
```

Then lossy Lenience. `ell=0.2` is the aggressive end of the paper's
`{0.2, 0.4, 0.6, 0.8}` sweep and is the best first attempt to expose a failure:

```bash
LENIENCE=0.2 bash remote/run_server.sh lossy 2>&1 | tee server-lossy-l0p2.log
python3 scripts/run_lossy_experiment.py \
  --mode lossy --lenience 0.2 --cases case_001 \
  --max-new-tokens 64 --tag smoke_lossy_l0p2
```

Do not use `temperature=0`: SGLang selects its greedy verifier in that case,
which bypasses the probabilistic `threshold_acc` path being tested.

## 3. Paired long-prompt run

Use the same default sampling settings in all modes: temperature 0.7, top-p
1.0, seed 0, repetition penalty 1.0, and 4096 maximum new tokens. The default
case selection is the five L-Eval prompts marked `selected_for_pilot`.

With the baseline server running:

```bash
python3 scripts/run_lossy_experiment.py --mode baseline
```

Restart in strict mode:

```bash
python3 scripts/run_lossy_experiment.py --mode strict
```

Restart in lossy mode with `LENIENCE=0.2` and make the recorded client value
match:

```bash
python3 scripts/run_lossy_experiment.py --mode lossy --lenience 0.2
```

Each run is archived under:

```text
runs/case_001/seed_0/baseline/
runs/case_001/seed_0/strict/
runs/case_001/seed_0/lossy_l0p2/
```

Every directory contains the full prompt, full output, request, raw response,
server information, configuration, finish reason, EOS signal, SGLang
acceptance statistics when exposed by the installed version, and a simple
consecutive repeated-token signal.

## 4. Decision after the first pass

A useful paired failure is:

```text
baseline: normal EOS
strict:   normal EOS
lossy:    repetition/rambling and length termination
```

If only one case looks suspicious, rerun that fixed case with several seeds:

```bash
python3 scripts/run_lossy_experiment.py \
  --mode lossy --lenience 0.2 --cases case_003 --seeds 0 1 2 3 4
```

If all Lenience outputs are normal, stop sweeping this method after the paper's
four values (`0.2`, `0.4`, `0.6`, `0.8`). The next scoped step is a small
SpecCascade verifier patch, not a new dataset or a new drafter.
