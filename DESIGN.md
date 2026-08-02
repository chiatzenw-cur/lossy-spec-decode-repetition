# Design: Reproduce GPT-OSS Lossy Speculative-Decoding Repetition

Status: implementation scaffold ready; remote H100 execution pending

## 1. Objective

Reproduce at least one paired GPT-OSS-20B generation in which:

1. ordinary target-model decoding terminates normally;
2. strict EAGLE-3 speculative decoding terminates normally; and
3. lossy speculative decoding, with every other request parameter fixed,
   enters obvious repetition, a reasoning loop, garbage-like degeneration, or
   reaches the output limit without EOS.

The first lossy method is **Lenience**, because current SGLang exposes the
required relaxation as a server argument and therefore needs no verifier
patch. This phase is a failure-reproduction pilot, not a benchmark and not an
algorithm comparison.

## 2. Success criteria

The task is complete only when one of these terminal outcomes is documented.

### 2.1 Reproduction found

For the same prompt, target checkpoint, draft checkpoint, sampling parameters,
and seed:

```text
baseline: normal EOS, no obvious degeneration
strict:   normal EOS, no obvious degeneration
lossy:    repetition/loop/garbage or length termination without EOS
```

The result must replay at least once with the same configuration and seed.

### 2.2 Lenience exhausted without reproduction

All five pilot prompts have been run under baseline, strict EAGLE-3, and the
paper's four Lenience values (`0.2`, `0.4`, `0.6`, `0.8`) with fixed sampling
parameters. All outputs and metadata have been preserved, and the result is
reported honestly as no reproduction under the tested configuration.

This second outcome completes the Lenience pilot but opens a separately scoped
follow-up: implement SpecCascade in the SGLang verifier.

## 3. Non-goals

Do not expand this task into any of the following:

- designing a new lossy algorithm;
- training or modifying a draft model;
- comparing many speculative-decoding frameworks;
- installing the paper's Fast-HSD environment into the existing SGLang
  environment;
- running a full LongBench or L-Eval benchmark;
- building a learned repetition classifier;
- applying a repetition penalty or another mitigation;
- explaining the causal mechanism before a failure is reproducible;
- changing prompt, seed, draft length, output limit, and Lenience value in the
  same experiment;
- implementing SpecCascade before the Lenience decision gate is reached.

## 4. Constraints

### 4.1 Compute

- Remote accelerator: one NVIDIA H100.
- The coding workspace does not itself provide access to the remote H100.
- A human may need to copy the bundle, run commands, and return logs. Do not
  claim remote validation without those logs.

### 4.2 Storage

- The remote machine has approximately 18 GB free.
- The primary GPT-OSS-20B safetensor shards occupy about 13.8 GB.
- `nebius/EAGLE3-gpt-oss-20b` occupies about 701 MB.
- Reuse an existing Hugging Face cache or existing local target path.
- Do not `git clone` the complete GPT-OSS model repository; it includes
  additional weight formats and is much larger than the primary SGLang files.
- Do not create a second local-directory copy of an already cached model.
- Before downloading anything, record the filesystem and cache usage with
  `remote/preflight.sh`.

### 4.3 Framework

- Use the SGLang installation already present on the remote machine.
- The installed build must expose:
  - `--speculative-algorithm`;
  - `--speculative-draft-model-path`; and
  - `--speculative-accept-threshold-acc`.
- If any required flag is missing, stop and report the installed versions. Do
  not mutate the shared environment without explicit approval.

## 5. Algorithm choice

Let `p(x)` be the target probability, `q(x)` the draft probability, and
`ell` the Lenience factor. Standard Lenience accepts a draft token with:

```text
min(1, p(x) / (ell * q(x)))
```

In the paper's EAGLE-3 tree-verification formulation, draft probabilities do
not participate in verification, so the effective verifier uses `q=1`:

```text
min(1, p(x) / ell)
```

SGLang implements this target-only relaxation through:

```text
--speculative-accept-threshold-acc ell
```

The three server modes are therefore:

| Mode | EAGLE-3 | `threshold_acc` | Purpose |
|---|---:|---:|---|
| `baseline` | no | n/a | Target-only control |
| `strict` | yes | `1.0` | Lossless speculative control |
| `lossy` | yes | `0.2` initially | Aggressive Lenience pilot |

Keep `--speculative-accept-threshold-single 1.0` in both speculative modes so
that it does not introduce a separate hard-accept rule.

All experiment requests must use `temperature > 0`. With `temperature=0`,
SGLang selects its greedy verification path and the probabilistic
`threshold_acc` relaxation under test is not exercised.

## 6. Models and prompts

### 6.1 Models

- Target: `openai/gpt-oss-20b`, or a byte-identical existing local snapshot.
- Draft: `nebius/EAGLE3-gpt-oss-20b`.
- Served name: `gpt-oss-20b`.
- Default EAGLE configuration:
  - draft steps: `6`;
  - EAGLE top-k: `1`;
  - maximum draft tokens: `7`.

Record exact local paths or Hugging Face revisions in every final report. If a
remote deployment already pins revisions, preserve those pins.

### 6.2 Prompt corpus

Use the archived prompts under:

```text
prompts/gpt_oss_120b_9k_11k_leval/
```

The directory name is historical. GPT-OSS-20B and GPT-OSS-120B use the same
`o200k_harmony` encoding, so the rendered Harmony prompts are valid for the
20B pilot. The server must still report its own prompt-token count in each run.

The initial pilot consists of the five entries marked
`selected_for_pilot=true`:

```text
case_001
case_002
case_003
case_004
case_005
```

Do not regenerate or re-render these prompts remotely. Send each
`rendered_prompt.txt` directly to SGLang's native `/generate` endpoint so that
the chat template is not applied twice.

## 7. Fixed request configuration

Unless a phase below explicitly says otherwise, use:

```json
{
  "temperature": 0.7,
  "top_p": 1.0,
  "top_k": -1,
  "sampling_seed": 0,
  "max_new_tokens": 4096,
  "repetition_penalty": 1.0,
  "skip_special_tokens": false,
  "spaces_between_special_tokens": false
}
```

`repetition_penalty=1.0` is intentional. A penalty could conceal the failure
being investigated.

The only intended difference between strict and lossy runs is the server-side
`threshold_acc` value.

## 8. Existing implementation

Agents must inspect and preserve these files before changing anything:

| Path | Responsibility |
|---|---|
| `remote/preflight.sh` | Reports disk, GPU, package versions, required flags, and cached snapshots |
| `remote/run_server.sh` | Starts baseline, strict, or lossy SGLang server |
| `scripts/run_lossy_experiment.py` | Sends archived prompts and stores complete run artifacts |
| `LOSSY_GPT_OSS_20B.md` | Human-facing remote runbook |
| `prompts/gpt_oss_120b_9k_11k_leval/` | Five pilot prompts plus three reserve prompts |

The request runner currently records:

- full prompt and full generated text;
- exact request payload and raw response;
- prompt and completion token counts;
- seed and sampling parameters;
- finish reason and derived EOS status;
- wall-clock time;
- SGLang `spec_accept_rate`, `spec_accept_length`, and `spec_verify_ct` when
  exposed by the installed version;
- a simple signal for 8-, 16-, or 32-token blocks repeated three times
  consecutively;
- `/get_server_info` output.

Local validation already completed:

- Python syntax check;
- default selection of the five pilot cases;
- rejection of `temperature=0` and invalid Lenience values;
- mock `/generate` end-to-end test, including result archival and repeated
  token detection.

This is not evidence that the scripts work with the remote SGLang build. That
requires the phases below.

## 9. Execution plan

### Phase 0: Preserve and inventory

1. Inspect the repository status and avoid overwriting unrelated user files.
2. Copy only the experiment bundle to the remote machine:

   ```text
   DESIGN.md
   LOSSY_GPT_OSS_20B.md
   remote/
   scripts/run_lossy_experiment.py
   prompts/gpt_oss_120b_9k_11k_leval/
   ```

3. Run:

   ```bash
   bash remote/preflight.sh
   ```

4. Save the output as `runs/environment-preflight.txt` or return it to the
   coding agent.

Exit criteria:

- H100 is visible;
- SGLang and dependencies are identified;
- all required SGLang flags are present;
- the target checkpoint path is known;
- sufficient disk space exists for any missing draft files.

If an exit criterion fails, stop and report the blocker. Do not begin a model
download or package upgrade speculatively.

### Phase 1: Baseline smoke test

Start the target-only server:

```bash
MODEL_PATH=/path/to/gpt-oss-20b bash remote/run_server.sh baseline \
  2>&1 | tee server-baseline.log
```

From another terminal:

```bash
python3 scripts/run_lossy_experiment.py \
  --mode baseline \
  --cases case_001 \
  --max-new-tokens 64 \
  --tag smoke_baseline
```

Exit criteria:

- server reaches ready state;
- request returns HTTP 200;
- archived prompt-token count is near the expected 9,046 tokens;
- generated text is readable Harmony output;
- all expected files exist under
  `runs/case_001/seed_0/smoke_baseline/`.

### Phase 2: Strict EAGLE-3 smoke test

Stop the baseline server cleanly, then start:

```bash
MODEL_PATH=/path/to/gpt-oss-20b \
DRAFT_MODEL_PATH=nebius/EAGLE3-gpt-oss-20b \
bash remote/run_server.sh strict 2>&1 | tee server-strict.log
```

Run:

```bash
python3 scripts/run_lossy_experiment.py \
  --mode strict \
  --cases case_001 \
  --max-new-tokens 64 \
  --tag smoke_strict
```

Exit criteria:

- drafter loads without shape, architecture, tokenizer, or CUDA errors;
- output is readable;
- speculative statistics are non-null, either in `meta_info` or server logs;
- `spec_verify_ct > 0` for the request.

If the server runs but speculative statistics remain zero or absent, inspect
the server startup and decode logs before continuing. Do not label the run
strict speculative decoding until draft verification is confirmed.

### Phase 3: Lossy-path smoke test

Stop the strict server and start the aggressive Lenience setting:

```bash
MODEL_PATH=/path/to/gpt-oss-20b \
DRAFT_MODEL_PATH=nebius/EAGLE3-gpt-oss-20b \
LENIENCE=0.2 \
bash remote/run_server.sh lossy 2>&1 | tee server-lossy-l0p2.log
```

Run:

```bash
python3 scripts/run_lossy_experiment.py \
  --mode lossy \
  --lenience 0.2 \
  --cases case_001 \
  --max-new-tokens 64 \
  --tag smoke_lossy_l0p2
```

Exit criteria:

- request succeeds;
- `server_info.json` or the saved startup command confirms
  `threshold_acc=0.2`;
- speculative verification occurs;
- lossy acceptance statistics differ plausibly from strict statistics.

Current SGLang metadata may not expose a per-token
`lossy_only_accepted_tokens` counter. Do not invent this value. If proving
lossy-only acceptance is required, add narrowly scoped instrumentation in the
verifier as a later task and record the exact SGLang revision.

### Phase 4: Five-prompt paired pilot

Run all five default prompts once under each server mode. Restart the server
between modes and preserve each complete server log.

Baseline client:

```bash
python3 scripts/run_lossy_experiment.py --mode baseline
```

Strict client:

```bash
python3 scripts/run_lossy_experiment.py --mode strict
```

Lossy client, while the server uses `LENIENCE=0.2`:

```bash
python3 scripts/run_lossy_experiment.py --mode lossy --lenience 0.2
```

Expected result layout:

```text
runs/
  case_001/
    seed_0/
      baseline/
      strict/
      lossy_l0p2/
  ...
  case_005/
```

For every case, compare:

- `run.json` finish reason and EOS status;
- output token count;
- consecutive-repeat signal;
- full `output.txt`, especially reasoning-channel loops;
- speculative acceptance length and rate;
- server-side errors or warnings.

### Phase 5: Candidate replay

If any lossy output is suspicious while its baseline and strict controls are
normal:

1. freeze prompt, checkpoints, EAGLE parameters, Lenience value, request
   parameters, and seed;
2. rerun the exact lossy command at least once;
3. verify that the same class of degeneration recurs;
4. then run seeds `0 1 2 3 4` for the candidate case:

   ```bash
   python3 scripts/run_lossy_experiment.py \
     --mode lossy \
     --lenience 0.2 \
     --cases case_003 \
     --seeds 0 1 2 3 4
   ```

Use unique tags or `--overwrite` deliberately. Never overwrite the first
successful failure artifact accidentally.

### Phase 6: Bounded Lenience sweep

If no failure occurs at `ell=0.2`, repeat the lossy mode for `0.4`, `0.6`, and
`0.8`, changing only `LENIENCE` and the matching client `--lenience` argument.

Although `0.2` is the most permissive setting, run the remaining paper values
because stochastic interactions may not be monotonic for a small prompt set.
Do not alter the prompt pool, sampling parameters, or EAGLE block at the same
time.

After all four values, enter one of the terminal outcomes in Section 2.

## 10. Failure classification

Automatic signals are candidate selectors, not final labels.

### 10.1 Automatic signals

- `finish_reason == "length"` and `eos_reached == false`;
- an 8-, 16-, or 32-token span repeats at least three times consecutively;
- lossy output is more than twice the strict output length;
- output continues to the 4,096-token limit;
- speculative acceptance length rises substantially relative to strict.

### 10.2 Human labels

Assign one concise label after reading the full output:

- `normal`;
- `long_but_valid`;
- `lexical_repetition`;
- `reasoning_loop`;
- `rambling`;
- `garbage`;
- `no_eos`.

A length-limit finish by itself is not sufficient if the output remains valid
and the prompt naturally requests a long response.

## 11. Artifact contract

Each run directory must contain:

```text
config.json
output.txt
prompt.txt
request.json
response.json
run.json
server_info.json
```

Preserve corresponding server logs outside the individual request directory.
The final reproduction bundle must also include:

```text
environment-preflight.txt
server-baseline.log
server-strict.log
server-lossy-<setting>.log
reproduction.md
```

`reproduction.md` must state:

```text
Target model and revision/path:
Draft model and revision/path:
SGLang version/commit:
GPU:
Prompt case and input tokens:
Seed:
Sampling parameters:
EAGLE parameters:
Lossy method and threshold:
Baseline result:
Strict result:
Lossy result:
Exact replay commands:
```

## 12. Code-change rules for agents

1. Read the current files before editing; the workspace may contain unrelated
   uncommitted user work.
2. Preserve the native `/generate` request path and pre-rendered Harmony prompt.
3. Keep the runner dependency-free unless the remote environment proves that a
   dependency is necessary.
4. Keep baseline, strict, and lossy output directories disjoint.
5. Reject configurations that silently bypass the tested verifier, especially
   `temperature=0`.
6. Never report inferred server flags as observed facts; archive
   `/get_server_info` and startup logs.
7. Do not mark the reproduction complete from an automatic n-gram signal alone.
8. Add tests for every runner behavior changed locally. Remote-only behavior
   must be supported by returned logs.
9. If the installed SGLang response schema differs, make the smallest
   backward-compatible parser change and retain the raw response.
10. Do not commit generated `runs/` artifacts containing huge outputs unless
    the user explicitly asks; they are ignored by default.

## 13. SpecCascade follow-up gate

The paper's conspicuous qualitative degeneration examples concern
truncation-based verification, including SpecCascade and Typical Acceptance.
Lenience is selected here for implementation simplicity, not because it is the
most likely method to reproduce the paper's exact qualitative example.

Only after Section 2.2 is satisfied should a new design be written for
SpecCascade. That design must implement the per-step relative target cutoff:

```text
accept x iff p(x) >= p_base * max_v p(v)
```

SGLang's `--speculative-accept-threshold-single` is an absolute probability
threshold and is therefore not an implementation of SpecCascade. Do not
mislabel it as one.

The SpecCascade follow-up must pin the remote SGLang revision, identify the
exact verifier function used by that revision, add unit tests for boundary
probabilities, and retain baseline/strict parity before running long prompts.

## 14. References

- Lossy-verification paper:
  <https://arxiv.org/html/2607.26627>
- Paper implementation reference:
  <https://github.com/ZhouYuxuanYX/Fast-HSD>
- SGLang server arguments:
  <https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md>
- Current SGLang EAGLE verifier source:
  <https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/eagle_utils.py>
- GPT-OSS-20B:
  <https://huggingface.co/openai/gpt-oss-20b>
- EAGLE-3 GPT-OSS-20B draft:
  <https://huggingface.co/nebius/EAGLE3-gpt-oss-20b>
