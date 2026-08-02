# Resolved environment for the GPT-OSS-20B pilot

Recorded 2026-08-02. `remote/preflight.sh` captures versions and flags; this file
records the facts it does not (checkpoint revisions, why versions are pinned).

**The project has switched from SGLang to vLLM.** The vLLM environment is the
active one; the SGLang section below is retained because `BASELINE_RECORD.md` was
gathered under it and its constraints explain several decisions.

---

# Active: vLLM (`.venv-vllm`)

Target environment, from the paper: vLLM 0.20.1, torch 2.11.0+cu130, single
H100/B200, no tensor/data/pipeline parallelism and no FSDP.

What we run, and the one forced deviation:

| | paper | here | why |
|---|---|---|---|
| vLLM | 0.20.1 | **0.20.1** | matched |
| torch | 2.11.0 | **2.11.0** | matched |
| CUDA build | cu130 | **cu129** | forced — see below |
| GPU | H100 / B200 | H100 PCIe | matched (H100) |
| parallelism | none | none (`--tensor-parallel-size 1 --pipeline-parallel-size 1`) | matched |

### Why cu129 and not cu130

The driver here is 570.195.03 (CUDA 12.8). A cu130 build needs driver 580+, and
the driver cannot be changed. cu130 is a *major* version step, so there is no
compatibility path — PyPI's default `torch==2.11.0` is a cu130 build and reports
"driver too old", and every compiled extension in the PyPI `vllm==0.20.1` wheel
(`_C`, `_moe_C`, `_vllm_fa2_C`, `_vllm_fa3_C`, ...) links `libcudart.so.13`.

cu129 is a *minor* version step, which CUDA covers with minor-version
compatibility: a 12.9-built binary runs on a 12.x driver. Verified empirically —
`torch 2.11.0+cu129` reports `cuda available: True`, `NVIDIA H100 PCIe`, and a
GPU matmul returns finite values on the 12.8 driver.

vLLM publishes a matching cu129 build at `wheels.vllm.ai/0.20.1/cu129/`, so the
vLLM and torch versions are both exact matches to the paper.

### Install

Order matters: installing vLLM pulls PyPI's cu130 torch, so torch is restored
from the cu129 index afterwards. Same for torchvision/torchaudio, whose PyPI
builds are also cu13-linked.

```bash
uv venv --python 3.12 .venv-vllm
uv pip install --python .venv-vllm/bin/python \
  --index-url https://download.pytorch.org/whl/cu129 torch==2.11.0
uv pip install --python .venv-vllm/bin/python "vllm==0.20.1+cu129" \
  --extra-index-url https://wheels.vllm.ai/0.20.1/cu129/ --index-strategy unsafe-best-match
uv pip install --python .venv-vllm/bin/python \
  --index-url https://download.pytorch.org/whl/cu129 \
  --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
```

Verify afterwards — this must print only `cudnn/_compiled_module` and
`tilelang/lib/libcudart_stub.so`, neither of which is on the serving path:

```bash
cd .venv-vllm/lib/python3.12/site-packages
for so in $(find . -name "*.so" -not -path "./torch/*" -not -path "./nvidia/*"); do
  strings "$so" 2>/dev/null | grep -q "libcudart\.so\.13" && echo "CU13: $so"
done
```

### Acceptance-rule knobs in vLLM 0.20.1

`rejection_sample_method` is `Literal["strict", "probabilistic", "synthetic"]`
(`vllm/config/speculative.py:66`):

- `strict` (vLLM's default) — target and draft sampled tokens must match
  exactly. Harsh, and *not* what the draft model cards are benchmarked under.
- `probabilistic` — standard rejection sampling, `min(1, p/q)`. Lossless with
  respect to the target distribution. This is the correct **strict control** arm,
  and the rule nebius measured its acceptance length of ~3.3 under.
- `synthetic` — accepts draft tokens at prescribed per-position rates
  *irrespective of the target distribution*, driven by
  `synthetic_acceptance_length` or `synthetic_acceptance_rates`. This is a lossy
  verifier available as configuration, with no verifier patch required.

There is **no Lenience** (`min(1, p/(ell*q))`) control in vLLM 0.20.1. Lenience
specifically would require patching `vllm/v1/sample/rejection_sampler.py`.

---

# Superseded: SGLang (`.venv`)

Retained because `BASELINE_RECORD.md` was gathered here. See
`SPECULATIVE_BLOCKERS.md` for why the speculative arms could not proceed on it.

## Machine

The H100 is the same machine as the coding workspace, not a separate remote box.

- GPU: NVIDIA H100 PCIe, 81,559 MiB
- Driver: 570.195.03 (CUDA 12.8)
- Disk: `/dev/vda1`, 96 G total (~37 G free after both checkpoints)

## Interpreter

Dedicated venv at `./.venv` (gitignored). Every command must pass
`PYTHON=$PWD/.venv/bin/python`; the system `python3` has no SGLang.

- Python 3.12.3
- sglang 0.5.10.post1
- sglang-kernel 0.4.1
- torch 2.9.1+cu128
- transformers 5.3.0
- huggingface-hub 1.26.0

Installed with a plain `uv pip install sglang==0.5.10.post1` — no version
overrides are needed, because this release's whole stack is CUDA-12 native.

### Why this version, and what fails above it

sglang 0.5.11 and newer cannot run on this driver. The reasons compound:

- **0.5.13+ hard-requires `flashinfer_python[cu13]`**, which needs driver 580 or
  newer. This driver is 570 (CUDA 12.8).
- **0.5.11 and 0.5.12 pin `torch==2.11.0`**, whose PyPI default is a cu130 build:
  `torch.cuda.is_available()` is `False` with "driver too old". Reinstalling
  torch from `download.pytorch.org/whl/cu128` fixes torch itself but not the rest
  of the stack — the binary dependencies are still built against CUDA 13, and the
  server dies at startup on `deep_gemm/_C.so: libcudart.so.13: cannot open shared
  object file`. Removing `deep_gemm` does not rescue it either: the mandatory
  `sglang-kernel==0.4.2.post2` wheel is itself cu13-linked
  (`sgl_kernel/sm90/common_ops.abi3.so`), and PyPI ships no cu12 variant. That
  package supplies the speculative verify kernel, so it cannot be dropped.
- **0.5.10.post1 pins `torch==2.9.1`, whose PyPI default *is* cu128**
  (`nvidia-cuda-runtime-cu12==12.8.90`), and its `sglang-kernel==0.4.1` is
  CUDA-12 linked. Nothing needs patching.

A useful check after any dependency change — this must print nothing except the
`cudnn/_compiled_module` entry, which is unused here:

```bash
cd .venv/lib/python3.12/site-packages
for so in $(find . -name "*.so" -not -path "./torch/*" -not -path "./nvidia/*"); do
  strings "$so" 2>/dev/null | grep -q "libcudart\.so\.13" && echo "CU13: $so"
done
```

Upgrading sglang here requires a driver upgrade to 580+ first, not a pip fix.

## Build toolchain

SGLang JIT-compiles CUDA kernels during startup (`sglang/jit_kernel/rope.py` via
`tvm_ffi`), shelling out to `ninja` and `nvcc`. Passing `PYTHON=.../.venv/bin/python`
does not put the venv's `bin/` on `PATH`, so the server used to die with
`FileNotFoundError: [Errno 2] No such file or directory: 'ninja'`, reported only
indirectly as `Received sigquit from a child process`. `remote/run_server.sh` now
derives the venv bin directory from `$PYTHON`, adds `$CUDA_HOME/bin`, and fails
fast with a clear message if either tool is still missing.

- `ninja`: `.venv/bin/ninja` (pulled in as an sglang dependency)
- `nvcc`: `/usr/local/cuda/bin/nvcc`, release 12.8.93 — matches the driver
- `g++`/`gcc`: `/usr/bin`, system default

First server start pays the JIT compile cost; later starts reuse
`~/.cache/flashinfer` and the tvm_ffi build cache.

### Verified in this build

- `--speculative-algorithm`, `--speculative-draft-model-path`, and
  `--speculative-accept-threshold-acc` all present.
- `threshold_acc` reaches the EAGLE verify call at
  `sglang/srt/speculative/eagle_info.py:394` and `eagle_info_v2.py:455`. (In
  0.5.16 this same wiring lives in `eagle_utils.py`; the flag is not inert here.)
- `srt/models/gpt_oss.py` sets `capture_aux_hidden_states`, and
  `srt/models/llama_eagle3.py` exists, so EAGLE-3 drafting on this target is
  supported.
- `sampling_seed` is a real sampling parameter
  (`srt/sampling/sampling_params.py:112`), and `/generate` returns `output_ids`,
  so the runner's seed control and repeat detector both work.

## Checkpoints

Both resolved from the shared Hugging Face cache; no `--local-dir` copies.

| Role | Repo | Revision |
|---|---|---|
| Target | `openai/gpt-oss-20b` | `6cee5e81ee83917806bbde320786a8fb61efebee` |
| Draft | `nebius/EAGLE3-gpt-oss-20b` | `9dd45bb1da8b1ddc8cabd52691d9ac170b41484f` |

The target was fetched with `--exclude "original/*" --exclude "metal/*"`; only the
three primary safetensor shards are present (13,761,264,768 bytes per
`model.safetensors.index.json`).

## Reproduce the environment

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision==0.26.0
uv pip install --python .venv/bin/python sglang==0.5.12.post1
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 --reinstall-package torch torch==2.11.0
uv pip install --python .venv/bin/python kernels==0.12.3
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```
