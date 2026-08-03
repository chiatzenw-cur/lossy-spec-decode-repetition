#!/usr/bin/env python3
"""Acceptance test for the Lenience patch. Run by patches/apply.sh.

Checks the two things that have actually gone wrong here before:

1. the factor reaches the module (it was silently 1.0 when passed via the
   environment, because vLLM sanitises EngineCore's env), and
2. the factor reaches the *kernel* (the V2 file was patched while the V1 runner
   was live, so the lossy arm was bit-identical to strict).

(2) needs a GPU and is skipped without one; (1) is not.
"""

from __future__ import annotations

import math
import os
import pathlib
import subprocess
import sys

FACTOR_FILE = pathlib.Path(f"/tmp/lossy-spec-decode-lenience-{os.getuid()}")
MODULES = (
    "vllm.v1.sample.rejection_sampler",
    "vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils",
)

READ_BACK = """
import importlib, json, sys
out = {}
for name in sys.argv[1:]:
    m = importlib.import_module(name)
    out[name] = {
        "factor": getattr(m, "_LENIENCE_FACTOR", None),
        "log_factor": getattr(m, "_LENIENCE_LOG_FACTOR", None),
        "path": getattr(m, "_LENIENCE_FACTOR_FILE", None),
    }
print("JSON:" + json.dumps(out))
"""


def read_back_in_subprocess() -> dict[str, dict[str, object]]:
    """Import both modules fresh; the factor is read once, at import."""
    import json

    proc = subprocess.run(
        [sys.executable, "-c", READ_BACK, *MODULES],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(f"import failed:\n{proc.stderr[-3000:]}")
    for line in proc.stdout.splitlines():
        if line.startswith("JSON:"):
            return json.loads(line[5:])
    raise AssertionError(f"no result from subprocess:\n{proc.stdout[-2000:]}")


def test_factor_plumbing() -> None:
    saved = FACTOR_FILE.read_text() if FACTOR_FILE.is_file() else None
    try:
        FACTOR_FILE.write_text("0.37\n")
        got = read_back_in_subprocess()
        for name in MODULES:
            assert got[name]["factor"] == 0.37, f"{name}: {got[name]}"
            assert got[name]["path"] == str(FACTOR_FILE), f"{name}: {got[name]}"
        log = got["vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils"]["log_factor"]
        assert log is not None and abs(log - math.log(0.37)) < 1e-12, log
        print(f"  ok  both modules read {FACTOR_FILE} -> 0.37")

        FACTOR_FILE.unlink()
        got = read_back_in_subprocess()
        for name in MODULES:
            assert got[name]["factor"] == 1.0, f"{name}: {got[name]}"
        print("  ok  missing file falls back to 1.0 (stock rule)")
    finally:
        if saved is None:
            FACTOR_FILE.unlink(missing_ok=True)
        else:
            FACTOR_FILE.write_text(saved)


def test_kernel_uses_factor() -> None:
    """Drive the V1 verify kernel directly: no model, no server, no sampler.

    One request per uniform draw, each with a single draft token whose
    target/draft ratio is fixed, so the accept/reject boundary is exactly
    u <= ratio / factor and can be predicted in closed form.
    """
    import torch

    if not torch.cuda.is_available():
        print("  skip  no CUDA device; kernel test not run")
        return
    from vllm.v1.sample.rejection_sampler import rejection_random_sample_kernel

    device = "cuda"
    vocab, draft_tok, recovered_tok, bonus_tok = 8, 3, 5, 7
    uniform = torch.linspace(0.05, 0.95, 19, device=device, dtype=torch.float32)
    n = uniform.numel()

    def run(ratio: float, factor: float) -> torch.Tensor:
        draft_probs = torch.full((n, vocab), 0.5 / (vocab - 1), device=device)
        draft_probs[:, draft_tok] = 0.5
        target_probs = torch.full((n, vocab), (1.0 - 0.5 * ratio) / (vocab - 1), device=device)
        target_probs[:, draft_tok] = 0.5 * ratio
        out = torch.full((n, 2), -1, dtype=torch.int32, device=device)
        rejection_random_sample_kernel[(n,)](
            out,
            torch.arange(1, n + 1, dtype=torch.int32, device=device),
            torch.full((n,), draft_tok, dtype=torch.int32, device=device),
            draft_probs.contiguous(),
            target_probs.contiguous(),
            torch.full((n,), bonus_tok, dtype=torch.int32, device=device),
            torch.full((n,), recovered_tok, dtype=torch.int32, device=device),
            uniform,
            torch.zeros(n, dtype=torch.bool, device=device),
            1,  # max_spec_len
            vocab,
            None,  # synthetic_conditional_rates
            factor,
            NO_DRAFT_PROBS=False,
            SYNTHETIC_MODE=False,
        )
        return out.cpu()

    for ratio, factor in ((0.5, 1.0), (0.5, 0.2), (0.1, 0.2), (0.1, 1.0)):
        out = run(ratio, factor)
        accepted = out[:, 0] == draft_tok
        expected = (uniform.cpu() <= ratio / factor)
        assert torch.equal(accepted, expected), (
            f"ratio={ratio} factor={factor}\n got      {accepted.tolist()}\n expected {expected.tolist()}"
        )
        # An accepted draft is followed by the bonus token; a rejected one is
        # replaced by the recovered token and stops the request.
        assert torch.equal(out[accepted][:, 1], torch.full((int(accepted.sum()),), bonus_tok, dtype=torch.int32))
        assert torch.equal(out[~accepted][:, 0], torch.full((int((~accepted).sum()),), recovered_tok, dtype=torch.int32))
        print(f"  ok  kernel accepts iff u <= p/(q*lam)   p/q={ratio} lam={factor}")

    strict = run(0.5, 1.0)
    lenient = run(0.5, 0.2)
    assert (lenient[:, 0] == draft_tok).sum() > (strict[:, 0] == draft_tok).sum(), (
        "lam=0.2 did not accept more than lam=1.0; the factor is not reaching the kernel"
    )
    print("  ok  lam=0.2 accepts strictly more draft tokens than lam=1.0")


def main() -> int:
    failures = 0
    for test in (test_factor_plumbing, test_kernel_uses_factor):
        print(f"{test.__name__}:")
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {exc}")
    print("FAILED" if failures else "all lenience patch checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
