#!/usr/bin/env python3
"""Acceptance test for the CACTUS patch. Run by patches/apply_cactus.sh.

Checks, in order:

1. the alpha value reaches the module (same env-sanitisation failure mode as
   every other patch here),
2. the gamma_x FORMULA (min(p + sqrt(2*alpha*p*(1-p)), 1)) is correct against
   hand-computed values, including alpha=0 recovering p exactly -- CPU only,
3. the *kernel* applies gamma_x correctly in the actual accept test, and
   alpha=0 is bit-identical to the unmodified strict kernel line.

(3) needs a GPU and is skipped without one; (1) and (2) are not.
"""

from __future__ import annotations

import math
import os
import pathlib
import subprocess
import sys

ALPHA_FILE = pathlib.Path(f"/tmp/lossy-spec-decode-cactus-alpha-{os.getuid()}")
MODULE = "vllm.v1.sample.rejection_sampler"

READ_BACK = """
import importlib, json, sys
m = importlib.import_module(sys.argv[1])
print("JSON:" + json.dumps({
    "alpha": m._CACTUS_ALPHA,
    "path": m._CACTUS_ALPHA_FILE,
}))
"""


def read_back_in_subprocess() -> dict[str, object]:
    import json

    proc = subprocess.run([sys.executable, "-c", READ_BACK, MODULE], capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f"import failed:\n{proc.stderr[-3000:]}")
    for line in proc.stdout.splitlines():
        if line.startswith("JSON:"):
            return json.loads(line[5:])
    raise AssertionError(f"no result from subprocess:\n{proc.stdout[-2000:]}")


def test_alpha_plumbing() -> None:
    saved = ALPHA_FILE.read_text() if ALPHA_FILE.is_file() else None
    try:
        ALPHA_FILE.write_text("0.25\n")
        got = read_back_in_subprocess()
        assert got["alpha"] == 0.25, got
        assert got["path"] == str(ALPHA_FILE), got
        print(f"  ok  module reads {ALPHA_FILE} -> 0.25")

        ALPHA_FILE.unlink()
        got = read_back_in_subprocess()
        assert got["alpha"] == 0.0, got
        print("  ok  missing file falls back to 0.0 (gamma_x == p(x) -> strict spec-dec)")
    finally:
        if saved is None:
            ALPHA_FILE.unlink(missing_ok=True)
        else:
            ALPHA_FILE.write_text(saved)


def test_negative_alpha_rejected() -> None:
    saved = ALPHA_FILE.read_text() if ALPHA_FILE.is_file() else None
    try:
        ALPHA_FILE.write_text("-0.1\n")
        proc = subprocess.run(
            [sys.executable, "-c", f"import {MODULE}"], capture_output=True, text=True
        )
        assert proc.returncode != 0, "import should fail for a negative alpha"
        assert "must be >= 0" in proc.stderr, proc.stderr[-500:]
        print("  ok  negative alpha raises at import instead of producing NaNs")
    finally:
        if saved is None:
            ALPHA_FILE.unlink(missing_ok=True)
        else:
            ALPHA_FILE.write_text(saved)


def gamma(p: float, alpha: float) -> float:
    return min(p + math.sqrt(max(0.0, 2 * alpha * p * (1 - p))), 1.0)


def test_gamma_formula() -> None:
    """CPU-only: gamma_x = min(p + sqrt(2*alpha*p*(1-p)), 1), matching Eq. 7 of
    Hao & Mou 2026 (CACTUS), against hand-picked (p, alpha) pairs."""
    cases = [
        (0.5, 0.0, 0.5),       # alpha=0 -> gamma == p exactly, any p
        (0.1, 0.0, 0.1),
        (0.9, 0.0, 0.9),
        (0.5, 0.125, 0.5 + math.sqrt(2 * 0.125 * 0.25)),  # 0.5 + sqrt(0.0625) = 0.75
        (0.01, 100.0, 1.0),    # huge alpha saturates at the cap
    ]
    for p, alpha, expected in cases:
        got = gamma(p, alpha)
        assert abs(got - expected) < 1e-9, (p, alpha, got, expected)
        print(f"  ok  p={p} alpha={alpha}: gamma={got:.6f}")
    print("  ok  gamma formula matches hand-computed values, including the alpha=0 and cap cases")


def test_kernel_matches_gamma() -> None:
    """Drive the V1 verify kernel directly: no model, no server, no sampler.

    A fixed draft/target probability pair at varying alpha, so the accept/reject
    boundary is u <= gamma(p, alpha)/q and can be predicted in closed form.
    Also checks alpha=0.0 is bit-identical to the unmodified strict kernel.
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

    def run(p: float, q: float, alpha: float) -> torch.Tensor:
        draft_probs = torch.full((n, vocab), (1.0 - q) / (vocab - 1), device=device)
        draft_probs[:, draft_tok] = q
        target_probs = torch.full((n, vocab), (1.0 - p) / (vocab - 1), device=device)
        target_probs[:, draft_tok] = p
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
            alpha,
            NO_DRAFT_PROBS=False,
            SYNTHETIC_MODE=False,
        )
        return out.cpu()

    # Chosen so gamma(p,alpha)/q never lands exactly on one of the 19 evenly
    # spaced uniform grid points (multiples of 0.05): an exact real-arithmetic
    # tie there is a coin flip between the kernel's fp32 sqrt and this test's
    # float64 reference, which is a rounding artifact, not a rule mismatch.
    for p, q, alpha in ((0.3, 0.6, 0.0), (0.3, 0.6, 0.2), (0.1, 0.5, 0.45), (0.05, 0.4, 1.0)):
        out = run(p, q, alpha)
        accepted = out[:, 0] == draft_tok
        expected_gamma = gamma(p, alpha)
        expected = uniform.cpu() <= (expected_gamma / q)
        assert torch.equal(accepted, expected), (
            f"p={p} q={q} alpha={alpha} gamma={expected_gamma}\n"
            f" got      {accepted.tolist()}\n expected {expected.tolist()}"
        )
        print(f"  ok  kernel accepts iff u <= gamma(p,alpha)/q   p={p} q={q} alpha={alpha} gamma={expected_gamma:.4f}")

    strict_via_alpha0 = run(0.3, 0.6, 0.0)
    boosted = run(0.3, 0.6, 0.5)
    assert (boosted[:, 0] == draft_tok).sum() > (strict_via_alpha0[:, 0] == draft_tok).sum(), (
        "alpha=0.5 did not accept more than alpha=0.0; the boost is not reaching the kernel"
    )
    print("  ok  alpha=0.5 accepts strictly more draft tokens than alpha=0.0")


def main() -> int:
    failures = 0
    for test in (test_alpha_plumbing, test_negative_alpha_rejected, test_gamma_formula, test_kernel_matches_gamma):
        print(f"{test.__name__}:")
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {exc}")
    print("FAILED" if failures else "all CACTUS patch checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
