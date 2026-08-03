#!/usr/bin/env python3
"""Check the per-proposal tracer records the right counterfactual.

Drives the tracer directly with hand-built tensors, so strict/lossy/lossy-only
are predictable in closed form. No model, no server.
"""
from __future__ import annotations
import json, os, pathlib, sys, tempfile

def main() -> int:
    import torch
    dest = pathlib.Path(tempfile.mkdtemp()) / "trace.jsonl"
    pathlib.Path(f"/tmp/lossy-spec-decode-trace-{os.getuid()}").write_text(str(dest))
    from vllm.v1.sample import lenience_trace
    tracer = lenience_trace._Tracer()
    assert tracer.enabled, "tracer did not pick up the destination file"

    V, DT = 8, 3
    lam = 0.2
    # one request, 3 draft positions with ratios 0.9 / 0.3 / 0.05
    ratios = [0.9, 0.3, 0.05]
    n = len(ratios)
    draft_probs = torch.full((n, V), 0.5 / (V - 1)); draft_probs[:, DT] = 0.5
    target_probs = torch.zeros(n, V)
    for i, r in enumerate(ratios):
        target_probs[i] = (1.0 - 0.5 * r) / (V - 1); target_probs[i, DT] = 0.5 * r
    u = torch.tensor([0.5, 0.5, 0.5])           # fixed draw
    # strict: accept iff ratio >= u -> [T, F, F]
    # lossy(0.2): accept iff ratio/0.2 >= u -> [T, T, F]  => pos1 is lossy-only
    out = torch.tensor([[DT, DT, 99, -1]], dtype=torch.int32)  # first two accepted, third rejected->recovered
    tracer.record(
        draft_token_ids=torch.full((n,), DT, dtype=torch.int32),
        draft_probs=draft_probs, target_probs=target_probs, uniform_probs=u,
        recovered_token_ids=torch.full((n,), 99, dtype=torch.int32),
        bonus_token_ids=torch.tensor([[7]], dtype=torch.int32),
        output_token_ids=out,
        cu_num_draft_tokens=torch.tensor([n], dtype=torch.int32),
        num_draft_tokens=[n], lenience_factor=lam,
    )
    tracer.close()
    rows = [json.loads(l) for l in dest.read_text().splitlines()]
    assert len(rows) == 3, f"expected 3 rows, got {len(rows)}"
    exp_strict = [True, False, False]
    exp_lossy  = [True, True, False]
    exp_only   = [False, True, False]
    exp_src    = ["accepted_draft", "accepted_draft", "recovered"]
    ok = True
    for i, r in enumerate(rows):
        for field, want in (("strict_would_accept", exp_strict[i]),
                            ("lossy_would_accept", exp_lossy[i]),
                            ("lossy_only_accepted", exp_only[i]),
                            ("emission_source", exp_src[i]),
                            ("output_position", i)):
            if r[field] != want:
                print(f"  FAIL row {i} {field}: got {r[field]!r} want {want!r}"); ok = False
    # distribution features: q is uniform-ish outside the draft token, p is
    # sharper, so KL(p||q) and KL(q||p) must both be finite and positive, and
    # TV must lie in [0,1]. Checked rather than eyeballed because a silent NaN
    # here would poison every downstream feature.
    import math
    for i, r in enumerate(rows):
        for f in ("draft_entropy", "kl_target_draft", "kl_draft_target", "tv_distance"):
            v = r.get(f)
            if v is None or not math.isfinite(v):
                print(f"  FAIL row {i} {f} = {v!r}"); ok = False
        if not (0.0 <= r["tv_distance"] <= 1.0):
            print(f"  FAIL row {i} tv_distance out of range: {r['tv_distance']}"); ok = False
        if r["kl_target_draft"] < 0 or r["kl_draft_target"] < 0:
            print(f"  FAIL row {i} negative KL"); ok = False
    if ok:
        print("  ok    strict/lossy/lossy-only/emission_source/output_position all correct")
        print(f"  ok    dist feats row0: H(q)={rows[0]['draft_entropy']:.3f} "
              f"H(p)={rows[0]['target_entropy']:.3f} "
              f"KL(p||q)={rows[0]['kl_target_draft']:.3f} "
              f"KL(q||p)={rows[0]['kl_draft_target']:.3f} TV={rows[0]['tv_distance']:.3f}")
        print(f"  ok    row0 p={rows[0]['p']:.4f} q={rows[0]['q']:.4f} ratio={rows[0]['p_over_q']:.3f} rank={rows[0]['target_rank']}")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
