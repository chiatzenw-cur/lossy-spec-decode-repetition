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
    RECOVERED = 5  # a real in-vocab index (V=8), distinct from DT=3
    # strict: accept iff ratio >= u -> [T, F, F]
    # lossy(0.2): accept iff ratio/0.2 >= u -> [T, T, F]  => pos1 is lossy-only
    out = torch.tensor([[DT, DT, RECOVERED, -1]], dtype=torch.int32)  # first two accepted, third rejected->recovered
    tracer.record(
        draft_token_ids=torch.full((n,), DT, dtype=torch.int32),
        draft_probs=draft_probs, target_probs=target_probs, uniform_probs=u,
        recovered_token_ids=torch.full((n,), RECOVERED, dtype=torch.int32),
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
    # emitted_p/emitted_target_rank/emitted_top1_shortfall: null on accepted
    # rows (draft==emitted there, redundant), populated on the recovered row
    # with the RECOVERED token's own numbers -- not the rejected draft
    # proposal's. Row 2's recovered token (index 5) beats the draft's own
    # ratio=0.05 row, since target_probs there gives every non-DT index
    # (1-0.5*0.05)/7 ~= 0.1393 vs the draft token's 0.025, so it must NOT
    # equal row 2's own p (0.025) or match DT's rank (0).
    for i in (0, 1):
        for f in ("emitted_p", "emitted_target_rank", "emitted_top1_shortfall"):
            if rows[i][f] is not None:
                print(f"  FAIL row {i} {f} should be null on an accepted row, got {rows[i][f]!r}"); ok = False
    r2 = rows[2]
    if r2["emitted_p"] is None or abs(r2["emitted_p"] - r2["p"]) < 1e-9:
        print(f"  FAIL row 2 emitted_p should differ from the rejected draft proposal's own p={r2['p']!r}, got {r2['emitted_p']!r}"); ok = False
    if r2["emitted_target_rank"] is None or r2["emitted_target_rank"] == r2["target_rank"]:
        print(f"  FAIL row 2 emitted_target_rank should differ from the draft proposal's own rank={r2['target_rank']!r}, got {r2['emitted_target_rank']!r}"); ok = False
    if ok:
        print(f"  ok    emitted_p/rank on the recovered row describe the RECOVERED token "
              f"(p={r2['emitted_p']:.4f} rank={r2['emitted_target_rank']}), not the rejected "
              f"draft proposal (p={r2['p']:.4f} rank={r2['target_rank']})")
        print("  ok    strict/lossy/lossy-only/emission_source/output_position all correct")
        print(f"  ok    dist feats row0: H(q)={rows[0]['draft_entropy']:.3f} "
              f"H(p)={rows[0]['target_entropy']:.3f} "
              f"KL(p||q)={rows[0]['kl_target_draft']:.3f} "
              f"KL(q||p)={rows[0]['kl_draft_target']:.3f} TV={rows[0]['tv_distance']:.3f}")
        print(f"  ok    row0 p={rows[0]['p']:.4f} q={rows[0]['q']:.4f} ratio={rows[0]['p_over_q']:.3f} rank={rows[0]['target_rank']}")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
