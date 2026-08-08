#!/usr/bin/env python3
"""Same fixed-effects (within-loop demeaning) regression as
analyze_loop_fixed_effects.py, restricted to cycles 0-4 to match the
balanced-cohort window in analyze_loop_balanced_cohort.py. The full-range fit
dilutes the sharp early rise with the long flat plateau (loops run up to 31
cycles); this restriction is what actually reconciles with the balanced-
cohort bootstrap median deltas (e.g. p: +0.119 FE vs +0.107 balanced-cohort
at cycle 4).

Run from repo root, after collect_loop_token_metrics.py.
"""
import json, statistics
from collections import defaultdict
import numpy as np

rows = [json.loads(l) for l in open("data/loop_token_metrics/all_relaxed_greedy_loops.jsonl")]

def mean(vals):
    v = [x for x in vals if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return statistics.mean(v) if v else None

MAX_CYCLE = 4  # restrict to cycles 0..4, matching the balanced-cohort window

def fe_slope(field, use_strict=False):
    panel = []
    for loop_id, r in enumerate(rows):
        period = max(1, r["match_length_tokens"])
        start = r["loop_token_start"]
        by_cycle = defaultdict(list)
        for t in r["loop_tokens"]:
            cycle = (t["token_index"] - start) // period
            if cycle > MAX_CYCLE:
                continue
            by_cycle[cycle].append(t)
        for cycle, toks in by_cycle.items():
            if use_strict:
                vals = [1.0 if t.get("strict_would_accept") else 0.0 for t in toks if t.get("strict_would_accept") is not None]
            else:
                vals = [t.get(field) for t in toks if t.get(field) is not None]
            y = mean(vals)
            if y is not None:
                panel.append((loop_id, cycle, y))

    by_loop = defaultdict(list)
    for loop_id, cycle, y in panel:
        by_loop[loop_id].append((cycle, y))
    by_loop = {k: v for k, v in by_loop.items() if len(v) >= 2}

    j_dm, y_dm = [], []
    for loop_id, obs in by_loop.items():
        js = np.array([o[0] for o in obs], dtype=float)
        ys = np.array([o[1] for o in obs], dtype=float)
        j_dm.append(js - js.mean())
        y_dm.append(ys - ys.mean())
    j_dm = np.concatenate(j_dm)
    y_dm = np.concatenate(y_dm)

    beta1 = float((j_dm * y_dm).sum() / (j_dm * j_dm).sum())
    resid = y_dm - beta1 * j_dm
    n = len(j_dm)
    k_loops = len(by_loop)
    dof = n - k_loops - 1
    sigma2 = float((resid**2).sum() / max(dof, 1))
    se = (sigma2 / float((j_dm**2).sum())) ** 0.5
    t_stat = beta1 / se if se > 0 else float("nan")
    return beta1, se, t_stat, n, k_loops

print(f"Fixed-effects slope, restricted to cycles 0-{MAX_CYCLE} (comparable to the balanced-cohort window)")
print(f"{'metric':30s} {'beta1/cycle':>14s} {'x4 cycles':>12s} {'SE':>10s} {'t-stat':>8s} {'n_obs':>8s} {'n_loops':>8s}")
for field, label, is_strict in [
    ("p", "p (target prob)", False),
    ("target_entropy", "entropy", False),
    ("target_top1_shortfall", "shortfall", False),
    (None, "strict_would_accept", True),
]:
    b1, se, t, n, k = fe_slope(field, use_strict=is_strict)
    print(f"  {label:28s} {b1:+14.5f} {b1*4:+12.4f} {se:10.5f} {t:8.2f} {n:8d} {k:8d}")
