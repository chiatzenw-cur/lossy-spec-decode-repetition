#!/usr/bin/env python3
"""Fixed-effects (within-loop demeaning) regression of p/entropy/shortfall/
strict_would_accept on cycle index, across the FULL cycle range (0-31) for
all 689 loops. Bias-robust alternative to naive per-cycle averaging: demean
both y and cycle-index within each loop, then OLS-slope the demeaned pair
(equivalent to LSDV without needing statsmodels, which isn't in this venv).
A full-range fit dilutes the sharp early-cycle rise with the long flat
plateau -- see analyze_loop_fixed_effects_early.py for the cycle 0-4
restriction that reconciles closely with the balanced-cohort bootstrap deltas.

Run from repo root, after collect_loop_token_metrics.py.
"""
import json, statistics
from collections import defaultdict
import numpy as np

rows = [json.loads(l) for l in open("data/loop_token_metrics/all_relaxed_greedy_loops.jsonl")]

def mean(vals):
    v = [x for x in vals if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return statistics.mean(v) if v else None

# Build per-(loop, cycle) panel: one row per loop*occurrence, y = cycle mean of
# the metric, j = cycle index, loop_id = which loop. Then fixed-effects via
# within-loop demeaning (equivalent to LSDV without needing statsmodels):
# demean y and j within each loop, OLS slope on the demeaned pair = the
# fixed-effects estimate, robust to between-loop heterogeneity (each loop's
# own baseline level is absorbed by u_i).
def fe_slope(field, use_strict=False):
    panel = []  # (loop_id, cycle, y)
    for loop_id, r in enumerate(rows):
        period = max(1, r["match_length_tokens"])
        start = r["loop_token_start"]
        by_cycle = defaultdict(list)
        for t in r["loop_tokens"]:
            cycle = (t["token_index"] - start) // period
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
    # keep only loops with >=2 distinct cycles (need within-loop variance to identify slope)
    by_loop = {k: v for k, v in by_loop.items() if len(v) >= 2}

    j_dm, y_dm = [], []
    for loop_id, obs in by_loop.items():
        js = np.array([o[0] for o in obs], dtype=float)
        ys = np.array([o[1] for o in obs], dtype=float)
        j_dm.append(js - js.mean())
        y_dm.append(ys - ys.mean())
    j_dm = np.concatenate(j_dm)
    y_dm = np.concatenate(y_dm)

    # OLS slope through origin (demeaned data has zero intercept by construction)
    beta1 = float((j_dm * y_dm).sum() / (j_dm * j_dm).sum())
    resid = y_dm - beta1 * j_dm
    n = len(j_dm)
    k_loops = len(by_loop)
    dof = n - k_loops - 1  # loop fixed effects consume k_loops-1 dof roughly; conservative
    sigma2 = float((resid**2).sum() / max(dof, 1))
    se = (sigma2 / float((j_dm**2).sum())) ** 0.5
    t_stat = beta1 / se if se > 0 else float("nan")
    return beta1, se, t_stat, n, k_loops

print(f"{'metric':30s} {'beta1 (slope/cycle)':>20s} {'SE':>10s} {'t-stat':>8s} {'n_obs':>8s} {'n_loops':>8s}")
for field, label, is_strict in [
    ("p", "p (target prob)", False),
    ("target_entropy", "entropy", False),
    ("target_top1_shortfall", "shortfall", False),
    (None, "strict_would_accept", True),
]:
    b1, se, t, n, k = fe_slope(field, use_strict=is_strict)
    print(f"  {label:28s} {b1:+20.5f} {se:10.5f} {t:8.2f} {n:8d} {k:8d}")
