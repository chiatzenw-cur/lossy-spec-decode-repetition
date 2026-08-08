#!/usr/bin/env python3
"""Survivorship-bias-corrected within-loop trajectory: restrict to the
balanced cohort of loops reaching >=5 cycles (n=464/689), then report each
loop's own per-cycle delta from its cycle-0 value (equal loop weight, not
equal token weight), with a bootstrap 95% CI, for p/entropy/shortfall/
strict_would_accept. Naive cross-cycle averaging over ALL loops conflates
"long loops are inherently more confident" with genuine within-loop
entrenchment, since cycle 0 has all 689 loops but cycle 14 only has the
loops that survived that long; this cohort removes that confound by
construction. Cross-check against analyze_loop_fixed_effects*.py, a fully
independent method (within-loop demeaning regression) -- they agree closely.

Run from repo root, after collect_loop_token_metrics.py.
"""
import json, statistics, random
from collections import defaultdict

random.seed(0)
rows = [json.loads(l) for l in open("data/loop_token_metrics/all_relaxed_greedy_loops.jsonl")]

def mean(vals):
    v = [x for x in vals if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return statistics.mean(v) if v else None

# Per-loop, per-cycle mean p/entropy/shortfall/strict_ok, using ONLY loops with
# >=5 complete cycles (0..4), so every loop in the cohort contributes to every
# cycle bucket -- no survivorship bias.
MIN_CYCLES = 5
cohort = []
for r in rows:
    period = max(1, r["match_length_tokens"])
    start = r["loop_token_start"]
    by_cycle = defaultdict(list)
    for t in r["loop_tokens"]:
        cycle = (t["token_index"] - start) // period
        by_cycle[cycle].append(t)
    max_cycle = max(by_cycle.keys())
    if max_cycle < MIN_CYCLES - 1:
        continue  # doesn't reach cycle 4
    cohort.append((r, by_cycle))

print(f"balanced cohort: {len(cohort)}/{len(rows)} loops reach >= {MIN_CYCLES} cycles")

def cycle_mean(by_cycle, cycle, field):
    toks = by_cycle.get(cycle, [])
    if field == "strict_ok":
        vals = [1.0 if t.get("strict_would_accept") else 0.0 for t in toks if t.get("strict_would_accept") is not None]
    else:
        vals = [t.get(field) for t in toks if t.get(field) is not None]
    return mean(vals)

def bootstrap_ci(deltas, n_boot=5000):
    if not deltas:
        return None, None
    n = len(deltas)
    boot_medians = []
    for _ in range(n_boot):
        sample = [deltas[random.randrange(n)] for _ in range(n)]
        boot_medians.append(statistics.median(sample))
    boot_medians.sort()
    lo = boot_medians[int(0.025 * n_boot)]
    hi = boot_medians[int(0.975 * n_boot)]
    return lo, hi

for field, label in [("p", "p (target prob)"), ("target_entropy", "entropy"),
                      ("target_top1_shortfall", "shortfall"), ("strict_ok", "strict_would_accept")]:
    print(f"\n=== {label}: per-loop Delta from cycle 0 (n={len(cohort)} loops, equal loop weight) ===")
    for cycle in range(1, MIN_CYCLES):
        deltas = []
        for r, by_cycle in cohort:
            c0 = cycle_mean(by_cycle, 0, field)
            cj = cycle_mean(by_cycle, cycle, field)
            if c0 is not None and cj is not None:
                deltas.append(cj - c0)
        if not deltas:
            continue
        med = statistics.median(deltas)
        lo, hi = bootstrap_ci(deltas)
        print(f"  cycle {cycle}: median_delta={med:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]  n={len(deltas)}")
