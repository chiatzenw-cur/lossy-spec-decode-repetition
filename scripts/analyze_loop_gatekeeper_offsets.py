#!/usr/bin/env python3
"""For each confirmed-abnormal loop, bucket in-loop tokens by their offset
within the repeat period (position mod period) and look for a single offset
with a much higher lossy_only_accepted rate than the rest -- a "gatekeeper"
position: the specific word/token in the repeated phrase that needs
relaxation to get through, without which the loop might not self-sustain.

Finding: 79% of loops have a >=3x peaked single-offset gatekeeper, and 99% of
loops contain at least one lossy-only-accepted token -- but only ~11% of
individual in-loop tokens are lossy-only-accepted (89% are strict-compatible
too). Relaxation's causal role looks localized to specific positions, not
uniform sustenance across the whole loop.

Run from repo root, after collect_loop_token_metrics.py.
"""
import json, statistics
from collections import defaultdict

rows = [json.loads(l) for l in open("data/loop_token_metrics/all_relaxed_greedy_loops.jsonl")]

# For each loop, bucket tokens by their offset within the repeat period
# (position mod period). If one or two specific offsets show a much higher
# lossy_only_accepted rate than the rest, those are candidate "gatekeeper"
# positions -- the specific word/token in the repeated phrase that requires
# relaxation to get through, without which the loop might not self-sustain.
by_offset_all = defaultdict(list)  # offset -> [lossy_only bool, ...] pooled across loops (offset normalized 0..1 bucket, since periods differ)
per_loop_offset_rates = []  # (loop_id, [rate per offset within THIS loop's own period])

for loop_id, r in enumerate(rows):
    period = max(1, r["match_length_tokens"])
    start = r["loop_token_start"]
    by_offset = defaultdict(list)
    for t in r["loop_tokens"]:
        if t.get("emission_source") != "accepted_draft":
            continue
        offset = (t["token_index"] - start) % period
        lo = t.get("lossy_only_accepted")
        if lo is not None:
            by_offset[offset].append(1.0 if lo else 0.0)
    if len(by_offset) < 2:
        continue
    rates = {off: statistics.mean(v) for off, v in by_offset.items() if v}
    if not rates:
        continue
    max_off = max(rates, key=rates.get)
    per_loop_offset_rates.append((loop_id, period, rates, max_off, rates[max_off]))

# Does the MAX-rate offset concentrate on a small number of positions, or is
# it uniform (no gatekeeper)? Compare each loop's max-offset rate to its own
# mean rate across all offsets.
ratios = []
for loop_id, period, rates, max_off, max_rate in per_loop_offset_rates:
    mean_rate = statistics.mean(rates.values())
    if mean_rate > 0:
        ratios.append(max_rate / mean_rate)
    elif max_rate > 0:
        ratios.append(float("inf"))

finite_ratios = [r for r in ratios if r != float("inf")]
print(f"loops with >=2 offsets and >=1 lossy-only-accepted token: {len(per_loop_offset_rates)}")
print(f"loops where a single offset had 100%+ higher lossy_only rate than uniform baseline: "
      f"{sum(1 for r in ratios if r >= 2.0)}/{len(ratios)}")
if finite_ratios:
    print(f"median (max-offset rate / mean-offset rate): {statistics.median(finite_ratios):.2f}")
    print(f"share of loops with a peaked (>=3x) single-offset gatekeeper: "
          f"{sum(1 for r in finite_ratios if r >= 3.0)/len(finite_ratios):.3f}")

# overall: what fraction of loops have ANY lossy-only-accepted token at all vs none
n_any_lossy_only = sum(1 for r in rows if any(
    t.get("lossy_only_accepted") is True for t in r["loop_tokens"] if t.get("emission_source") == "accepted_draft"
))
print(f"\nloops with >=1 lossy-only-accepted token anywhere in the loop: {n_any_lossy_only}/{len(rows)} ({n_any_lossy_only/len(rows):.3f})")
n_zero_lossy_only = len(rows) - n_any_lossy_only
print(f"loops with ZERO lossy-only-accepted tokens (would have looped under STRICT decoding too): {n_zero_lossy_only}/{len(rows)} ({n_zero_lossy_only/len(rows):.3f})")
