#!/usr/bin/env python3
"""Rank distribution of in-loop tokens under the target's own distribution,
split by emission_source (accepted_draft vs recovered; bonus rows carry no
p/q/rank at all -- see the tracer's bonus-append block, a known gap), plus
lossy_only_accepted vs strict-compatible accepted_draft tokens, plus a clean
non-loop baseline for comparison. Resolves the "mean_p=0.889 vs mean_rank=13.7"
mixture appearance: it's not one population, it's ~99% near-rank-0
accepted_draft tokens diluted by a small, genuinely deep-tail recovered tail.

Run from repo root, after collect_loop_token_metrics.py /
collect_loop_baseline_metrics.py have produced their JSONL/JSON.
"""
import json, statistics
from collections import defaultdict

rows = [json.loads(l) for l in open("data/loop_token_metrics/all_relaxed_greedy_loops.jsonl")]
all_loop_tokens = [t for r in rows for t in r["loop_tokens"]]

def pctile(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)

def report(label, toks):
    ranks = [t["target_rank"] for t in toks if t.get("target_rank") is not None]
    if not ranks:
        print(f"  {label:28s} n=0")
        return
    n = len(ranks)
    print(f"  {label:28s} n={n:6d}  median={statistics.median(ranks):6.1f}  "
          f"P(rank=0)={sum(1 for r in ranks if r==0)/n:5.3f}  "
          f"P(rank<=1)={sum(1 for r in ranks if r<=1)/n:5.3f}  "
          f"P(rank<=5)={sum(1 for r in ranks if r<=5)/n:5.3f}  "
          f"P(rank>100)={sum(1 for r in ranks if r>100)/n:5.3f}  "
          f"P90={pctile(ranks,0.9):7.1f}  P99={pctile(ranks,0.99):7.1f}  mean={statistics.mean(ranks):7.2f}")

print("=== ALL in-loop tokens, by emission_source ===")
by_source = defaultdict(list)
for t in all_loop_tokens:
    by_source[t.get("emission_source")].append(t)
report("ALL", all_loop_tokens)
for source, toks in sorted(by_source.items(), key=lambda kv: -len(kv[1])):
    report(f"emission_source={source}", toks)

print("\n=== stratified by lossy_only_accepted (accepted_draft tokens only) ===")
draft_toks = by_source.get("accepted_draft", [])
lossy_only = [t for t in draft_toks if t.get("lossy_only_accepted") is True]
strict_ok = [t for t in draft_toks if t.get("lossy_only_accepted") is False]
report("accepted_draft, lossy_only_accepted=True", lossy_only)
report("accepted_draft, lossy_only_accepted=False (strict-ok too)", strict_ok)

print("\n=== for comparison: baseline (non-loop) tokens ===")
# quick reload of baseline via same run set, non-loop tokens, split by source
import sys
sys.path.insert(0, "scripts")
from lib_trace_align import align
GROUPS = [
    ("runs/aime24_fresh", "lenience0p05Greedy12k", None),
    ("runs/humaneval_fresh", "lenience0p05Greedy9k", None),
]
import pathlib
excluded = set()
for jf in ["data/adjacent_loop_judgements/aime24_lenience0p05Greedy12k_full.jsonl",
           "data/adjacent_loop_judgements/humaneval_lenience0p05Greedy9k_full.jsonl",
           "data/adjacent_loop_judgements/longbench_v2_greedy_relaxed_full.jsonl"]:
    for line in open(jf):
        r = json.loads(line)
        if r.get("verdict") != "abnormal":
            continue
        for ti in range(r["loop_token_start"], r["loop_token_end"]+1):
            excluded.add((r["runs_root"], r["case"], r["tag"], ti))

baseline_toks = []
for root, tag, _ in GROUPS:
    rp = pathlib.Path(root)
    for case_dir in sorted(rp.glob("case_*")):
        run_dir = case_dir / "seed_0" / tag
        if not run_dir.is_dir():
            continue
        raw, records = align(run_dir)
        if records is None:
            continue
        for rec in records:
            if (root, case_dir.name, tag, rec.get("token_index")) not in excluded:
                baseline_toks.append(rec)
report("baseline (aime24+humaneval, non-loop)", baseline_toks)
by_source_baseline = defaultdict(list)
for t in baseline_toks:
    by_source_baseline[t.get("emission_source")].append(t)
for source, toks in sorted(by_source_baseline.items(), key=lambda kv: -len(kv[1])):
    report(f"baseline emission_source={source}", toks)
