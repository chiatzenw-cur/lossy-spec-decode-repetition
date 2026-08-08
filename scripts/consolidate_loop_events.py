#!/usr/bin/env python3
"""Turn extract_repetition_clusters.py's shingle-seed cluster rows into
candidate LOOP EVENTS: group by cluster_id, chain consecutive
gap_tokens_since_previous==0 occurrences starting from the origin, filter to
chain_repeat_count>=3 and total_span_tokens>=15, then interval-merge
overlapping events per case (keeping the larger-span one).

This is the purely algorithmic step -- no LLM involved -- that produces exact
token positions for judge_adjacent_loops.py to classify as abnormal/legitimate
without ever asking the judge model for a position itself.

Same 8-group "truly relaxed greedy" list used throughout this investigation
(collect_loop_baseline_metrics.py, recollect_with_fixed_tracer.sh): AIME24 +
HumanEval lenience-radical greedy, and 6 LongBench-v2 EAGLE3/P-EAGLE x
lenience/CACTUS combinations on case_019 and case_079.

Run from repo root, after extract_repetition_clusters.py has produced
clusters.jsonl for each group. Output:
data/loop_token_metrics/loop_candidate_events_pre_judge.json
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

GROUPS = [
    ("aime24", "runs/aime24_fresh", "lenience0p05Greedy12k",
     "data/repetition_clusters/lenience0p05Greedy12k/seed_0/context_040/all_cases/clusters.jsonl"),
    ("humaneval", "runs/humaneval_fresh", "lenience0p05Greedy9k",
     "data/repetition_clusters/lenience0p05Greedy9k/seed_0/context_040/all_cases/clusters.jsonl"),
    ("longbench_v2", "runs/longbench_v2_fresh", "cactus2Greedy",
     "data/repetition_clusters/cactus2Greedy/seed_0/context_040/case_019/clusters.jsonl"),
    ("longbench_v2", "runs/longbench_v2_fresh", "cactus2PEagleGreedy",
     "data/repetition_clusters/cactus2PEagleGreedy/seed_0/context_040/case_019/clusters.jsonl"),
    ("longbench_v2", "runs/longbench_v2_fresh", "lenience0p002Greedy",
     "data/repetition_clusters/lenience0p002Greedy/seed_0/context_040/case_019/clusters.jsonl"),
    ("longbench_v2", "runs/longbench_v2_fresh", "lenience0p002PEagleGreedy",
     "data/repetition_clusters/lenience0p002PEagleGreedy/seed_0/context_040/case_019/clusters.jsonl"),
    ("longbench_v2", "runs/longbench_v2_fresh", "lenience0p2GreedyLongBudget",
     "data/repetition_clusters/lenience0p2GreedyLongBudget/seed_0/context_040/case_079/clusters.jsonl"),
    ("longbench_v2", "runs/longbench_v2_fresh", "lenience0p2PEagleGreedy",
     "data/repetition_clusters/lenience0p2PEagleGreedy/seed_0/context_040/case_079/clusters.jsonl"),
]

MIN_CHAIN = 3
MIN_SPAN = 15
OUT = pathlib.Path("data/loop_token_metrics/loop_candidate_events_pre_judge.json")


def main() -> int:
    all_events = []
    for benchmark, runs_root, tag, clusters_path in GROUPS:
        path = pathlib.Path(clusters_path)
        if not path.is_file():
            print(f"MISSING: {clusters_path}", file=sys.stderr)
            continue
        rows = [json.loads(l) for l in path.open()]
        by_cluster = defaultdict(list)
        for r in rows:
            by_cluster[r["cluster_id"]].append(r)
        for cid in by_cluster:
            by_cluster[cid].sort(key=lambda r: r["occurrence_index"])

        events = []
        for cid, occs in by_cluster.items():
            origin_start = occs[0]["origin_token_start"]
            origin_end = occs[0]["origin_token_end"]
            chain_end = origin_end
            chain_count = 1
            ignition_token = None
            match_len = occs[0]["match_length_tokens"]
            for occ in occs:
                if occ["gap_tokens_since_previous"] == 0:
                    if ignition_token is None:
                        ignition_token = occ["recurrence_token_start"]
                    chain_end = occ["recurrence_token_end"]
                    chain_count += 1
                else:
                    break
            if ignition_token is None:
                continue
            total_span = chain_end - origin_start + 1
            if chain_count < MIN_CHAIN or total_span < MIN_SPAN:
                continue
            events.append({
                "benchmark": benchmark, "runs_root": runs_root, "tag": tag,
                "case": occs[0]["case"], "match_length_tokens": match_len,
                "chain_repeat_count": chain_count,
                "loop_token_start": origin_start, "loop_token_end": chain_end,
                "ignition_token_index": ignition_token,
                "total_span_tokens": total_span,
                "origin_match_text": occs[0]["origin_match_text"],
            })

        # interval-merge overlaps per case
        by_case = defaultdict(list)
        for e in events:
            by_case[e["case"]].append(e)
        merged = []
        for case, evs in by_case.items():
            evs.sort(key=lambda e: (e["loop_token_start"], -e["total_span_tokens"]))
            m = []
            for e in evs:
                if m and e["loop_token_start"] <= m[-1]["loop_token_end"]:
                    if e["total_span_tokens"] > m[-1]["total_span_tokens"]:
                        m[-1] = e
                    else:
                        m[-1]["loop_token_end"] = max(m[-1]["loop_token_end"], e["loop_token_end"])
                else:
                    m.append(dict(e))
            merged.extend(m)

        print(f"{benchmark}/{tag}: {len(rows)} raw rows -> {len(events)} events (chain>=3,span>=15) -> {len(merged)} after merge")
        all_events.extend(merged)

    print(f"\nTOTAL events to judge: {len(all_events)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(all_events, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
