#!/usr/bin/env python3
"""Clean baseline: the same per-token metrics as collect_loop_token_metrics.py,
averaged over every token in the same runs EXCLUDING anything inside a
confirmed-abnormal loop span (loop_token_start..loop_token_end inclusive).

This is the population the pre-onset/ignition/escape numbers should be read
against -- without it, "escape tokens average target_rank 150" tells you
nothing on its own.

Reads the same judged JSONL files as collect_loop_token_metrics.py (for the
exclusion set) plus a run manifest (case list per runs_root/tag) so it can
cover cases that had ZERO loops too -- those contribute pure baseline tokens.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from collections import defaultdict
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lib_trace_align import align  # noqa: E402

NUMERIC_METRICS = (
    "p", "q", "p_over_q", "u", "target_rank", "target_top1_prob", "target_top1_shortfall",
    "target_entropy", "draft_entropy", "kl_target_draft", "kl_draft_target", "tv_distance",
    "consecutive_accepted_length",
)
BOOLEAN_METRICS = ("strict_would_accept", "lossy_would_accept", "actually_accepted", "lossy_only_accepted")

# (benchmark label, runs_root, tag, cases -- None means "every case_* under runs_root")
GROUPS: list[tuple[str, str, str, list[str] | None]] = [
    ("aime24", "runs/aime24_fresh", "lenience0p05Greedy12k", None),
    ("humaneval", "runs/humaneval_fresh", "lenience0p05Greedy9k", None),
    ("longbench_v2", "runs/longbench_v2_fresh", "cactus2Greedy", ["case_019"]),
    ("longbench_v2", "runs/longbench_v2_fresh", "cactus2PEagleGreedy", ["case_019"]),
    ("longbench_v2", "runs/longbench_v2_fresh", "lenience0p002Greedy", ["case_019"]),
    ("longbench_v2", "runs/longbench_v2_fresh", "lenience0p002PEagleGreedy", ["case_019"]),
    ("longbench_v2", "runs/longbench_v2_fresh", "lenience0p2GreedyLongBudget", ["case_079"]),
    ("longbench_v2", "runs/longbench_v2_fresh", "lenience0p2PEagleGreedy", ["case_079"]),
]


def mean(values: list[Any]) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return statistics.mean(nums) if nums else None


# target_top1_shortfall was renamed from target_top1_margin; older
# proposals.jsonl files predate the rename and still use the old key. Same
# quantity (top1_prob - p(x)) either name.
_LEGACY_METRIC_NAMES = {"target_top1_shortfall": "target_top1_margin"}


def get_metric(rec: dict, name: str) -> Any:
    if name in rec:
        return rec.get(name)
    legacy = _LEGACY_METRIC_NAMES.get(name)
    return rec.get(legacy) if legacy else None


def summarize(records: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_tokens": len(records)}
    for name in NUMERIC_METRICS:
        out[f"mean_{name}"] = mean([get_metric(r, name) for r in records])
    for name in BOOLEAN_METRICS:
        bools = [r.get(name) for r in records if isinstance(r.get(name), bool)]
        out[f"share_{name}"] = (sum(bools) / len(bools)) if bools else None
    bonus = [r.get("emission_source") == "bonus" for r in records]
    out["share_bonus"] = (sum(bonus) / len(bonus)) if bonus else None
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("judged_jsonl", nargs="+", type=pathlib.Path, help="Judged loop files, for the exclusion set.")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    # Build exclusion set: (runs_root, case, tag, token_index) for every token
    # inside a confirmed-abnormal loop.
    excluded: set[tuple[str, str, str, int]] = set()
    for jpath in args.judged_jsonl:
        for line in jpath.open():
            row = json.loads(line)
            if row.get("verdict") != "abnormal":
                continue
            key_base = (row["runs_root"], row["case"], row["tag"])
            for ti in range(row["loop_token_start"], row["loop_token_end"] + 1):
                excluded.add(key_base + (ti,))

    per_benchmark: dict[str, list[dict]] = defaultdict(list)
    per_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    overall: list[dict] = []

    for benchmark, runs_root_str, tag, case_filter in GROUPS:
        runs_root = pathlib.Path(runs_root_str)
        if case_filter is None:
            cases = sorted(p.name for p in runs_root.glob("case_*") if (p / "seed_0" / tag).is_dir())
        else:
            cases = case_filter

        for case in cases:
            run_dir = runs_root / case / "seed_0" / tag
            raw, records = align(run_dir)
            if records is None:
                print(f"warning: unreadable {run_dir}", file=sys.stderr)
                continue
            key_base = (runs_root_str, case, tag)
            kept = [r for r in records if (key_base + (r.get("token_index"),)) not in excluded]
            per_benchmark[benchmark].extend(kept)
            per_group[(runs_root_str, tag)].extend(kept)
            overall.extend(kept)

    result = {
        "overall": summarize(overall),
        "by_benchmark": {b: summarize(recs) for b, recs in per_benchmark.items()},
        "by_group": {f"{root}::{tag}": summarize(recs) for (root, tag), recs in per_group.items()},
        "excluded_loop_tokens": len(excluded),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"wrote baseline to {args.out}")
    print(f"overall n_tokens={result['overall']['n_tokens']}  excluded_loop_tokens={len(excluded)}")
    for b, s in result["by_benchmark"].items():
        print(f"  {b}: n={s['n_tokens']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
