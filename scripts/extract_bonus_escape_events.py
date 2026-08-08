#!/usr/bin/env python3
"""For every confirmed-abnormal loop whose escape token was bonus-emitted
(sampled directly/stochastically from the target with zero verification),
extract the exact prefix (everything generated up to but not including the
escape token) plus the identity of two tokens: the "expected" periodic-
continuation token (whatever appeared exactly one period earlier in the same
output) and the "actual" token the bonus draw emitted.

This is the input to replay_bonus_escape_argmax.py, which asks the target
model alone (temperature=0, no drafter) what its own argmax was at that exact
position -- the question this whole pipeline exists to answer: was the escape
a genuine stochastic override of an entrenched target (rank_expected==0,
rank_actual>0), or had the target's own top pick already moved off the loop
before the sample was even drawn (rank_expected>0, rank_actual==0)?

Run from repo root. Output: data/loop_token_metrics/bonus_escape_events.jsonl
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, "scripts")
from lib_trace_align import align  # noqa: E402

LOOPS = pathlib.Path("data/loop_token_metrics/all_relaxed_greedy_loops.jsonl")
JUDGED = [
    "data/adjacent_loop_judgements/aime24_lenience0p05Greedy12k_full.jsonl",
    "data/adjacent_loop_judgements/humaneval_lenience0p05Greedy9k_full.jsonl",
    "data/adjacent_loop_judgements/longbench_v2_greedy_relaxed_full.jsonl",
]
OUT = pathlib.Path("data/loop_token_metrics/bonus_escape_events.jsonl")


def main() -> int:
    rows = [json.loads(l) for l in LOOPS.open()]

    # runs_root isn't stored per-loop in LOOPS (it's in the judged jsonl, not
    # the collected one) -- rebuild the mapping from the three judged files.
    runs_root_by_key: dict[tuple, str] = {}
    for jf in JUDGED:
        for line in open(jf):
            r = json.loads(line)
            runs_root_by_key[(r["case"], r["tag"], r["loop_token_start"], r["loop_token_end"])] = r["runs_root"]

    events = []
    run_cache: dict[tuple, tuple] = {}
    missing_runs_root = 0
    for r in rows:
        esc = r.get("escape_token")
        if esc is None or esc.get("emission_source") != "bonus":
            continue
        key = (r["case"], r["tag"], r["loop_token_start"], r["loop_token_end"])
        runs_root = runs_root_by_key.get(key)
        if runs_root is None:
            missing_runs_root += 1
            continue
        run_key = (runs_root, r["case"], r["tag"])
        if run_key not in run_cache:
            run_dir = pathlib.Path(runs_root) / r["case"] / "seed_0" / r["tag"]
            raw, records = align(run_dir)
            run_cache[run_key] = (raw, records)
        raw, records = run_cache[run_key]
        if records is None:
            continue

        escape_token_index = r["loop_token_end"] + 1
        period = max(1, r["match_length_tokens"])
        expected_index = escape_token_index - period
        if expected_index < 1 or expected_index > len(records):
            continue

        esc_rec = records[escape_token_index - 1]
        exp_rec = records[expected_index - 1]
        if esc_rec.get("token_index") != escape_token_index or exp_rec.get("token_index") != expected_index:
            continue

        actual_token_id = esc_rec.get("emitted_token_id")
        expected_token_id = exp_rec.get("emitted_token_id")
        prefix_bytes = raw[: esc_rec["byte_start"]]

        events.append({
            "case": r["case"], "tag": r["tag"], "runs_root": runs_root,
            "escape_token_index": escape_token_index, "period": period,
            "actual_token_id": actual_token_id, "actual_text": esc_rec.get("text"),
            "expected_token_id": expected_token_id, "expected_text": exp_rec.get("text"),
            "same_token": actual_token_id == expected_token_id,
            # hex, not base64, to keep it plain-ASCII-safe in JSON
            "prefix_b64": prefix_bytes.hex(),
        })

    print(f"total bonus-escape events: {len(events)}  (missing_runs_root skipped: {missing_runs_root})")
    same = sum(1 for e in events if e["same_token"])
    print(f"same_token (boundary artifact, pattern actually continued): {same}")
    print(f"genuine bonus escapes to replay: {len(events) - same}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
