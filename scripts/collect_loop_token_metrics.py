#!/usr/bin/env python3
"""For every LLM-confirmed abnormal adjacent loop, collect the full per-token
speculative-decoding metrics for: the token immediately BEFORE the loop (the
"onset" context token), every token INSIDE the loop (loop_token_start through
loop_token_end inclusive), and the token immediately AFTER the loop (the
"escape" token -- null if the trace ends before one exists, e.g. the run was
truncated by its token budget while still mid-loop).

Consumes judged rows from judge_adjacent_loops.py (one JSONL per runs_root),
each of which already carries an EXACT token position -- no re-resolution
happens here either; position and metrics are joined purely by token_index.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lib_trace_align import align  # noqa: E402

NUMERIC_METRICS = (
    "p", "q", "p_over_q", "u", "target_rank", "target_top1_prob", "target_top1_shortfall",
    "target_entropy", "draft_entropy", "kl_target_draft", "kl_draft_target", "tv_distance",
    "consecutive_accepted_length",
    # The EMITTED token's own p/rank/shortfall -- only populated on rows
    # collected after the tracer fix, and only non-null when accepted is
    # False (draft != emitted). Null (not absent) on older data and on
    # accepted_draft rows either way; see patches/lenience_trace.py.
    "emitted_p", "emitted_target_rank", "emitted_top1_shortfall",
)
BOOLEAN_METRICS = ("strict_would_accept", "lossy_would_accept", "actually_accepted", "lossy_only_accepted")

# target_top1_shortfall was renamed from target_top1_margin; older
# proposals.jsonl files predate the rename and still use the old key. Same
# quantity (top1_prob - p(x)) either name.
_LEGACY_METRIC_NAMES = {"target_top1_shortfall": "target_top1_margin"}


def get_metric(rec: dict, name: str) -> Any:
    if name in rec:
        return rec.get(name)
    legacy = _LEGACY_METRIC_NAMES.get(name)
    return rec.get(legacy) if legacy else None


def token_snapshot(records: list[dict], token_index: int) -> dict[str, Any] | None:
    idx = token_index - 1
    if not (0 <= idx < len(records)):
        return None
    rec = records[idx]
    out: dict[str, Any] = {"token_index": token_index, "text": rec.get("text")}
    for name in NUMERIC_METRICS:
        out[name] = get_metric(rec, name)
    for name in BOOLEAN_METRICS:
        out[name] = rec.get(name)
    out["emission_source"] = rec.get("emission_source")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("judged_jsonl", nargs="+", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    run_cache: dict[tuple[str, str, str], tuple[bytes, list[dict]] | None] = {}
    out_rows = []
    skipped = 0

    for jpath in args.judged_jsonl:
        rows = [json.loads(l) for l in jpath.open()]
        for row in rows:
            if row.get("verdict") != "abnormal":
                continue
            runs_root = pathlib.Path(row["runs_root"])
            key = (str(runs_root), row["case"], row["tag"])
            if key not in run_cache:
                run_dir = runs_root / row["case"] / "seed_0" / row["tag"]
                raw, records = align(run_dir)
                run_cache[key] = records
            records = run_cache[key]
            if records is None:
                skipped += 1
                continue

            # context_before_origin: before the pattern is said even once.
            context_before_origin = token_snapshot(records, row["loop_token_start"] - 1)
            # pre_repeat_boundary: the LAST token of the origin (first) occurrence,
            # i.e. immediately before the model decides to repeat rather than
            # continue -- this is the actually-diagnostic boundary, not
            # context_before_origin (which can sit tens of tokens earlier).
            pre_repeat_boundary = token_snapshot(records, row["ignition_token_index"] - 1)
            loop_tokens = []
            for ti in range(row["loop_token_start"], row["loop_token_end"] + 1):
                snap = token_snapshot(records, ti)
                if snap is not None:
                    loop_tokens.append(snap)
            escape = token_snapshot(records, row["loop_token_end"] + 1)

            out_rows.append({
                "benchmark": row["benchmark"], "case": row["case"], "tag": row["tag"],
                "loop_token_start": row["loop_token_start"], "loop_token_end": row["loop_token_end"],
                "ignition_token_index": row["ignition_token_index"],
                "chain_repeat_count": row["chain_repeat_count"],
                "match_length_tokens": row["match_length_tokens"],
                "total_span_tokens": row["total_span_tokens"],
                "category": row.get("category"),
                "reasoning": row.get("reasoning"),
                "context_before_origin_token": context_before_origin,
                "pre_repeat_boundary_token": pre_repeat_boundary,
                "loop_tokens": loop_tokens,
                "escape_token": escape,  # null iff the trace ends before an escape token exists
                "escaped_within_trace": escape is not None,
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(out_rows)} loops with full token metrics to {args.out} ({skipped} skipped: run unreadable)")
    n_escaped = sum(1 for r in out_rows if r["escaped_within_trace"])
    print(f"escaped within trace: {n_escaped}/{len(out_rows)}  (rest: loop still running when trace ends, e.g. budget cutoff)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
