#!/usr/bin/env python3
"""Append one manually-identified degradation onset to the manual label file.

This exists because two regex-based detectors (arithmetic evaluation,
near-miss value clustering) were checked against the actual text and were wrong
on 14/14 sampled hits -- reasoning text mixes algebra, bullet-point minus signs,
binary literals, truncated numbers and comma-separated lists in ways a lexical
matcher cannot tell apart from a genuinely false claim. Those detectors' output
lives in analysis/degradation_labels.jsonl, tagged untrusted. This file is
separate and is the ground truth: a human (or an LLM reading the text directly,
blind to the trace) decides where a degradation starts, and this script's only
job is turning that judgment into an exact, verified token position.

The quote must be an exact substring of output.txt. Requiring exactness -- not
a fuzzy or line-based match -- is what makes the recorded token position
trustworthy; anything looser would silently drift on whitespace or punctuation
differences between what was read and what is in the file.

Usage
    scripts/record_label.py runs/aime24_fresh/case_006/seed_0/lenience0p2 \\
        --category computation_error \\
        --quote "Wait compute: 80*272 = 21760; 2*272 = 544; sum=222." \\
        --note "sum of two correct partial products emitted as 222 instead of 22304"

    scripts/record_label.py ... --occurrence 2   # if the quote appears more than once
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lib_trace_align import align, token_at  # noqa: E402

CATEGORIES = (
    "semantic_nonsense",
    "repetition",
    "syntax_error",
    "computation_error",
    "reasoning_loop",
    "non_termination",
    "other",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=pathlib.Path)
    p.add_argument("--category", required=True, choices=CATEGORIES)
    p.add_argument("--quote", required=True, help="Exact substring of output.txt marking the onset.")
    p.add_argument("--occurrence", type=int, default=1, help="1-based, if --quote is not unique.")
    p.add_argument("--note", default="", help="Why this is the onset, in your own words.")
    p.add_argument("--severity", choices=("minor", "major", "fatal"), default="major")
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("analysis/degradation_labels_manual.jsonl"))
    return p.parse_args()


def parse_run_identity(run_dir: pathlib.Path) -> tuple[str, str, str]:
    parts = run_dir.resolve().parts
    return parts[-3], parts[-2], parts[-1]  # case, seed, tag


def main() -> int:
    args = parse_args()
    raw, recs = align(args.run_dir)
    if recs is None:
        print(f"error: cannot align {args.run_dir} (missing proposals.jsonl/output.txt, or trace mismatch)",
              file=sys.stderr)
        return 2

    needle = args.quote.encode("utf-8")
    positions, start = [], 0
    while True:
        k = raw.find(needle, start)
        if k < 0:
            break
        positions.append(k)
        start = k + 1
    if not positions:
        print(f"error: quote not found verbatim in output.txt: {args.quote!r}", file=sys.stderr)
        return 1
    if args.occurrence > len(positions):
        print(f"error: only {len(positions)} occurrence(s), asked for #{args.occurrence}", file=sys.stderr)
        return 1
    if len(positions) > 1 and args.occurrence == 1:
        print(f"warning: quote occurs {len(positions)} times; recording occurrence 1 "
              f"(pass --occurrence to pick another)", file=sys.stderr)

    byte_pos = positions[args.occurrence - 1]
    tok = token_at(recs, byte_pos)
    if tok is None:
        print(f"error: byte {byte_pos} falls in the prefill prefix (never verified); "
              f"cannot attribute to a token", file=sys.stderr)
        return 1

    case, seed, tag = parse_run_identity(args.run_dir)
    record = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "case": case,
        "seed": seed,
        "tag": tag,
        "method": "manual",
        "category": args.category,
        "severity": args.severity,
        "token_index": tok["token_index"],
        "output_position": tok["output_position"],
        "byte_start": tok["byte_start"],
        "byte_end": tok["byte_end"],
        "lossy_only_at_token": bool(tok.get("lossy_only_accepted")),
        "actually_accepted_at_token": bool(tok.get("actually_accepted")),
        "emission_source_at_token": tok.get("emission_source"),
        "quote": args.quote,
        "occurrence": args.occurrence,
        "note": args.note,
        "context": raw[max(0, byte_pos - 100) : tok["byte_end"] + 100].decode("utf-8", errors="replace"),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"recorded: {case} tok={tok['token_index']} pos={tok['output_position']} "
          f"category={args.category} lossy_only={tok.get('lossy_only_accepted')} "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
