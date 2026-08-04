#!/usr/bin/env python3
"""Split a combined lossy-only token JSONL into per-case datasets.

The combined source is retained. By default, an input at::

    .../context_064/all_cases/tokens.jsonl

is split into::

    .../context_064/case_001/tokens.jsonl
    .../context_064/case_001/summary.json
    .../context_064/case_002/tokens.jsonl
    ...
"""

from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import re
from collections import Counter, defaultdict
from typing import Any, TextIO


DEFAULT_INPUT = pathlib.Path(
    "data/lossy_only_tokens/lenience0p2/seed_0/context_064/all_cases/tokens.jsonl"
)
CASE_PATTERN = re.compile(r"case_\d{3,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", nargs="?", type=pathlib.Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--summary",
        type=pathlib.Path,
        help="Combined summary; defaults to summary.json beside INPUT.",
    )
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        help="Parent of case_NNN directories; defaults above INPUT's all_cases directory.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_summary(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read combined summary {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"combined summary must be a JSON object: {path}")
    return value


def validate_cases(summary: dict[str, Any]) -> list[str]:
    cases = sorted({str(case) for case in summary.get("cases", [])})
    invalid = [case for case in cases if not CASE_PATTERN.fullmatch(case)]
    if not cases or invalid:
        raise SystemExit(f"summary has missing or invalid cases: {invalid or cases}")
    return cases


def rank_band(row: dict[str, Any]) -> str:
    rank = row.get("target_rank")
    rank = int(rank) if isinstance(rank, (int, float)) else 0
    if rank >= 100:
        return "rank_100_plus"
    if rank >= 20:
        return "rank_20_99"
    if rank >= 1:
        return "rank_1_19"
    return "rank_0"


def output_paths(root: pathlib.Path, cases: list[str]) -> dict[str, tuple[pathlib.Path, pathlib.Path]]:
    return {
        case: (root / case / "tokens.jsonl", root / case / "summary.json")
        for case in cases
    }


def refuse_existing(
    paths: dict[str, tuple[pathlib.Path, pathlib.Path]], overwrite: bool
) -> None:
    existing = [path for pair in paths.values() for path in pair if path.exists()]
    if existing and not overwrite:
        preview = ", ".join(str(path) for path in existing[:3])
        more = " ..." if len(existing) > 3 else ""
        raise SystemExit(
            f"refusing to overwrite {len(existing)} existing output file(s): "
            f"{preview}{more}; pass --overwrite"
        )


def write_case_summaries(
    combined: dict[str, Any],
    paths: dict[str, tuple[pathlib.Path, pathlib.Path]],
    counts: Counter[str],
    rank_counts: dict[str, Counter[str]],
) -> None:
    runs = combined.get("runs") or []
    for case, (_, summary_path) in paths.items():
        case_summary = {
            "cases": [case],
            "context_tokens_each_side": combined.get("context_tokens_each_side"),
            "output": str(paths[case][0]).replace("\\", "/"),
            "emitted_lossy_only_tokens": counts[case],
            "target_rank_bands": dict(sorted(rank_counts[case].items())),
            "runs": [
                run
                for run in runs
                if isinstance(run, dict) and run.get("case") == case
            ],
            "split_from": str(combined.get("output", "")),
        }
        summary_path.write_text(
            json.dumps(case_summary, indent=2) + "\n", encoding="utf-8"
        )


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"combined token file does not exist: {args.input}")
    summary_path = args.summary or args.input.parent / "summary.json"
    combined = load_summary(summary_path)
    cases = validate_cases(combined)
    root = args.output_root or args.input.parent.parent
    paths = output_paths(root, cases)
    refuse_existing(paths, args.overwrite)
    for token_path, _ in paths.values():
        token_path.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    rank_counts: dict[str, Counter[str]] = defaultdict(Counter)
    with contextlib.ExitStack() as stack:
        handles: dict[str, TextIO] = {
            case: stack.enter_context(
                token_path.open("w", encoding="utf-8", newline="\n")
            )
            for case, (token_path, _) in paths.items()
        }
        with args.input.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(
                        f"invalid JSON at {args.input}:{line_number}: {exc}"
                    ) from exc
                case = str(row.get("case"))
                if case not in handles:
                    raise SystemExit(
                        f"row {line_number} has case {case!r} absent from the summary"
                    )
                handles[case].write(line if line.endswith("\n") else line + "\n")
                counts[case] += 1
                rank_counts[case][rank_band(row)] += 1

    expected_total = combined.get("emitted_lossy_only_tokens")
    if expected_total != sum(counts.values()):
        raise SystemExit(
            f"row-count mismatch: summary={expected_total}, split={sum(counts.values())}"
        )
    empty = [case for case in cases if not counts[case]]
    if empty:
        raise SystemExit(f"no rows found for cases listed in summary: {empty}")

    write_case_summaries(combined, paths, counts, rank_counts)
    print(
        f"split {sum(counts.values()):,} rows into {len(cases)} case directories "
        f"under {root}"
    )
    for case in cases:
        print(f"{case}: {counts[case]:,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
