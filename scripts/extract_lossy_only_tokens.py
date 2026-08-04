#!/usr/bin/env python3
"""Extract emitted lossy-only accepted tokens for selected experiment cases.

The proposal trace is byte-aligned to ``output.txt`` using the emitted token
IDs. This avoids BPE re-encoding drift and excludes proposal rows from an
uncommitted trailing verification batch.

Examples
--------
Extract one case with 32 tokens of context on each side::

    python scripts/extract_lossy_only_tokens.py --case 004

Extract several cases::

    python scripts/extract_lossy_only_tokens.py --case 004 --case case_005 \
        --out data/lossy_only_tokens_cases_004_005.jsonl

Prepare exact 64-token windows for every available case::

    python scripts/extract_lossy_only_tokens.py --all-cases --context-tokens 64

Without ``--out``, results use a structured path such as::

    data/lossy_only_tokens/lenience0p2/seed_0/context_064/all_cases/
        tokens.jsonl
        summary.json

The extractor retains boundary rows with shorter context.  The judge excludes
those rows rather than padding or presenting an asymmetric window.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lib_trace_align import align  # noqa: E402


ALIGNMENT_ONLY_FIELDS = {"byte_start", "byte_end", "token_index", "text"}


def normalize_case(value: str) -> str:
    value = value.strip()
    if value.startswith("case_"):
        suffix = value[5:]
    else:
        suffix = value
    if not suffix.isdigit():
        raise argparse.ArgumentTypeError(
            f"case must be numeric or formatted as case_NNN, got {value!r}"
        )
    return f"case_{int(suffix):03d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    cases = parser.add_mutually_exclusive_group(required=True)
    cases.add_argument(
        "--case",
        dest="cases",
        action="append",
        type=normalize_case,
        help="Case to extract, e.g. 004 or case_004. Repeat for multiple cases.",
    )
    cases.add_argument(
        "--all-cases",
        action="store_true",
        help="Discover every case with a matching proposal trace.",
    )
    parser.add_argument(
        "--runs-root",
        type=pathlib.Path,
        default=pathlib.Path("runs/aime24_fresh"),
    )
    parser.add_argument(
        "--seed",
        default="seed_0",
        help="Seed directory name, or 'all' to discover every seed.",
    )
    parser.add_argument("--tag", default="lenience0p2")
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=32,
        help="Number of aligned emitted tokens to include before and after each token.",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        help="Output JSONL. Defaults to an organized path under data/lossy_only_tokens/.",
    )
    parser.add_argument(
        "--summary-out",
        type=pathlib.Path,
        help="Optional JSON summary; defaults beside the token JSONL.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated token file.",
    )
    return parser.parse_args()


def char_position(raw: bytes, byte_position: int) -> int:
    return len(raw[:byte_position].decode("utf-8", errors="replace"))


def decode_span(raw: bytes, start: int, end: int) -> str:
    return raw[start:end].decode("utf-8", errors="replace").replace("\r", "")


def discover_runs(args: argparse.Namespace) -> list[pathlib.Path]:
    runs = []
    for case in sorted(set(args.cases)):
        seed_pattern = "seed_*" if args.seed.lower() == "all" else args.seed
        runs.extend(sorted((args.runs_root / case).glob(f"{seed_pattern}/{args.tag}")))
    return [run for run in runs if (run / "proposals.jsonl").is_file()]


def discover_cases(args: argparse.Namespace) -> list[str]:
    seed_pattern = "seed_*" if args.seed.lower() == "all" else args.seed
    pattern = f"case_*/{seed_pattern}/{args.tag}/proposals.jsonl"
    return sorted({path.parts[-4] for path in args.runs_root.glob(pattern)})


def safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned or "unnamed"


def default_output_path(args: argparse.Namespace) -> pathlib.Path:
    if args.all_cases:
        scope = "all_cases"
    elif len(args.cases) == 1:
        scope = args.cases[0]
    else:
        scope = "selected_cases"
    seed_scope = "all_seeds" if args.seed.lower() == "all" else args.seed
    return (
        pathlib.Path("data/lossy_only_tokens")
        / safe_component(args.tag)
        / safe_component(seed_scope)
        / f"context_{args.context_tokens:03d}"
        / scope
        / "tokens.jsonl"
    )


def extract_run(run_dir: pathlib.Path, context_tokens: int) -> tuple[list[dict], dict]:
    raw, records = align(run_dir)
    if raw is None or records is None:
        raise RuntimeError(f"cannot byte-align proposals to output: {run_dir}")

    case, seed, tag = run_dir.parts[-3:]
    rows = []
    for record_offset, record in enumerate(records):
        if not record.get("lossy_only_accepted"):
            continue
        left_offset = max(0, record_offset - context_tokens)
        right_offset = min(len(records), record_offset + context_tokens + 1)
        left_byte = records[left_offset]["byte_start"]
        right_byte = records[right_offset - 1]["byte_end"]
        token_text = decode_span(raw, record["byte_start"], record["byte_end"])
        context_before = decode_span(raw, left_byte, record["byte_start"])
        context_after = decode_span(raw, record["byte_end"], right_byte)

        # Preserve every original proposals.jsonl field at top level. Alignment
        # and context fields below add exact locations in the committed output.
        row = {
            key: value
            for key, value in record.items()
            if key not in ALIGNMENT_ONLY_FIELDS
        }
        row.update(
            {
                "case": case,
                "seed": seed,
                "tag": tag,
                "token_index": int(record["token_index"]),
                "token_position_0based": int(record["token_index"]) - 1,
                "byte_start": int(record["byte_start"]),
                "byte_end": int(record["byte_end"]),
                "char_start": char_position(raw, record["byte_start"]),
                "char_end": char_position(raw, record["byte_end"]),
                "token_text": token_text,
                "context_token_start": int(records[left_offset]["token_index"]),
                "context_token_end": int(records[right_offset - 1]["token_index"]),
                "context_before": context_before,
                "context_after": context_after,
                "marked_context": context_before + "⟦" + token_text + "⟧" + context_after,
            }
        )
        rows.append(row)

    proposal_count = sum(
        1
        for line in (run_dir / "proposals.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return rows, {
        "case": case,
        "seed": seed,
        "tag": tag,
        "proposal_rows": proposal_count,
        "committed_emitted_rows": len(records),
        "uncommitted_trailing_rows": proposal_count - len(records),
        "emitted_lossy_only_tokens": len(rows),
    }


def main() -> int:
    args = parse_args()
    if args.context_tokens < 0:
        raise SystemExit("--context-tokens must be nonnegative")
    if args.all_cases:
        args.cases = discover_cases(args)
        if not args.cases:
            raise SystemExit(
                f"no cases with proposal traces under {args.runs_root} for "
                f"seed={args.seed} tag={args.tag}"
            )
    output_was_default = args.out is None
    args.out = args.out or default_output_path(args)
    if args.out.exists() and not args.overwrite:
        raise SystemExit(
            f"refusing to overwrite {args.out}; pass --overwrite to replace it"
        )
    run_dirs = discover_runs(args)
    if not run_dirs:
        raise SystemExit(
            "no matching proposal runs for cases " + ", ".join(sorted(set(args.cases)))
        )
    found_cases = {run.parts[-3] for run in run_dirs}
    missing_cases = sorted(set(args.cases) - found_cases)
    if missing_cases:
        raise SystemExit(
            "no matching proposal run for requested cases: " + ", ".join(missing_cases)
        )

    rows = []
    run_summaries = []
    for run_dir in run_dirs:
        run_rows, run_summary = extract_run(run_dir, args.context_tokens)
        rows.extend(run_rows)
        run_summaries.append(run_summary)
    rows.sort(key=lambda row: (row["case"], row["seed"], row["tag"], row["token_index"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary_path = args.summary_out or (
        args.out.parent / "summary.json"
        if output_was_default or args.out.name == "tokens.jsonl"
        else args.out.with_name(args.out.stem + "_summary.json")
    )
    rank_counts = Counter(
        "rank_100_plus"
        if (row.get("target_rank") or 0) >= 100
        else "rank_20_99"
        if (row.get("target_rank") or 0) >= 20
        else "rank_1_19"
        if (row.get("target_rank") or 0) >= 1
        else "rank_0"
        for row in rows
    )
    summary = {
        "cases": sorted(set(args.cases)),
        "context_tokens_each_side": args.context_tokens,
        "output": str(args.out).replace("\\", "/"),
        "emitted_lossy_only_tokens": len(rows),
        "target_rank_bands": dict(sorted(rank_counts.items())),
        "runs": run_summaries,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {len(rows)} emitted lossy-only tokens to {args.out}")
    print(f"wrote summary to {summary_path}")
    for run_summary in run_summaries:
        print(
            f"{run_summary['case']}/{run_summary['seed']}/{run_summary['tag']}: "
            f"lossy_only={run_summary['emitted_lossy_only_tokens']} "
            f"uncommitted_trailing={run_summary['uncommitted_trailing_rows']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
