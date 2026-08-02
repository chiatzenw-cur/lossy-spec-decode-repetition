#!/usr/bin/env python3
"""Aggregate archived run directories into one comparable table."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


FIELDS = (
    "case",
    "seed",
    "tag",
    "mode",
    "status",
    "input_tokens",
    "output_tokens",
    "finish_reason",
    "eos_reached",
    "wall_time_seconds",
    "output_tokens_per_second",
    "repeat_ngram_tokens",
    "repeat_start_token",
    "spec_accept_rate",
    "spec_accept_length",
    "spec_verify_ct",
    "lossy_method",
    "threshold_acc",
    "output_chars",
    "output_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=pathlib.Path, default=pathlib.Path("runs"))
    parser.add_argument(
        "--tags",
        nargs="+",
        default=None,
        help="Restrict to these run tags. Default: every tag found.",
    )
    parser.add_argument(
        "--out-prefix",
        type=pathlib.Path,
        default=None,
        help="Write <prefix>.json/.csv/.md. Default: <runs-root>/summary.",
    )
    return parser.parse_args()


def read_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def collect_row(run_dir: pathlib.Path) -> dict[str, Any] | None:
    run = read_json(run_dir / "run.json")
    config = read_json(run_dir / "config.json")
    if not isinstance(run, dict) or not isinstance(config, dict):
        return None
    if not run and not config:
        return None

    repeat = run.get("consecutive_repeat_signal") or {}
    if not isinstance(repeat, dict):
        repeat = {}
    lossy = config.get("lossy_parameters") or {}
    if not isinstance(lossy, dict):
        lossy = {}

    output_path = run_dir / "output.txt"
    output_chars = None
    if output_path.is_file():
        try:
            output_chars = len(output_path.read_text(encoding="utf-8"))
        except OSError:
            output_chars = None

    output_tokens = run.get("output_tokens")
    wall = run.get("wall_time_seconds")
    throughput = None
    if isinstance(output_tokens, (int, float)) and isinstance(wall, (int, float)) and wall > 0:
        throughput = round(output_tokens / wall, 2)

    return {
        "case": run_dir.parent.parent.name,
        "seed": config.get("seed"),
        "tag": run_dir.name,
        "mode": config.get("mode"),
        "status": run.get("status", "error"),
        "input_tokens": run.get("input_tokens", config.get("input_tokens_archived")),
        "output_tokens": output_tokens,
        "finish_reason": run.get("finish_reason"),
        "eos_reached": run.get("eos_reached"),
        "wall_time_seconds": round(wall, 2) if isinstance(wall, (int, float)) else None,
        "output_tokens_per_second": throughput,
        "repeat_ngram_tokens": repeat.get("ngram_tokens"),
        "repeat_start_token": repeat.get("start_token"),
        "spec_accept_rate": run.get("spec_accept_rate"),
        "spec_accept_length": run.get("spec_accept_length"),
        "spec_verify_ct": run.get("spec_verify_ct"),
        "lossy_method": config.get("lossy_method"),
        "threshold_acc": lossy.get("threshold_acc"),
        "output_chars": output_chars,
        "output_path": str(output_path),
    }


def render_markdown(rows: list[dict[str, Any]]) -> str:
    columns = (
        "case",
        "tag",
        "seed",
        "input_tokens",
        "output_tokens",
        "finish_reason",
        "eos_reached",
        "wall_time_seconds",
        "output_tokens_per_second",
        "repeat_ngram_tokens",
    )
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join("" if row.get(c) is None else str(row[c]) for c in columns) + " |")
    return "\n".join(lines) + "\n"


def render_csv(rows: list[dict[str, Any]]) -> str:
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(FIELDS))
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in FIELDS})
    return buffer.getvalue()


def main() -> int:
    args = parse_args()
    if not args.runs_root.is_dir():
        print(f"no runs directory: {args.runs_root}", file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    for run_json in sorted(args.runs_root.glob("*/seed_*/*/run.json")):
        run_dir = run_json.parent
        if args.tags and run_dir.name not in args.tags:
            continue
        row = collect_row(run_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        print(f"no run.json files under {args.runs_root}", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: (str(r["case"]), str(r["tag"]), r["seed"] if r["seed"] is not None else -1))
    prefix = args.out_prefix or (args.runs_root / "summary")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps({"runs": rows, "count": len(rows)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    prefix.with_suffix(".csv").write_text(render_csv(rows), encoding="utf-8")
    prefix.with_suffix(".md").write_text(render_markdown(rows), encoding="utf-8")

    print(render_markdown(rows), end="")
    ok = sum(1 for row in rows if row["status"] == "ok")
    no_eos = sum(1 for row in rows if row["status"] == "ok" and not row["eos_reached"])
    repeats = sum(1 for row in rows if row["repeat_ngram_tokens"] is not None)
    print(
        f"\n{len(rows)} runs ({ok} ok), {no_eos} without EOS, {repeats} with a repeat signal",
        flush=True,
    )
    print(f"wrote {prefix}.json, {prefix}.csv, {prefix}.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
