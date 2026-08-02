#!/usr/bin/env python3
"""Count self-correction / hesitation markers across run outputs, per arm.

Degeneration under lossy verification shows up as the model repeatedly catching
and re-doing its own work, so the density of markers like "wait" and "recompute"
is a cheap proxy for it. Counts are normalised per 1000 generated tokens, because
the lossy arm produces longer outputs and raw counts would just re-measure length.

Usage:
    python scripts/analyze_markers.py                       # markdown table
    python scripts/analyze_markers.py --format csv
    python scripts/analyze_markers.py --arms b1_equiv10 b02_lenience --labels strict lossy
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

# Ordered so the table reads from "hesitation" to "explicit rework".
MARKERS: dict[str, str] = {
    "wait": r"\bwait\b",
    "hmm": r"\bhm+\b",
    "actually": r"\bactually\b",
    "oops": r"\boops\b|\bhold on\b",
    "mistake": r"\bmistake\b|\bmiscalc\w*|\bwrong\b",
    "should be": r"\bshould be\b|\bshould equal\b",
    "recompute": r"\brecompute\b|\bre-?calculat\w*|\bcompute again\b",
    "recheck": r"\brecheck\b|\bre-?check\b|\bdouble-?check\b|\bverify again\b",
    "redo": r"\bredo\b|\bdo (?:it|this|that) again\b|\bstart over\b|\bagain from\b",
    "let's compute": r"let'?s (?:compute|calculate)",
}

GROUPS: dict[str, tuple[str, ...]] = {
    "hesitation": ("wait", "hmm", "actually", "oops"),
    "error-flag": ("mistake", "should be"),
    "rework": ("recompute", "recheck", "redo", "let's compute"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", type=pathlib.Path, default=pathlib.Path("runs/aime24"))
    p.add_argument("--arms", nargs="+", default=["b1_equiv10", "b02_lenience"])
    p.add_argument("--labels", nargs="+", default=["strict", "lossy"])
    p.add_argument("--seed", default="seed_0")
    p.add_argument("--format", choices=("markdown", "csv"), default="markdown")
    p.add_argument("--out", type=pathlib.Path, help="Write here instead of stdout.")
    return p.parse_args()


def count(text: str) -> dict[str, int]:
    return {name: len(re.findall(pat, text, re.IGNORECASE)) for name, pat in MARKERS.items()}


def collect(runs_root: pathlib.Path, arm: str, seed: str) -> list[tuple[str, dict[str, int], int]]:
    """(case, marker counts, output tokens) for every case that has this arm."""
    rows = []
    for case_dir in sorted(runs_root.glob("case_*")):
        run_dir = case_dir / seed / arm
        out, meta = run_dir / "output.txt", run_dir / "run.json"
        if not (out.is_file() and meta.is_file()):
            continue
        tokens = json.loads(meta.read_text(encoding="utf-8")).get("output_tokens") or 0
        rows.append((case_dir.name, count(out.read_text(encoding="utf-8")), tokens))
    return rows


def per_1k(n: int, tokens: int) -> float:
    return (n / tokens * 1000) if tokens else 0.0


def render(args: argparse.Namespace) -> str:
    data = {label: collect(args.runs_root, arm, args.seed) for arm, label in zip(args.arms, args.labels)}
    labels = [l for l in args.labels if data.get(l)]
    if not labels:
        raise SystemExit(f"no runs found under {args.runs_root}")

    totals = {l: {m: sum(r[1][m] for r in data[l]) for m in MARKERS} for l in labels}
    toks = {l: sum(r[2] for r in data[l]) for l in labels}

    if args.format == "csv":
        lines = ["marker," + ",".join(f"{l}_count,{l}_per1k" for l in labels)]
        for m in MARKERS:
            cells = []
            for l in labels:
                cells += [str(totals[l][m]), f"{per_1k(totals[l][m], toks[l]):.2f}"]
            lines.append(m + "," + ",".join(cells))
        return "\n".join(lines) + "\n"

    out = []
    out.append(f"Markers per 1000 generated tokens, summed over "
               f"{len(data[labels[0]])} cases ({', '.join(f'{l}: {toks[l]:,} tok' for l in labels)}).\n")
    head = "| marker | " + " | ".join(f"{l} /1k" for l in labels) + " | ratio |"
    out.append(head)
    out.append("|---|" + "---:|" * len(labels) + "---:|")
    for m in MARKERS:
        vals = [per_1k(totals[l][m], toks[l]) for l in labels]
        ratio = (vals[-1] / vals[0]) if len(vals) > 1 and vals[0] else float("nan")
        cells = " | ".join(f"{v:.2f}" for v in vals)
        rtxt = "-" if ratio != ratio else f"{ratio:.2f}x"
        out.append(f"| {m} | {cells} | {rtxt} |")

    out.append("")
    out.append("| group | " + " | ".join(f"{l} /1k" for l in labels) + " | ratio |")
    out.append("|---|" + "---:|" * len(labels) + "---:|")
    for g, members in GROUPS.items():
        vals = [per_1k(sum(totals[l][m] for m in members), toks[l]) for l in labels]
        ratio = (vals[-1] / vals[0]) if len(vals) > 1 and vals[0] else float("nan")
        cells = " | ".join(f"{v:.2f}" for v in vals)
        rtxt = "-" if ratio != ratio else f"{ratio:.2f}x"
        out.append(f"| **{g}** | {cells} | **{rtxt}** |")

    # Per-case, so a single dominant case cannot masquerade as a trend.
    out.append("")
    out.append("Per-case totals across all markers, per 1k tokens:\n")
    out.append("| case | " + " | ".join(labels) + " | ratio |")
    out.append("|---|" + "---:|" * len(labels) + "---:|")
    cases = [r[0] for r in data[labels[0]]]
    up = 0
    for i, c in enumerate(cases):
        vals = []
        for l in labels:
            row = data[l][i]
            vals.append(per_1k(sum(row[1].values()), row[2]))
        ratio = (vals[-1] / vals[0]) if len(vals) > 1 and vals[0] else float("nan")
        if ratio == ratio and ratio > 1:
            up += 1
        cells = " | ".join(f"{v:.1f}" for v in vals)
        out.append(f"| {c} | {cells} | {'-' if ratio != ratio else f'{ratio:.2f}x'} |")
    if len(labels) > 1:
        out.append(f"\n{labels[-1]} higher in {up}/{len(cases)} cases.")
    return "\n".join(out) + "\n"


def main() -> int:
    args = parse_args()
    text = render(args)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
