#!/usr/bin/env python3
"""Compare metrics for benign and non-benign judged token events.

The default comparison is deliberately non-ambiguous: ``benign`` rows are
compared with ``degradation`` rows, while ``ambiguous`` rows are counted but
excluded.  The report includes raw distribution summaries, Cliff's delta, a
tie-corrected Mann-Whitney screening test, and Benjamini-Hochberg adjusted
p-values.

The tests treat rows as independent, which is not literally true for adjacent
tokens from one generation.  They are useful for ranking candidate signals,
not for confirmatory inference.  Replication across independent cases and
seeds should be the next step.

Examples
--------
Print the default Markdown report::

    python scripts/analyze_judged_tokens.py

Write machine-readable output::

    python scripts/analyze_judged_tokens.py --format csv --out analysis/metrics.csv

Choose a smaller metric set or redefine the non-benign group::

    python scripts/analyze_judged_tokens.py --metrics target_rank tv_distance p
    python scripts/analyze_judged_tokens.py \
        --non-benign-labels degradation ambiguous
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import pathlib
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable


DEFAULT_INPUT = pathlib.Path(
    "data/lossy_only_tokens/lenience0p2/seed_0/context_064"
)
DEFAULT_JUDGEMENT_GLOB = "case_*/judgements/gpt-oss-20b/medium.jsonl"

# Curated rather than auto-discovered: identifiers, byte offsets, and duplicate
# token-position fields are numeric too, but they are not model diagnostics.
DEFAULT_METRICS = (
    "p",
    "q",
    "p_over_q",
    "u",
    "target_rank",
    "target_top1_prob",
    "target_top1_margin",
    "target_entropy",
    "draft_entropy",
    "kl_target_draft",
    "kl_draft_target",
    "tv_distance",
    "consecutive_accepted_length",
    "pos_in_round",
    "output_position",
)


@dataclass(frozen=True)
class Distribution:
    n: int
    missing: int
    mean: float
    std: float
    q1: float
    median: float
    q3: float


@dataclass
class MetricResult:
    metric: str
    benign: Distribution
    non_benign: Distribution
    mean_difference: float
    median_difference: float
    cliffs_delta: float
    mann_whitney_u: float
    p_value: float
    q_value: float = math.nan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", nargs="?", type=pathlib.Path, default=DEFAULT_INPUT)
    parser.add_argument("--benign-label", default="benign")
    parser.add_argument(
        "--non-benign-labels",
        nargs="+",
        default=["degradation"],
        help="Labels in the non-benign group. Ambiguous is excluded by default.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Numeric fields to compare.",
    )
    parser.add_argument(
        "--sort",
        choices=("effect", "input", "q-value"),
        default="effect",
        help="Ordering of metric rows in the report.",
    )
    parser.add_argument("--format", choices=("markdown", "csv", "json"), default="markdown")
    parser.add_argument("--out", type=pathlib.Path, help="Write here instead of stdout.")
    return parser.parse_args()


def resolve_inputs(path: pathlib.Path) -> list[pathlib.Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        direct = path / "judgements/gpt-oss-20b/medium.jsonl"
        if direct.is_file():
            return [direct]
        split = sorted(path.glob(DEFAULT_JUDGEMENT_GLOB))
        if split:
            return split
    raise SystemExit(
        f"no judged JSONL at {path} or under {path / DEFAULT_JUDGEMENT_GLOB}"
    )


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"cannot open {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise SystemExit(f"expected an object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise SystemExit(f"no rows found in {path}")
    return rows


def numeric_values(rows: Iterable[dict[str, Any]], metric: str) -> tuple[list[float], int]:
    values: list[float] = []
    missing = 0
    for row in rows:
        value = row.get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            missing += 1
            continue
        value = float(value)
        if not math.isfinite(value):
            missing += 1
            continue
        values.append(value)
    return values, missing


def percentile(sorted_values: list[float], fraction: float) -> float:
    """Linearly interpolated quantile, matching the common type-7 definition."""
    if not sorted_values:
        return math.nan
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize(values: list[float], missing: int) -> Distribution:
    ordered = sorted(values)
    return Distribution(
        n=len(ordered),
        missing=missing,
        mean=statistics.fmean(ordered) if ordered else math.nan,
        std=statistics.stdev(ordered) if len(ordered) > 1 else math.nan,
        q1=percentile(ordered, 0.25),
        median=percentile(ordered, 0.50),
        q3=percentile(ordered, 0.75),
    )


def mann_whitney(non_benign: list[float], benign: list[float]) -> tuple[float, float, float]:
    """Return U, Cliff's delta, and a two-sided asymptotic p-value.

    U and delta are oriented so positive delta means larger non-benign values.
    The normal approximation includes tie and continuity corrections.
    """
    n_non = len(non_benign)
    n_benign = len(benign)
    if not n_non or not n_benign:
        return math.nan, math.nan, math.nan

    pooled = [(value, 1) for value in non_benign]
    pooled.extend((value, 0) for value in benign)
    pooled.sort(key=lambda item: item[0])

    rank_sum_non = 0.0
    tie_term = 0
    start = 0
    while start < len(pooled):
        end = start + 1
        while end < len(pooled) and pooled[end][0] == pooled[start][0]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        rank_sum_non += average_rank * sum(group for _, group in pooled[start:end])
        tie_size = end - start
        tie_term += tie_size**3 - tie_size
        start = end

    u_value = rank_sum_non - n_non * (n_non + 1) / 2.0
    pair_count = n_non * n_benign
    cliffs_delta = 2.0 * u_value / pair_count - 1.0

    total = n_non + n_benign
    variance = pair_count / 12.0 * (
        total + 1.0 - tie_term / (total * (total - 1.0))
    )
    if variance <= 0.0:
        p_value = 1.0
    else:
        distance = abs(u_value - pair_count / 2.0)
        z_value = max(0.0, distance - 0.5) / math.sqrt(variance)
        p_value = math.erfc(z_value / math.sqrt(2.0))
    return u_value, cliffs_delta, p_value


def adjust_bh(results: list[MetricResult]) -> None:
    """Apply the Benjamini-Hochberg false-discovery-rate adjustment in place."""
    finite = [
        (index, result.p_value)
        for index, result in enumerate(results)
        if math.isfinite(result.p_value)
    ]
    finite.sort(key=lambda item: item[1])
    adjusted = [math.nan] * len(results)
    running_minimum = 1.0
    count = len(finite)
    for rank_index in range(count - 1, -1, -1):
        result_index, p_value = finite[rank_index]
        rank = rank_index + 1
        running_minimum = min(running_minimum, p_value * count / rank)
        adjusted[result_index] = min(1.0, running_minimum)
    for result, q_value in zip(results, adjusted):
        result.q_value = q_value


def compare_metrics(
    benign_rows: list[dict[str, Any]],
    non_benign_rows: list[dict[str, Any]],
    metrics: list[str],
) -> list[MetricResult]:
    results = []
    for metric in metrics:
        benign, benign_missing = numeric_values(benign_rows, metric)
        non_benign, non_benign_missing = numeric_values(non_benign_rows, metric)
        if not benign or not non_benign:
            print(
                f"warning: skipping metric {metric!r}; it has no numeric values "
                "in one or both comparison groups",
                file=sys.stderr,
            )
            continue
        benign_summary = summarize(benign, benign_missing)
        non_benign_summary = summarize(non_benign, non_benign_missing)
        u_value, cliffs_delta, p_value = mann_whitney(non_benign, benign)
        results.append(
            MetricResult(
                metric=metric,
                benign=benign_summary,
                non_benign=non_benign_summary,
                mean_difference=non_benign_summary.mean - benign_summary.mean,
                median_difference=non_benign_summary.median - benign_summary.median,
                cliffs_delta=cliffs_delta,
                mann_whitney_u=u_value,
                p_value=p_value,
            )
        )
    if not results:
        raise SystemExit("none of the requested metrics contained numeric values")
    adjust_bh(results)
    return results


def ordered_results(
    results: list[MetricResult], sort_order: str, metrics: list[str]
) -> list[MetricResult]:
    if sort_order == "input":
        positions = {metric: index for index, metric in enumerate(metrics)}
        return sorted(results, key=lambda result: positions[result.metric])
    if sort_order == "q-value":
        return sorted(results, key=lambda result: (result.q_value, -abs(result.cliffs_delta)))
    return sorted(results, key=lambda result: abs(result.cliffs_delta), reverse=True)


def format_number(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return "NA"
    if value == 0.0:
        return "0"
    if abs(value) < 10 ** (-digits) or abs(value) >= 10 ** (digits + 1):
        return f"{value:.3e}"
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def format_distribution(distribution: Distribution) -> str:
    median = format_number(distribution.median)
    q1 = format_number(distribution.q1)
    q3 = format_number(distribution.q3)
    mean = format_number(distribution.mean)
    return f"{median} [{q1}, {q3}]; {mean}"


def category_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(row.get("judge_category") or "uncategorized") for row in rows)


def render_markdown(
    path: pathlib.Path,
    all_rows: list[dict[str, Any]],
    benign_rows: list[dict[str, Any]],
    non_benign_rows: list[dict[str, Any]],
    excluded_rows: list[dict[str, Any]],
    results: list[MetricResult],
    args: argparse.Namespace,
) -> str:
    label_counts = Counter(str(row.get("judge_label")) for row in all_rows)
    excluded_counts = Counter(str(row.get("judge_label")) for row in excluded_rows)
    lines = ["# Judged-token metric screen", ""]
    lines.append(f"Input: `{path.as_posix()}` ({len(all_rows):,} rows)")
    lines.append("")
    lines.append(
        f"Clean comparison: **{len(benign_rows):,} {args.benign_label}** vs "
        f"**{len(non_benign_rows):,} non-benign** "
        f"({', '.join(args.non_benign_labels)})."
    )
    lines.append(
        f"Excluded from the comparison: **{len(excluded_rows):,}** rows"
        + (f" ({dict(excluded_counts)})" if excluded_counts else "")
        + "."
    )
    lines.append("")
    lines.append("Label counts: " + ", ".join(f"{label}={count:,}" for label, count in label_counts.most_common()) + ".")
    if non_benign_rows:
        lines.append(
            "Non-benign categories: "
            + ", ".join(f"{category}={count:,}" for category, count in category_counts(non_benign_rows).most_common())
            + "."
        )
    lines.extend(
        [
            "",
            "Values are `median [Q1, Q3]; mean`. Differences and Cliff's delta are "
            "oriented as non-benign minus benign; positive means higher among "
            "non-benign rows.",
            "",
            "| metric | benign | non-benign | median diff | mean diff | Cliff's delta | MW p | BH q |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        lines.append(
            f"| {result.metric} | {format_distribution(result.benign)} | "
            f"{format_distribution(result.non_benign)} | "
            f"{format_number(result.median_difference)} | "
            f"{format_number(result.mean_difference)} | "
            f"{format_number(result.cliffs_delta, 3)} | "
            f"{format_number(result.p_value, 3)} | "
            f"{format_number(result.q_value, 3)} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- `MW p` is a two-sided, tie-corrected Mann-Whitney normal approximation; "
            "`BH q` adjusts across the displayed metrics.",
            "- The metric named `p` is the target-model token probability; do not confuse "
            "it with the `MW p` test column.",
            "- P-values are exploratory because neighboring token rows within each case "
            "are dependent and the default dataset contains only one seed.",
            "- The analysis is conditional on a token already being lossy-only accepted. "
            "It identifies associations with the judgment, not causal effects of lenience.",
            "- Large class imbalance makes raw accuracy a poor follow-up criterion; validate "
            "candidate metrics across independent cases/seeds and report precision-recall behavior.",
        ]
    )
    return "\n".join(lines) + "\n"


def result_to_flat_dict(result: MetricResult) -> dict[str, Any]:
    row: dict[str, Any] = {"metric": result.metric}
    for prefix, distribution in (("benign", result.benign), ("non_benign", result.non_benign)):
        for name, value in asdict(distribution).items():
            row[f"{prefix}_{name}"] = value
    row.update(
        {
            "mean_difference": result.mean_difference,
            "median_difference": result.median_difference,
            "cliffs_delta": result.cliffs_delta,
            "mann_whitney_u": result.mann_whitney_u,
            "p_value": result.p_value,
            "q_value": result.q_value,
        }
    )
    return row


def render_csv(results: list[MetricResult]) -> str:
    rows = [result_to_flat_dict(result) for result in results]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def render_json(
    path: pathlib.Path,
    all_rows: list[dict[str, Any]],
    benign_rows: list[dict[str, Any]],
    non_benign_rows: list[dict[str, Any]],
    excluded_rows: list[dict[str, Any]],
    results: list[MetricResult],
    args: argparse.Namespace,
) -> str:
    payload = {
        "input": str(path),
        "rows": len(all_rows),
        "label_counts": dict(Counter(str(row.get("judge_label")) for row in all_rows)),
        "comparison": {
            "benign_label": args.benign_label,
            "benign_rows": len(benign_rows),
            "non_benign_labels": args.non_benign_labels,
            "non_benign_rows": len(non_benign_rows),
            "excluded_rows": len(excluded_rows),
            "excluded_label_counts": dict(Counter(str(row.get("judge_label")) for row in excluded_rows)),
        },
        "non_benign_category_counts": dict(category_counts(non_benign_rows)),
        "metrics": [result_to_flat_dict(result) for result in results],
    }
    return json.dumps(payload, indent=2, allow_nan=False) + "\n"


def main() -> int:
    args = parse_args()
    if args.benign_label in args.non_benign_labels:
        raise SystemExit("--benign-label must not also appear in --non-benign-labels")

    input_paths = resolve_inputs(args.input)
    rows = []
    for input_path in input_paths:
        rows.extend(load_jsonl(input_path))
    benign_rows = [row for row in rows if row.get("judge_label") == args.benign_label]
    non_benign_labels = set(args.non_benign_labels)
    non_benign_rows = [row for row in rows if row.get("judge_label") in non_benign_labels]
    selected_ids = {id(row) for row in benign_rows + non_benign_rows}
    excluded_rows = [row for row in rows if id(row) not in selected_ids]
    if not benign_rows:
        raise SystemExit(f"no rows with benign label {args.benign_label!r}")
    if not non_benign_rows:
        raise SystemExit(f"no rows with non-benign labels {args.non_benign_labels!r}")

    results = compare_metrics(benign_rows, non_benign_rows, args.metrics)
    results = ordered_results(results, args.sort, args.metrics)
    if args.format == "csv":
        text = render_csv(results)
    elif args.format == "json":
        text = render_json(
            args.input,
            rows,
            benign_rows,
            non_benign_rows,
            excluded_rows,
            results,
            args,
        )
    else:
        text = render_markdown(
            args.input,
            rows,
            benign_rows,
            non_benign_rows,
            excluded_rows,
            results,
            args,
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
