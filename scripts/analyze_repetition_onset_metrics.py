#!/usr/bin/env python3
"""Join flagged repetition onsets back to their speculative-decoding metrics.

scan_repetitive_sections.py records WHERE a repetitive section starts
(onset_token_index) but not WHAT that token looked like to the decoder --
its target probability, rank, entropy, KL divergence from the draft, whether
it was only accepted because of the lossy rule, etc. Those live in
proposals.jsonl, keyed by the same token_index used everywhere else in this
project (trace_lookup.py, lib_trace_align.py, record_label.py).

This script does the join: for every flagged section across every judged
benchmark, look up its onset token's record and report the token's own text
alongside every per-token metric proposals.jsonl carries. It also computes a
BASELINE -- the same metrics averaged over every emitted token in the same
scanned runs that was NOT flagged as an onset -- so the onset numbers have
something to be compared against rather than read in isolation. Without the
baseline, "mean target_rank 18" tells you nothing; against "baseline
target_rank 3," it says onset tokens are unusually surprising to the model.

Benchmarks and their (judgements root, runs root) are hardcoded below rather
than auto-discovered: the AIME24 judgements predate the multi-benchmark
layout and live one directory level up from the other three (no dataset name
in the path), so a generic glob can't distinguish them from one convention.

Usage
    python scripts/analyze_repetition_onset_metrics.py
    python scripts/analyze_repetition_onset_metrics.py --labels unproductive_repetition
    python scripts/analyze_repetition_onset_metrics.py --top 40 --out data/onset_metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from scan_repetitive_sections import load_run  # noqa: E402 -- reuses the align()/direct_tokenize fallback

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# (dataset label, judgements root, runs root)
BENCHMARKS: list[tuple[str, pathlib.Path, pathlib.Path]] = [
    ("aime24", REPO_ROOT / "data/repetitive_sections/lenience0p2/seed_0/step_1000_lookback_1000",
     REPO_ROOT / "runs/aime24_fresh"),
    ("longbench_v2", REPO_ROOT / "data/repetitive_sections/longbench_v2/lenience0p2/seed_0/step_1000_lookback_1000",
     REPO_ROOT / "runs/longbench_v2_fresh"),
    ("mtbench", REPO_ROOT / "data/repetitive_sections/mtbench/lenience0p2/seed_0/step_1000_lookback_1000",
     REPO_ROOT / "runs/mtbench_fresh"),
    ("livecodebench", REPO_ROOT / "data/repetitive_sections/livecodebench/lenience0p2/seed_0/step_1000_lookback_1000",
     REPO_ROOT / "runs/livecodebench_fresh"),
]

# Every per-token field proposals.jsonl carries that is meaningful to average
# or count. Booleans are reported both as their raw value (in the detail rows)
# and as a share (in the aggregate rows).
NUMERIC_METRICS = (
    "p", "q", "p_over_q", "u", "target_rank", "target_top1_prob", "target_top1_shortfall",
    "target_entropy", "draft_entropy", "kl_target_draft", "kl_draft_target", "tv_distance",
    "consecutive_accepted_length",
)
BOOLEAN_METRICS = ("strict_would_accept", "lossy_would_accept", "actually_accepted", "lossy_only_accepted")

RunKey = tuple[str, str, str, str]  # (benchmark, case, seed, tag)

# target_top1_shortfall was renamed from target_top1_margin; older
# proposals.jsonl files predate the rename and still use the old key. Same
# quantity (top1_prob - p(x)) either name.
_LEGACY_METRIC_NAMES = {"target_top1_shortfall": "target_top1_margin"}


def get_metric(record: dict[str, Any], name: str) -> Any:
    if name in record:
        return record.get(name)
    legacy = _LEGACY_METRIC_NAMES.get(name)
    return record.get(legacy) if legacy else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", nargs="+", default=None,
                         help="Restrict onset rows to these judge labels (e.g. unproductive_repetition); default keeps all.")
    parser.add_argument("--benchmarks", nargs="+", default=None,
                         help="Restrict to these benchmark names; default runs all four.")
    parser.add_argument("--top", type=int, default=30, help="Rows to print in the by-token-text console table.")
    parser.add_argument("--min-count", type=int, default=1, help="Drop token-text groups seen fewer than this many times from the by-token-text table.")
    parser.add_argument("--skip-baseline", action="store_true",
                         help="Skip the non-onset baseline pass (faster, but loses the comparison).")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/onset_metrics"),
                         help="Directory for onset_detail.jsonl, the aggregate CSVs, and baseline_metrics.json.")
    return parser.parse_args()


@dataclass
class OnsetRow:
    benchmark: str
    case: str
    seed: str
    tag: str
    chunk_index: int
    section_index: int
    label: str
    category: str | None
    confidence: float
    onset_quote: str
    onset_token_index: int
    token_text: str
    metrics: dict[str, Any]


def load_judged_sections(judgements_root: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(judgements_root.glob("case_*/judgements/*/*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def collect_onset_rows(
    args: argparse.Namespace,
) -> tuple[list[OnsetRow], dict[str, set[RunKey]], dict[RunKey, list[dict[str, Any]] | None], Counter[str]]:
    """Returns (matched onset rows, every run touched per benchmark, a shared
    records cache keyed by run, and a counter of skip reasons).

    The run registry and cache cover every case scan_repetitive_sections.py
    judged -- not just ones with a flagged section -- so compute_baseline can
    reuse them for the full non-onset token population without re-aligning.
    """
    wanted_labels = set(args.labels) if args.labels else None
    wanted_benchmarks = set(args.benchmarks) if args.benchmarks else None
    onset_rows: list[OnsetRow] = []
    run_registry: dict[str, set[RunKey]] = defaultdict(set)
    run_cache: dict[RunKey, list[dict[str, Any]] | None] = {}
    skipped: Counter[str] = Counter()

    def records_for(benchmark: str, runs_root: pathlib.Path, case: str, seed: str, tag: str) -> list[dict[str, Any]] | None:
        key: RunKey = (benchmark, case, seed, tag)
        if key not in run_cache:
            run_dir = runs_root / case / seed / tag
            try:
                _raw, records = load_run(run_dir)
            except (RuntimeError, OSError, json.JSONDecodeError, KeyError) as exc:
                print(f"warning: cannot load {run_dir}: {type(exc).__name__}: {exc}", file=sys.stderr)
                records = None
            run_cache[key] = records
        return run_cache[key]

    for benchmark, judgements_root, runs_root in BENCHMARKS:
        if wanted_benchmarks and benchmark not in wanted_benchmarks:
            continue
        if not judgements_root.is_dir():
            print(f"warning: no judgements at {judgements_root}, skipping {benchmark}", file=sys.stderr)
            continue
        rows = load_judged_sections(judgements_root)

        for row in rows:
            case, seed, tag = str(row.get("case")), str(row.get("seed")), str(row.get("tag"))
            run_registry[benchmark].add((benchmark, case, seed, tag))
            for section_index, section in enumerate(row.get("sections") or []):
                label = section.get("label")
                if wanted_labels and label not in wanted_labels:
                    continue
                token_index = section.get("onset_token_index")
                if token_index is None:
                    continue
                records = records_for(benchmark, runs_root, case, seed, tag)
                if records is None:
                    skipped["run_unreadable"] += 1
                    continue
                idx = int(token_index) - 1  # token_index is 1-based
                if not (0 <= idx < len(records)) or int(records[idx].get("token_index", -1)) != int(token_index):
                    skipped["token_index_out_of_range"] += 1
                    continue
                record = records[idx]
                metrics = {name: get_metric(record, name) for name in NUMERIC_METRICS}
                metrics.update({name: record.get(name) for name in BOOLEAN_METRICS})
                metrics["emission_source"] = record.get("emission_source")
                onset_rows.append(
                    OnsetRow(
                        benchmark=benchmark, case=case, seed=seed, tag=tag,
                        chunk_index=int(row.get("chunk_index", -1)), section_index=section_index,
                        label=str(label), category=section.get("category"),
                        confidence=float(section.get("confidence", 0.0)),
                        onset_quote=str(section.get("onset_quote", "")),
                        onset_token_index=int(token_index),
                        token_text=str(record.get("text", "")),
                        metrics=metrics,
                    )
                )
    return onset_rows, dict(run_registry), run_cache, skipped


def compute_baseline(
    onset_rows: list[OnsetRow],
    run_registry: dict[str, set[RunKey]],
    run_cache: dict[RunKey, list[dict[str, Any]] | None],
) -> dict[str, dict[str, Any]]:
    """Average metrics over every emitted token in the scanned runs EXCEPT
    the ones actually selected as onsets (respecting --labels), per
    benchmark and overall. This is the population the onset numbers should
    be read against."""
    excluded: set[tuple[str, str, str, str, int]] = {
        (r.benchmark, r.case, r.seed, r.tag, r.onset_token_index) for r in onset_rows
    }
    values_by_benchmark: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    values_overall: dict[str, list[Any]] = defaultdict(list)

    for benchmark, keys in run_registry.items():
        for key in keys:
            _benchmark, case, seed, tag = key
            records = run_cache.get(key)
            if not records:
                continue
            for record in records:
                token_index = int(record.get("token_index", -1))
                if (benchmark, case, seed, tag, token_index) in excluded:
                    continue
                for name in NUMERIC_METRICS:
                    value = get_metric(record, name)
                    values_by_benchmark[benchmark][name].append(value)
                    values_overall[name].append(value)
                for name in BOOLEAN_METRICS:
                    value = record.get(name)
                    values_by_benchmark[benchmark][name].append(value)
                    values_overall[name].append(value)

    def summarize(values: dict[str, list[Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"n_tokens": len(values.get("p", []))}
        for name in NUMERIC_METRICS:
            out[f"mean_{name}"] = mean(values.get(name, []))
        for name in BOOLEAN_METRICS:
            bools = [v for v in values.get(name, []) if isinstance(v, bool)]
            out[f"share_{name}"] = (sum(bools) / len(bools)) if bools else None
        return out

    result = {benchmark: summarize(values) for benchmark, values in values_by_benchmark.items()}
    result["overall"] = summarize(values_overall)
    return result


def mean(values: list[Any]) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return statistics.fmean(nums) if nums else None


def write_detail(onset_rows: list[OnsetRow], out_dir: pathlib.Path) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "onset_detail.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in onset_rows:
            record = {
                "benchmark": row.benchmark, "case": row.case, "seed": row.seed, "tag": row.tag,
                "chunk_index": row.chunk_index, "section_index": row.section_index,
                "label": row.label, "category": row.category, "confidence": row.confidence,
                "onset_quote": row.onset_quote, "onset_token_index": row.onset_token_index,
                "token_text": row.token_text, **row.metrics,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def _aggregate(rows: list[OnsetRow], total_for_pct: int) -> dict[str, Any]:
    entry: dict[str, Any] = {"count": len(rows), "pct_of_onsets": 100 * len(rows) / total_for_pct if total_for_pct else None}
    for metric in NUMERIC_METRICS:
        entry[f"mean_{metric}"] = mean([r.metrics.get(metric) for r in rows])
    for metric in BOOLEAN_METRICS:
        values = [r.metrics.get(metric) for r in rows if isinstance(r.metrics.get(metric), bool)]
        entry[f"share_{metric}"] = (sum(values) / len(values)) if values else None
    return entry


def aggregate_by_token_text(onset_rows: list[OnsetRow]) -> list[dict[str, Any]]:
    """Grouped by token text alone, across every selected benchmark. pct_of_onsets
    is this token's share of the WHOLE selected onset population (all benchmarks
    combined) -- see aggregate_by_benchmark_and_token_text for a per-benchmark share."""
    groups: dict[str, list[OnsetRow]] = defaultdict(list)
    for row in onset_rows:
        groups[row.token_text].append(row)

    total = len(onset_rows)
    table = []
    for token_text, rows in groups.items():
        entry = {"token_text": token_text, "benchmarks": ",".join(sorted({r.benchmark for r in rows})),
                 "labels": ",".join(sorted({r.label for r in rows})), **_aggregate(rows, total)}
        table.append(entry)
    table.sort(key=lambda e: (-e["count"], e["token_text"]))
    return table


def aggregate_by_benchmark_and_token_text(onset_rows: list[OnsetRow]) -> list[dict[str, Any]]:
    """Grouped by (benchmark, token text). pct_of_onsets is this token's share of
    that BENCHMARK's onset population -- what the request asked for directly."""
    groups: dict[tuple[str, str], list[OnsetRow]] = defaultdict(list)
    totals: Counter[str] = Counter()
    for row in onset_rows:
        groups[(row.benchmark, row.token_text)].append(row)
        totals[row.benchmark] += 1

    table = []
    for (benchmark, token_text), rows in groups.items():
        entry = {"benchmark": benchmark, "token_text": token_text,
                  "labels": ",".join(sorted({r.label for r in rows})), **_aggregate(rows, totals[benchmark])}
        table.append(entry)
    table.sort(key=lambda e: (e["benchmark"], -e["count"], e["token_text"]))
    return table


def write_csv(table: list[dict[str, Any]], fieldnames: list[str], path: pathlib.Path) -> pathlib.Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in table:
            writer.writerow(entry)
    return path


AGGREGATE_METRIC_FIELDS = [f"mean_{m}" for m in NUMERIC_METRICS] + [f"share_{m}" for m in BOOLEAN_METRICS]


def print_metrics_table(rows: list[tuple[str, dict[str, Any]]]) -> None:
    print(f"{'':<15} {'n':>8} {'mean_p':>9} {'mean_rank':>10} {'mean_entropy':>13} {'mean_kl_t_d':>12} {'lossy_only%':>12}")
    for label, stats in rows:
        n = stats.get("n_tokens", stats.get("count"))
        lossy_only = stats.get("share_lossy_only_accepted")
        share = f"{100 * lossy_only:.1f}%" if lossy_only is not None else "-"
        p = stats.get("mean_p")
        rank = stats.get("mean_target_rank")
        entropy = stats.get("mean_target_entropy")
        kl = stats.get("mean_kl_target_draft")
        print(
            f"{label:<15} {n:>8} "
            f"{p if p is not None else float('nan'):>9.3f} "
            f"{rank if rank is not None else float('nan'):>10.1f} "
            f"{entropy if entropy is not None else float('nan'):>13.3f} "
            f"{kl if kl is not None else float('nan'):>12.4f} "
            f"{share:>12}"
        )


def print_summary(
    onset_rows: list[OnsetRow], by_text: list[dict[str, Any]], by_bench_text: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]] | None, args: argparse.Namespace,
) -> None:
    if not onset_rows:
        print("no onset rows matched the given filters")
        return

    print(f"{len(onset_rows):,} flagged sections joined to token metrics across "
          f"{len({(r.benchmark, r.case) for r in onset_rows}):,} cases\n")

    print("## Onset tokens, by benchmark\n")
    print_metrics_table([
        (benchmark, {"n_tokens": len(rows), **_aggregate(rows, len(rows))})
        for benchmark, _, _ in BENCHMARKS
        for rows in [[r for r in onset_rows if r.benchmark == benchmark]] if rows
    ])

    if baseline is not None:
        print("\n## Baseline: ALL non-onset tokens in the same scanned runs, by benchmark\n")
        print_metrics_table([(b, baseline[b]) for b, _, _ in BENCHMARKS if b in baseline])
        print("\n## Onset vs baseline, overall\n")
        onset_overall = {"n_tokens": len(onset_rows), **_aggregate(onset_rows, len(onset_rows))}
        print_metrics_table([("onset", onset_overall), ("baseline", baseline["overall"])])

    print("\n## Onset tokens, by label\n")
    print_metrics_table([
        (label, {"n_tokens": len(rows), **_aggregate(rows, len(rows))})
        for label in sorted({r.label for r in onset_rows})
        for rows in [[r for r in onset_rows if r.label == label]]
    ])

    shown = [e for e in by_text if e["count"] >= args.min_count][: args.top]
    print(f"\n## Top {len(shown)} onset token texts by occurrence, all selected benchmarks combined (min_count={args.min_count})\n")
    print(f"{'token_text':<24} {'count':>6} {'% of onsets':>12} {'mean_p':>9} {'mean_rank':>10} {'mean_entropy':>13} {'lossy_only%':>12}  benchmarks")
    for entry in shown:
        text = repr(entry["token_text"])[:22]
        lossy_only = entry.get("share_lossy_only_accepted")
        share = f"{100*lossy_only:.1f}%" if lossy_only is not None else "-"
        pct = entry.get("pct_of_onsets")
        rank = entry.get("mean_target_rank")
        entropy = entry.get("mean_target_entropy")
        p = entry.get("mean_p")
        print(
            f"{text:<24} {entry['count']:>6} "
            f"{pct if pct is not None else float('nan'):>11.1f}% "
            f"{p if p is not None else float('nan'):>9.3f} "
            f"{rank if rank is not None else float('nan'):>10.1f} "
            f"{entropy if entropy is not None else float('nan'):>13.3f} "
            f"{share:>12}  {entry['benchmarks']}"
        )

    per_bench_shown = [e for e in by_bench_text if e["count"] >= args.min_count][: args.top]
    print(f"\n## Top {len(per_bench_shown)} (benchmark, token_text) pairs by occurrence, with %-of-that-benchmark's-onsets (min_count={args.min_count})\n")
    print(f"{'benchmark':<14} {'token_text':<20} {'count':>6} {'% of bench onsets':>18} {'mean_p':>9} {'mean_rank':>10}")
    for entry in sorted(by_bench_text, key=lambda e: -e["count"])[: args.top]:
        if entry["count"] < args.min_count:
            continue
        text = repr(entry["token_text"])[:18]
        pct = entry.get("pct_of_onsets")
        p = entry.get("mean_p")
        rank = entry.get("mean_target_rank")
        print(
            f"{entry['benchmark']:<14} {text:<20} {entry['count']:>6} "
            f"{pct if pct is not None else float('nan'):>17.1f}% "
            f"{p if p is not None else float('nan'):>9.3f} "
            f"{rank if rank is not None else float('nan'):>10.1f}"
        )


def main() -> int:
    args = parse_args()
    onset_rows, run_registry, run_cache, skipped = collect_onset_rows(args)
    if skipped:
        print("skipped: " + ", ".join(f"{k}={v}" for k, v in skipped.items()), file=sys.stderr)

    args.out.mkdir(parents=True, exist_ok=True)
    detail_path = write_detail(onset_rows, args.out)

    by_text = aggregate_by_token_text(onset_rows)
    by_text_path = write_csv(
        by_text, ["token_text", "count", "pct_of_onsets", "benchmarks", "labels"] + AGGREGATE_METRIC_FIELDS,
        args.out / "onset_by_token_text.csv",
    )

    by_bench_text = aggregate_by_benchmark_and_token_text(onset_rows)
    by_bench_text_path = write_csv(
        by_bench_text, ["benchmark", "token_text", "count", "pct_of_onsets", "labels"] + AGGREGATE_METRIC_FIELDS,
        args.out / "onset_by_benchmark_and_token_text.csv",
    )

    baseline = None
    baseline_path = None
    if not args.skip_baseline:
        # Needs every judged run's full record set, not just ones with a flagged
        # section -- reload whichever weren't already pulled in during collection.
        for benchmark, judgements_root, runs_root in BENCHMARKS:
            for key in run_registry.get(benchmark, ()):
                if run_cache.get(key) is None and key not in run_cache:
                    _benchmark, case, seed, tag = key
                    try:
                        _raw, records = load_run(runs_root / case / seed / tag)
                    except (RuntimeError, OSError, json.JSONDecodeError, KeyError) as exc:
                        print(f"warning: cannot load {runs_root / case / seed / tag} for baseline: {exc}", file=sys.stderr)
                        records = None
                    run_cache[key] = records
        baseline = compute_baseline(onset_rows, run_registry, run_cache)
        baseline_path = args.out / "baseline_metrics.json"
        baseline_path.write_text(json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print_summary(onset_rows, by_text, by_bench_text, baseline, args)
    print(f"\nwrote {len(onset_rows):,} rows to {detail_path}")
    print(f"wrote {len(by_text):,} token-text groups to {by_text_path}")
    print(f"wrote {len(by_bench_text):,} (benchmark, token-text) groups to {by_bench_text_path}")
    if baseline_path:
        print(f"wrote baseline metrics to {baseline_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
