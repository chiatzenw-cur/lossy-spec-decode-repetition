#!/usr/bin/env python3
"""Find repeated passages across a WHOLE output trace, algorithmically.

This is the whole-output-scale counterpart to extract_lossy_only_tokens.py.
That script looked at single tokens with a 64-token local window; this one
looks for the model repeating itself anywhere in a run, including thousands of
tokens apart, which a fixed local window cannot see at all.

Detection is a plain token-exact repeat finder (shingle seeding + greedy
extension + union-find clustering), not an LLM call, so it is essentially free
to run over the full 1k-33k token traces in this project: each run takes well
under a second. See judge_repetition_clusters.py for the LLM stage, which only
looks at the small set of candidate passages this script finds, each through a
bounded local excerpt -- that split is what keeps the LLM stage wall-time
cheap despite operating at whole-output scope.

Algorithm
---------
1. Hash every K-token window (a "shingle") of the emitted token-id sequence.
2. For any shingle seen at 2+ positions, seed a match between consecutive
   occurrences (not all pairs -- an O(m) chain over a bucket of m identical
   shingles still finds the full cluster via union-find, in O(m) instead of
   O(m^2)).
3. Greedily extend each seed left and right while token ids keep matching, to
   the maximal non-overlapping exact match.
4. Union-find over match endpoints: matches sharing a position merge into one
   cluster, so a passage repeated N times (not just twice) becomes one
   cluster with N occurrences, found without pairwise N^2 comparison.

A cluster's first occurrence is its ORIGIN; every later one is a RECURRENCE to
be judged. Clusters are capped to --max-occurrences per run so one pathological
loop (the same 20 tokens repeated 80 times) cannot blow up the judging budget;
the true occurrence count is still recorded on every row.

Examples
--------
    python scripts/extract_repetition_clusters.py --case 002
    python scripts/extract_repetition_clusters.py --all-cases --context-tokens 50

Without --out, results use a structured path such as::

    data/repetition_clusters/lenience0p2/seed_0/context_050/all_cases/
        clusters.jsonl
        summary.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lib_trace_align import align  # noqa: E402


def normalize_case(value: str) -> str:
    value = value.strip()
    suffix = value[5:] if value.startswith("case_") else value
    if not suffix.isdigit():
        raise argparse.ArgumentTypeError(f"case must be numeric or case_NNN, got {value!r}")
    return f"case_{int(suffix):03d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cases = parser.add_mutually_exclusive_group(required=True)
    cases.add_argument("--case", dest="cases", action="append", type=normalize_case,
                        help="Case to scan, e.g. 004 or case_004. Repeat for multiple cases.")
    cases.add_argument("--all-cases", action="store_true", help="Discover every case with a matching proposal trace.")
    parser.add_argument("--runs-root", type=pathlib.Path, default=pathlib.Path("runs/aime24_fresh"))
    parser.add_argument("--seed", default="seed_0", help="Seed directory name, or 'all'.")
    parser.add_argument("--tag", default="lenience0p2")
    parser.add_argument("--shingle-tokens", type=int, default=7,
                         help="Exact-match seed length in tokens. Lower catches shorter repeats but is noisier.")
    parser.add_argument("--min-match-tokens", type=int, default=7,
                         help="Discard matches shorter than this after extension.")
    parser.add_argument("--context-tokens", type=int, default=50,
                         help="Tokens of context on each side of a match in the extracted excerpt.")
    parser.add_argument("--max-occurrences", type=int, default=5,
                         help="Judge at most this many recurrences per cluster (plus the origin); "
                              "the true total is still recorded on every row.")
    parser.add_argument("--out", type=pathlib.Path, help="Output JSONL. Defaults to a structured path.")
    parser.add_argument("--summary-out", type=pathlib.Path, help="Optional JSON summary; defaults beside the JSONL.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing generated file.")
    return parser.parse_args()


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()) or "unnamed"


def default_output_path(args: argparse.Namespace) -> pathlib.Path:
    scope = "all_cases" if args.all_cases else (args.cases[0] if len(args.cases) == 1 else "selected_cases")
    seed_scope = "all_seeds" if args.seed.lower() == "all" else args.seed
    return (
        pathlib.Path("data/repetition_clusters")
        / safe_component(args.tag)
        / safe_component(seed_scope)
        / f"context_{args.context_tokens:03d}"
        / scope
        / "clusters.jsonl"
    )


def discover_runs(args: argparse.Namespace) -> list[pathlib.Path]:
    runs = []
    seed_pattern = "seed_*" if args.seed.lower() == "all" else args.seed
    for case in sorted(set(args.cases)):
        runs.extend(sorted((args.runs_root / case).glob(f"{seed_pattern}/{args.tag}")))
    return [run for run in runs if (run / "proposals.jsonl").is_file()]


def discover_cases(args: argparse.Namespace) -> list[str]:
    seed_pattern = "seed_*" if args.seed.lower() == "all" else args.seed
    pattern = f"case_*/{seed_pattern}/{args.tag}/proposals.jsonl"
    return sorted({path.parts[-4] for path in args.runs_root.glob(pattern)})


# --- repeat detection -------------------------------------------------------


def build_shingle_positions(ids: list[int], k: int) -> dict[tuple[int, ...], list[int]]:
    positions: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for i in range(len(ids) - k + 1):
        positions[tuple(ids[i : i + k])].append(i)
    return positions


def extend_match(ids: list[int], i: int, j: int) -> tuple[int, int, int]:
    """Greedily extend a seed match (i, j), i < j, to the maximal non-overlapping exact match."""
    n = len(ids)
    length = 0
    while j + length < n and i + length < j and ids[i + length] == ids[j + length]:
        length += 1
    back = 0
    while i - back - 1 >= 0 and j - back - 1 > i + length - 1 and ids[i - back - 1] == ids[j - back - 1]:
        back += 1
    return i - back, j - back, back + length


def find_matches(ids: list[int], shingle_tokens: int, min_match_tokens: int) -> list[tuple[int, int, int]]:
    if len(ids) < shingle_tokens:
        return []
    positions = build_shingle_positions(ids, shingle_tokens)
    seen: set[tuple[int, int, int]] = set()
    matches = []
    for pos_list in positions.values():
        if len(pos_list) < 2:
            continue
        for i, j in zip(pos_list, pos_list[1:]):
            si, sj, length = extend_match(ids, i, j)
            if length < min_match_tokens:
                continue
            key = (si, sj, length)
            if key in seen:
                continue
            seen.add(key)
            matches.append(key)
    return matches


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def cluster_matches(matches: list[tuple[int, int, int]]) -> list[list[int]]:
    """Group matches sharing an endpoint into clusters of sorted start positions.

    Deliberately does not carry a match length here: a position can be an
    endpoint of several matches with different lengths (it may share a long
    run with one cluster member and a shorter one with another), so any
    single "the" length for a position would be wrong for some pairing. Each
    origin/recurrence pair recomputes its own exact shared length instead --
    see extract_run.
    """
    uf = UnionFind()
    starts: set[int] = set()
    for si, sj, _length in matches:
        uf.union(si, sj)
        starts.add(si)
        starts.add(sj)

    groups: dict[int, set[int]] = defaultdict(set)
    for start in starts:
        groups[uf.find(start)].add(start)

    clusters = [sorted(group) for group in groups.values() if len(group) >= 2]
    clusters.sort(key=lambda occ: occ[0])
    return clusters


# --- text/context extraction (byte-exact, mirrors extract_lossy_only_tokens.py) --


def char_position(raw: bytes, byte_position: int) -> int:
    return len(raw[:byte_position].decode("utf-8", errors="replace"))


def decode_span(raw: bytes, start: int, end: int) -> str:
    return raw[start:end].decode("utf-8", errors="replace").replace("\r", "")


def build_excerpt(raw: bytes, records: list[dict], start_idx: int, length: int, context_tokens: int) -> dict:
    """start_idx/length: 0-based record index and token count of the matched block."""
    end_idx = start_idx + length  # exclusive
    left_offset = max(0, start_idx - context_tokens)
    right_offset = min(len(records), end_idx + context_tokens)
    left_byte = records[left_offset]["byte_start"]
    right_byte = records[right_offset - 1]["byte_end"]
    base_char = char_position(raw, left_byte)

    match_byte_start = records[start_idx]["byte_start"]
    match_byte_end = records[end_idx - 1]["byte_end"]

    full_text = decode_span(raw, left_byte, right_byte)
    rel_match_start = char_position(raw, match_byte_start) - base_char
    rel_match_end = char_position(raw, match_byte_end) - base_char

    token_boundaries = [
        [char_position(raw, records[i]["byte_start"]) - base_char, int(records[i]["token_index"])]
        for i in range(left_offset, right_offset)
    ]

    return {
        "context_before": full_text[:rel_match_start],
        "match_text": full_text[rel_match_start:rel_match_end],
        "context_after": full_text[rel_match_end:],
        "token_boundaries": token_boundaries,
        "token_start": int(records[start_idx]["token_index"]),
        "token_end": int(records[end_idx - 1]["token_index"]),
        "excerpt_token_start": int(records[left_offset]["token_index"]),
        "excerpt_token_end": int(records[right_offset - 1]["token_index"]),
    }


def extract_run(run_dir: pathlib.Path, args: argparse.Namespace) -> tuple[list[dict], dict]:
    raw, records = align(run_dir)
    if raw is None or records is None:
        raise RuntimeError(f"cannot byte-align proposals to output: {run_dir}")

    case, seed, tag = run_dir.parts[-3:]
    ids = [int(r["emitted_token_id"]) for r in records]
    matches = find_matches(ids, args.shingle_tokens, args.min_match_tokens)
    clusters = cluster_matches(matches)

    rows = []
    for cluster in clusters:
        origin_anchor = cluster[0]
        cluster_id = f"{case}:{seed}:{tag}:{records[origin_anchor]['token_index']}"
        recurrence_anchors = cluster[1:]
        judged = recurrence_anchors[: args.max_occurrences]
        prev_end = None
        for occurrence_index, anchor in enumerate(judged, start=1):
            # Recompute the exact shared span for THIS origin/recurrence pair rather
            # than reusing a length from a different pairing in the cluster: two
            # occurrences can share a long run with one cluster member and a much
            # shorter one with another, so a cluster-wide length would sometimes
            # overrun into the next occurrence (observed as a negative gap).
            si, sj, length = extend_match(ids, origin_anchor, anchor)
            origin = build_excerpt(raw, records, si, length, args.context_tokens)
            recurrence = build_excerpt(raw, records, sj, length, args.context_tokens)
            gap_since_origin = sj - (si + length)
            gap_since_previous = gap_since_origin if prev_end is None else max(0, sj - prev_end)
            prev_end = sj + length
            rows.append(
                {
                    "case": case,
                    "seed": seed,
                    "tag": tag,
                    "cluster_id": cluster_id,
                    "shingle_tokens": args.shingle_tokens,
                    "min_match_tokens": args.min_match_tokens,
                    "context_tokens": args.context_tokens,
                    "occurrence_index": occurrence_index,
                    "occurrences_judged": len(judged),
                    "occurrences_total": len(recurrence_anchors),
                    "match_length_tokens": length,
                    "gap_tokens_since_origin": gap_since_origin,
                    "gap_tokens_since_previous": gap_since_previous,
                    "origin_token_start": origin["token_start"],
                    "origin_token_end": origin["token_end"],
                    "origin_context_before": origin["context_before"],
                    "origin_match_text": origin["match_text"],
                    "origin_context_after": origin["context_after"],
                    "recurrence_token_start": recurrence["token_start"],
                    "recurrence_token_end": recurrence["token_end"],
                    "recurrence_context_before": recurrence["context_before"],
                    "recurrence_match_text": recurrence["match_text"],
                    "recurrence_context_after": recurrence["context_after"],
                    "recurrence_token_boundaries": recurrence["token_boundaries"],
                    "recurrence_excerpt_token_start": recurrence["excerpt_token_start"],
                    "recurrence_excerpt_token_end": recurrence["excerpt_token_end"],
                }
            )

    summary = {
        "case": case,
        "seed": seed,
        "tag": tag,
        "emitted_tokens": len(records),
        "matches_found": len(matches),
        "clusters_found": len(clusters),
        "recurrences_total": sum(len(c) - 1 for c in clusters),
        "recurrences_judged": len(rows),
    }
    return rows, summary


def main() -> int:
    args = parse_args()
    if args.shingle_tokens <= 0:
        raise SystemExit("--shingle-tokens must be positive")
    if args.min_match_tokens < args.shingle_tokens:
        raise SystemExit("--min-match-tokens must be >= --shingle-tokens")
    if args.context_tokens < 0:
        raise SystemExit("--context-tokens must be nonnegative")
    if args.max_occurrences <= 0:
        raise SystemExit("--max-occurrences must be positive")

    if args.all_cases:
        args.cases = discover_cases(args)
        if not args.cases:
            raise SystemExit(f"no cases with proposal traces under {args.runs_root} for seed={args.seed} tag={args.tag}")

    output_was_default = args.out is None
    args.out = args.out or default_output_path(args)
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {args.out}; pass --overwrite to replace it")

    run_dirs = discover_runs(args)
    if not run_dirs:
        raise SystemExit("no matching proposal runs for cases " + ", ".join(sorted(set(args.cases))))
    found_cases = {run.parts[-3] for run in run_dirs}
    missing_cases = sorted(set(args.cases) - found_cases)
    if missing_cases:
        raise SystemExit("no matching proposal run for requested cases: " + ", ".join(missing_cases))

    rows: list[dict] = []
    run_summaries = []
    for run_dir in run_dirs:
        run_rows, run_summary = extract_run(run_dir, args)
        rows.extend(run_rows)
        run_summaries.append(run_summary)
    rows.sort(key=lambda row: (row["case"], row["seed"], row["tag"], row["origin_token_start"], row["occurrence_index"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary_path = args.summary_out or (
        args.out.parent / "summary.json"
        if output_was_default or args.out.name == "clusters.jsonl"
        else args.out.with_name(args.out.stem + "_summary.json")
    )
    summary = {
        "cases": sorted(set(args.cases)),
        "shingle_tokens": args.shingle_tokens,
        "min_match_tokens": args.min_match_tokens,
        "context_tokens": args.context_tokens,
        "max_occurrences": args.max_occurrences,
        "output": str(args.out).replace("\\", "/"),
        "rows_written": len(rows),
        "runs": run_summaries,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {len(rows)} candidate recurrences to {args.out}")
    print(f"wrote summary to {summary_path}")
    for s in run_summaries:
        print(
            f"{s['case']}/{s['seed']}/{s['tag']}: tokens={s['emitted_tokens']} "
            f"clusters={s['clusters_found']} recurrences_total={s['recurrences_total']} "
            f"recurrences_judged={s['recurrences_judged']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
