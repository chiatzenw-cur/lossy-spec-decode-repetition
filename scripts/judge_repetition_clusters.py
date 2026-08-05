#!/usr/bin/env python3
"""Label whole-output-scale repeated passages with the baseline GPT-OSS-20B server.

extract_repetition_clusters.py finds candidate repeats algorithmically (token-
exact matching) across an entire trace -- cheap, and blind to meaning. This
script is the meaning check: for each candidate recurrence it shows the judge
a short excerpt of where the passage first appeared (ORIGIN) and a short
excerpt of the later occurrence being judged (RECURRENCE), and asks two
things: is this actually wasted, stuck repetition, and if so which exact words
in the RECURRENCE excerpt mark where the repeat begins.

This is a screening pass, not independent ground truth: GPT-OSS-20B is judging
text produced by GPT-OSS-20B. The prompt is blind to which arm (strict/lossy)
or algorithmic stats drove the flag; it only sees the two excerpts, the
problem, and a couple of factual counts from the detector.

Wall-time design: the detector (not this script) is what covers the whole
33k-token trace; this script only ever sends short, bounded excerpts -- one
call per candidate recurrence, capped per cluster by
extract_repetition_clusters.py's --max-occurrences. That keeps the per-call
cost close to the earlier local-token judge (judge_lossy_only_tokens.py)
while the candidates themselves come from whole-output scope.

Onset resolution: the judge must return onset_quote as an EXACT substring of
the RECURRENCE excerpt (context_before + match_text + context_after) -- the
same verbatim-quote discipline scripts/record_label.py uses for manual labels.
This script locates that substring and maps it to a token_index using the
token_boundaries recorded by the extractor, rather than trusting the judge to
count tokens itself (which BPE makes unreliable for any model to do exactly).

Run every case in ``runs/aime24_fresh``. Missing per-case extractions are
created automatically, then judged through one round-robin queue across cases::

    python scripts/judge_repetition_clusters.py

Preview all-case eligibility and the first prompt without calling the server::

    python scripts/judge_repetition_clusters.py --dry-run

The endpoint must be a plain, non-speculative GPT-OSS-20B vLLM server::

    PYTHON=$PWD/.venv-vllm/bin/python bash remote/run_server_vllm.sh baseline
    python scripts/judge_repetition_clusters.py
"""

from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import contextlib
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable


DEFAULT_RUNS_ROOT = pathlib.Path("runs/aime24_fresh")
DEFAULT_PROMPT_ROOT = pathlib.Path("prompts/aime24")
FINAL_MARKER = "<|channel|>final<|message|>"
PROMPT_VERSION = "repetition-cluster-v1"
LABELS = {"degrading_repetition", "benign_reference", "ambiguous"}
CATEGORIES = {"verbatim_loop", "re_derivation", "unresolved_cycle", "other"}
MIN_QUOTE_CHARS = 6

RUBRIC = """You are auditing a candidate REPEATED PASSAGE found by an automated scanner in a
language model's chain-of-thought on a competition maths problem. The scanner
found that an identical (or near-identical) run of text appears at TWO points
in the trace, possibly thousands of tokens apart. It has no understanding of
maths or English; it is a mechanical text match, so most flagged pairs are
ordinary and expected: restating a value when naming a variable, a deliberate
final recap, copying a problem's given numbers, or checking a result a second
time. Your job is to tell which flagged pairs are truly wasted, stuck
repetition and, for those, where the repeat itself begins.

You are shown two excerpts:
- ORIGIN: the first time this text appeared in the trace.
- RECURRENCE: a later point where the same (or nearly the same) text appears
  again. This is the occurrence to judge.

Both excerpts start and end mid-thought; that is expected, not a defect.

DECISION
Ask: at the RECURRENCE, is the model re-deriving or re-stating something it
already established, without making new progress -- or is it legitimately
reusing/reintroducing that content on the way to doing something new?

LABELS
- degrading_repetition: the RECURRENCE mostly repeats prior work with no new
  progress around it -- a stalled derivation, a stuck loop, or restating the
  same claim again for no visible reason.
- benign_reference: the RECURRENCE reuses the earlier content but the
  surrounding text is doing new work with it (assigning it to a definition,
  moving on to the next step, a deliberate one-off check, a final summary).
- ambiguous: the excerpts genuinely do not contain enough to decide.

The default, before you look, is benign_reference. Only call
degrading_repetition if the RECURRENCE excerpt itself shows no new work beyond
restating the matched text.

CATEGORY (use null unless label is degrading_repetition)
- verbatim_loop: the same short passage repeats with essentially no new
  surrounding content -- the model is stuck restating itself.
- re_derivation: the model reruns the full computation from scratch to reach a
  value it already had, instead of reusing it.
- unresolved_cycle: this occurrence is one step of a back-and-forth between
  the same small set of candidate values or approaches that never converges.
- other: a concrete repetition-driven waste of effort not covered above.

ONSET
If label is degrading_repetition, quote the shortest span of consecutive text
from the RECURRENCE EXCERPT block, copied EXACTLY (character for character,
including punctuation and spacing), that marks where the repeat itself begins
-- the point after which the RECURRENCE only restates prior content. This may
be the first duplicated word, or an earlier phrase like "let me redo this" if
that is genuinely where the unproductive repeat starts. If label is not
degrading_repetition, onset_quote must be "".

Reply with ONE JSON object and nothing else:
{"label":"degrading_repetition|benign_reference|ambiguous","category":null,
"onset_quote":"","confidence":0.0,
"note":"one concise sentence tied specifically to the RECURRENCE excerpt"}
For a degrading_repetition label, replace null with exactly one category string
from the list above, and onset_quote with the exact quote."""


@dataclass(frozen=True)
class Candidate:
    row: dict[str, Any]
    key: tuple[str, str, str, str, int]
    input_sha256: str
    output_path: pathlib.Path


@dataclass(frozen=True)
class Dataset:
    input_path: pathlib.Path
    output_path: pathlib.Path
    input_sha256: str


@dataclass(frozen=True)
class JudgeResult:
    candidate: Candidate
    judgement: dict[str, Any] | None
    raw_final: str
    attempts: int
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", type=pathlib.Path,
                         help="Existing clusters JSONL or directory containing clusters.jsonl.")
    parser.add_argument("--out", type=pathlib.Path, help="Judged JSONL override for a single input file only.")
    parser.add_argument("--runs-root", type=pathlib.Path, help=f"Auto-extract from runs here; defaults to {DEFAULT_RUNS_ROOT}.")
    parser.add_argument("--prompt-root", type=pathlib.Path, default=DEFAULT_PROMPT_ROOT,
                         help="Directory of case_NNN/source.json problem statements.")
    parser.add_argument("--cases", nargs="+", help="Cases to auto-extract, e.g. 004 case_005. Default: all cases.")
    parser.add_argument("--run-seed", default="seed_0", help="Run seed directory, or 'all'.")
    parser.add_argument("--tag", default="lenience0p2", help="Run arm directory to extract.")
    parser.add_argument("--work-dir", type=pathlib.Path, help="Root containing case_NNN dataset directories.")
    parser.add_argument("--refresh-extraction", action="store_true",
                         help="Rebuild every selected per-case clusters.jsonl before judging.")
    parser.add_argument("--shingle-tokens", type=int, default=7)
    parser.add_argument("--min-match-tokens", type=int, default=7)
    parser.add_argument("--context-tokens", type=int, default=50)
    parser.add_argument("--max-occurrences", type=int, default=5)
    parser.add_argument("--server-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="gpt-oss-20b")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"), help="Optional bearer token.")
    parser.add_argument("--judge-effort", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=0, help="Judge at most N pending rows; 0 means all.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Prepare input if needed and print counts/first prompt; do not call the judge.")
    parser.add_argument("--skip-server-check", action="store_true", help="Skip the /v1/models preflight request.")
    return parser.parse_args()


def normalize_case(value: str) -> str:
    value = value.strip()
    suffix = value[5:] if value.startswith("case_") else value
    if not suffix.isdigit():
        raise SystemExit(f"case must be numeric or formatted as case_NNN, got {value!r}")
    return f"case_{int(suffix):03d}"


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()) or "unnamed"


def default_work_dir(args: argparse.Namespace) -> pathlib.Path:
    seed_scope = "all_seeds" if args.run_seed.lower() == "all" else args.run_seed
    return (
        pathlib.Path("data/repetition_clusters")
        / safe_component(args.tag)
        / safe_component(seed_scope)
        / f"context_{args.context_tokens:03d}"
    )


def extraction_command(args: argparse.Namespace, cases: list[str] | None, input_path: pathlib.Path) -> list[str]:
    script = pathlib.Path(__file__).resolve().with_name("extract_repetition_clusters.py")
    command = [
        sys.executable, str(script),
        "--runs-root", str(args.runs_root or DEFAULT_RUNS_ROOT),
        "--tag", args.tag,
        "--shingle-tokens", str(args.shingle_tokens),
        "--min-match-tokens", str(args.min_match_tokens),
        "--context-tokens", str(args.context_tokens),
        "--max-occurrences", str(args.max_occurrences),
        "--out", str(input_path),
        "--summary-out", str(input_path.parent / "summary.json"),
        "--seed", args.run_seed,
    ]
    if cases:
        for case in cases:
            command.extend(["--case", case])
    else:
        command.append("--all-cases")
    if args.refresh_extraction:
        command.append("--overwrite")
    return command


def discovered_run_cases(args: argparse.Namespace) -> list[str]:
    seed_pattern = "seed_*" if args.run_seed.lower() == "all" else args.run_seed
    pattern = f"case_*/{seed_pattern}/{args.tag}/proposals.jsonl"
    runs_root = args.runs_root or DEFAULT_RUNS_ROOT
    return sorted({path.parts[-4] for path in runs_root.glob(pattern)})


def validate_cached_extraction(args: argparse.Namespace, expected_case: str, input_path: pathlib.Path) -> None:
    summary_path = input_path.parent / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot validate cached extraction {input_path}: {exc}; pass --refresh-extraction") from exc
    errors = []
    if sorted(str(c) for c in summary.get("cases", [])) != [expected_case]:
        errors.append("case set")
    if summary.get("shingle_tokens") != args.shingle_tokens:
        errors.append("shingle_tokens")
    if summary.get("min_match_tokens") != args.min_match_tokens:
        errors.append("min_match_tokens")
    if summary.get("context_tokens") != args.context_tokens:
        errors.append("context_tokens")
    if summary.get("max_occurrences") != args.max_occurrences:
        errors.append("max_occurrences")
    runs = summary.get("runs") or []
    if any(run.get("tag") != args.tag for run in runs if isinstance(run, dict)):
        errors.append("tag")
    if args.run_seed.lower() != "all" and any(run.get("seed") != args.run_seed for run in runs if isinstance(run, dict)):
        errors.append("seed")
    if errors:
        raise SystemExit(
            f"cached extraction {input_path} has incompatible {', '.join(errors)}; "
            "pass --refresh-extraction or choose another --work-dir"
        )


def default_output_path(input_path: pathlib.Path, args: argparse.Namespace) -> pathlib.Path:
    return input_path.parent / "judgements" / safe_component(args.model) / f"{args.judge_effort}.jsonl"


def explicit_input_paths(path: pathlib.Path) -> list[pathlib.Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise SystemExit(f"input path does not exist: {path}")
    direct = path / "clusters.jsonl"
    if direct.is_file():
        return [direct]
    split = sorted(path.glob("case_*/clusters.jsonl"))
    if not split:
        raise SystemExit(f"input directory has neither clusters.jsonl nor case_*/clusters.jsonl: {path}")
    return split


def prepare_split_inputs(args: argparse.Namespace) -> list[pathlib.Path]:
    args.runs_root = args.runs_root or DEFAULT_RUNS_ROOT
    cases = sorted({normalize_case(c) for c in args.cases}) if args.cases else discovered_run_cases(args)
    if not cases:
        raise SystemExit(f"no matching cases under {args.runs_root} for seed={args.run_seed} tag={args.tag}")
    root = args.work_dir or default_work_dir(args)
    inputs = []
    for case in cases:
        input_path = root / case / "clusters.jsonl"
        if args.refresh_extraction or not input_path.is_file():
            print(f"preparing {case} from {args.runs_root} in {input_path.parent}")
            input_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                subprocess.run(extraction_command(args, [case], input_path), check=True)
            except subprocess.CalledProcessError as exc:
                raise SystemExit(f"repetition-cluster extraction for {case} failed with exit code {exc.returncode}") from exc
        else:
            validate_cached_extraction(args, case, input_path)
        inputs.append(input_path)
    print(f"using {len(inputs)} split case files under {root}")
    return inputs


def resolve_datasets(args: argparse.Namespace) -> list[Dataset]:
    if args.input:
        incompatible = [
            name for name, val in (
                ("--runs-root", args.runs_root), ("--cases", args.cases),
                ("--work-dir", args.work_dir), ("--refresh-extraction", args.refresh_extraction),
            ) if val
        ]
        if incompatible:
            raise SystemExit("an explicit input cannot be combined with " + ", ".join(incompatible))
        input_paths = explicit_input_paths(args.input)
    else:
        input_paths = prepare_split_inputs(args)
    if args.out and len(input_paths) != 1:
        raise SystemExit("--out is only valid for one input file; split inputs write judgments inside each case directory")
    datasets = []
    for input_path in input_paths:
        output_path = args.out or default_output_path(input_path, args)
        if input_path.resolve() == output_path.resolve():
            raise SystemExit("input and output paths must be different")
        datasets.append(Dataset(input_path=input_path, output_path=output_path, input_sha256=file_sha256(input_path)))
    return datasets


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def identity(row: dict[str, Any]) -> tuple[str, str, str, str, int]:
    try:
        return (
            str(row["case"]), str(row["seed"]), str(row["tag"]),
            str(row["cluster_id"]), int(row["occurrence_index"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("row lacks a valid case/seed/tag/cluster_id/occurrence_index identity") from exc


_PROBLEM_CACHE: dict[str, str] = {}


def load_problem(prompt_root: pathlib.Path, case: str) -> str:
    if case in _PROBLEM_CACHE:
        return _PROBLEM_CACHE[case]
    try:
        source = json.loads((prompt_root / case / "source.json").read_text(encoding="utf-8"))
        problem = str(source.get("problem", "")).strip()
    except (OSError, json.JSONDecodeError):
        problem = ""
    _PROBLEM_CACHE[case] = problem
    return problem


def build_prompt(row: dict[str, Any], prompt_root: pathlib.Path) -> str:
    problem = load_problem(prompt_root, row["case"])
    stats = (
        f"This exact passage recurs {row['occurrences_total']} time(s) after its first appearance "
        f"in the full output. This is recurrence #{row['occurrence_index']} of the "
        f"{row['occurrences_judged']} being reviewed. The matched block is "
        f"{row['match_length_tokens']} tokens long; it first appeared "
        f"{row['gap_tokens_since_origin']} tokens earlier, and "
        f"{row['gap_tokens_since_previous']} tokens after the previous occurrence."
    )
    origin_text = row["origin_context_before"] + row["origin_match_text"] + row["origin_context_after"]
    recurrence_text = row["recurrence_context_before"] + row["recurrence_match_text"] + row["recurrence_context_after"]
    return (
        f"{RUBRIC}\n\n"
        f"PROBLEM:\n{problem}\n\n"
        f"{stats}\n\n"
        f"ORIGIN EXCERPT (first appearance):\n<<<\n{origin_text}\n>>>\n\n"
        f"RECURRENCE EXCERPT (judge this occurrence):\n<<<\n{recurrence_text}\n>>>"
    )


def resolve_onset_token(row: dict[str, Any], quote: str) -> int | None:
    text = row["recurrence_context_before"] + row["recurrence_match_text"] + row["recurrence_context_after"]
    pos = text.find(quote)
    if pos < 0:
        return None
    boundaries = row["recurrence_token_boundaries"]
    offsets = [b[0] for b in boundaries]
    idx = bisect.bisect_right(offsets, pos) - 1
    if idx < 0:
        idx = 0
    return int(boundaries[idx][1])


def select_candidates(rows: list[dict[str, Any]], input_sha256: str, output_path: pathlib.Path) -> tuple[list[Candidate], Counter[str]]:
    candidates: list[Candidate] = []
    skipped: Counter[str] = Counter()
    seen: set[tuple[str, str, str, str, int]] = set()
    for row_number, row in enumerate(rows, 1):
        try:
            key = identity(row)
        except ValueError as exc:
            print(f"warning: input row {row_number}: {exc}", file=sys.stderr)
            skipped["invalid"] += 1
            continue
        if key in seen:
            print(f"warning: duplicate input identity {key}; keeping first", file=sys.stderr)
            skipped["duplicate"] += 1
            continue
        seen.add(key)
        missing = [
            f for f in (
                "origin_context_before", "origin_match_text", "origin_context_after",
                "recurrence_context_before", "recurrence_match_text", "recurrence_context_after",
                "recurrence_token_boundaries",
            ) if f not in row
        ]
        if missing:
            print(f"warning: input row {row_number}: missing fields {missing}", file=sys.stderr)
            skipped["invalid"] += 1
            continue
        candidates.append(Candidate(row=row, key=key, input_sha256=input_sha256, output_path=output_path))
    return candidates, skipped


def round_robin_cases(candidates: list[Candidate]) -> list[Candidate]:
    buckets: dict[tuple[str, str, str], deque[Candidate]] = defaultdict(deque)
    for candidate in candidates:
        case, seed, tag, _, _ = candidate.key
        buckets[(case, seed, tag)].append(candidate)
    ordered: list[Candidate] = []
    active = sorted(buckets)
    while active:
        next_active = []
        for key in active:
            ordered.append(buckets[key].popleft())
            if buckets[key]:
                next_active.append(key)
        active = next_active
    return ordered


def compatible_provenance(row: dict[str, Any], args: argparse.Namespace, input_sha256: str) -> bool:
    return (
        row.get("judge_prompt_version") == PROMPT_VERSION
        and row.get("judge_model") == args.model
        and row.get("judge_effort") == args.judge_effort
        and row.get("judge_temperature") == args.temperature
        and row.get("judge_seed") == args.seed
        and row.get("judge_input_sha256") == input_sha256
        and row.get("judge_parse_ok") is True
    )


def completed_keys(path: pathlib.Path, args: argparse.Namespace, input_sha256: str) -> set[tuple[str, str, str, str, int]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    rows = load_jsonl(path)
    compatible: set[tuple[str, str, str, str, int]] = set()
    incompatible = 0
    for row in rows:
        if compatible_provenance(row, args, input_sha256):
            try:
                compatible.add(identity(row))
            except ValueError:
                incompatible += 1
        else:
            incompatible += 1
    if incompatible:
        raise SystemExit(
            f"refusing to append to {path}: found {incompatible} row(s) not produced "
            "by this model/prompt configuration; choose a new --out path"
        )
    return compatible


def harmony_renderer() -> Callable[[str, str], str]:
    try:
        from openai_harmony import (
            Conversation, HarmonyEncodingName, Message, ReasoningEffort, Role, SystemContent, load_harmony_encoding,
        )
    except ImportError as exc:
        raise SystemExit("openai-harmony is required; install requirements-tokenizer.txt") from exc

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    def render(user_text: str, effort: str) -> str:
        system = SystemContent.new().with_reasoning_effort(ReasoningEffort(effort.capitalize()))
        conversation = Conversation.from_messages(
            [Message.from_role_and_content(Role.SYSTEM, system), Message.from_role_and_content(Role.USER, user_text)]
        )
        tokens = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
        return encoding.decode(tokens)

    return render


def request_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def http_json(url: str, *, payload: dict[str, Any] | None, timeout: float, api_key: str | None) -> Any:
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        method="GET" if payload is None else "POST",
        headers=request_headers(api_key),
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def check_server(args: argparse.Namespace) -> None:
    response = http_json(f"{args.server_url.rstrip('/')}/v1/models", payload=None, timeout=min(args.timeout, 30.0), api_key=args.api_key)
    model_ids = {str(item.get("id")) for item in (response.get("data") or []) if isinstance(item, dict) and item.get("id")}
    if args.model not in model_ids:
        raise RuntimeError(f"model {args.model!r} is not served at {args.server_url}; available models: {sorted(model_ids) or 'none reported'}")


def parse_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def validate_judgement(value: dict[str, Any] | None, row: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if value is None:
        return None, "no JSON object in the final response"
    label = value.get("label")
    if label not in LABELS:
        return None, f"invalid label {label!r}"
    category = value.get("category")
    if isinstance(category, str) and category.lower() == "null":
        category = None
    if label == "degrading_repetition":
        if category not in CATEGORIES:
            return None, f"degrading_repetition needs one of {sorted(CATEGORIES)}, got {category!r}"
    elif category is not None:
        return None, f"{label} requires category null, got {category!r}"

    onset_quote = value.get("onset_quote")
    if not isinstance(onset_quote, str):
        return None, f"onset_quote must be a string, got {onset_quote!r}"
    onset_token_index = None
    if label == "degrading_repetition":
        if len(onset_quote.strip()) < MIN_QUOTE_CHARS:
            return None, f"onset_quote must be at least {MIN_QUOTE_CHARS} characters for a degrading_repetition label"
        onset_token_index = resolve_onset_token(row, onset_quote)
        if onset_token_index is None:
            return None, f"onset_quote not found verbatim in the RECURRENCE excerpt: {onset_quote!r}"
    elif onset_quote != "":
        return None, f"{label} requires onset_quote \"\", got {onset_quote!r}"

    confidence = value.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None, f"confidence must be numeric, got {confidence!r}"
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        return None, f"confidence must be in [0, 1], got {confidence}"

    note = value.get("note")
    if not isinstance(note, str) or not note.strip():
        return None, "note must be a non-empty string"

    return {
        "label": label,
        "category": category,
        "onset_quote": onset_quote,
        "onset_token_index": onset_token_index,
        "confidence": confidence,
        "note": note.strip(),
    }, None


def ask_once(args: argparse.Namespace, render: Callable[[str, str], str], row: dict[str, Any], user_prompt: str) -> tuple[dict[str, Any] | None, str, str | None]:
    payload = {
        "model": args.model,
        "prompt": render(user_prompt, args.judge_effort),
        "temperature": args.temperature,
        "top_p": 1.0,
        "max_tokens": args.max_new_tokens,
        "seed": args.seed,
        "add_special_tokens": False,
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        "stream": False,
    }
    response = http_json(f"{args.server_url.rstrip('/')}/v1/completions", payload=payload, timeout=args.timeout, api_key=args.api_key)
    choices = response.get("choices") or [{}]
    text = str(choices[0].get("text", "")) if isinstance(choices[0], dict) else ""
    final = text.split(FINAL_MARKER)[-1] if FINAL_MARKER in text else text
    parsed = parse_json_object(final)
    judgement, error = validate_judgement(parsed, row)
    return judgement, final, error


def judge_candidate(args: argparse.Namespace, render: Callable[[str, str], str], prompt_root: pathlib.Path, candidate: Candidate) -> JudgeResult:
    error: str | None = None
    raw_final = ""
    base_prompt = build_prompt(candidate.row, prompt_root)
    for attempt in range(1, args.max_attempts + 1):
        prompt = base_prompt
        if error:
            prompt += f"\n\nRETRY INSTRUCTION: The previous response was invalid: {error}. Return exactly the requested JSON schema."
        try:
            judgement, raw_final, error = ask_once(args, render, candidate.row, prompt)
        except (OSError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            judgement = None
            error = f"{type(exc).__name__}: {exc}"
            raw_final = ""
        if judgement is not None:
            return JudgeResult(candidate, judgement, raw_final, attempt, None)
        if attempt < args.max_attempts:
            time.sleep(min(30.0, args.retry_delay * 2 ** (attempt - 1)))
    return JudgeResult(candidate, None, raw_final, args.max_attempts, error)


def output_row(result: JudgeResult, args: argparse.Namespace) -> dict[str, Any]:
    assert result.judgement is not None
    row = {key: value for key, value in result.candidate.row.items() if not key.startswith("judge_")}
    row.update(
        {
            "judge_label": result.judgement["label"],
            "judge_category": result.judgement["category"],
            "judge_onset_quote": result.judgement["onset_quote"],
            "judge_onset_token_index": result.judgement["onset_token_index"],
            "judge_confidence": result.judgement["confidence"],
            "judge_note": result.judgement["note"],
            "judge_model": args.model,
            "judge_effort": args.judge_effort,
            "judge_prompt_version": PROMPT_VERSION,
            "judge_temperature": args.temperature,
            "judge_seed": args.seed,
            "judge_input_sha256": result.candidate.input_sha256,
            "judge_attempts": result.attempts,
            "judge_parse_ok": True,
            "judge_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "judge_raw_final": result.raw_final[:4000],
        }
    )
    return row


def print_selection_summary(total: int, runs: int, eligible: int, pending: int, completed: int, skipped: Counter[str]) -> None:
    print(f"input={total:,} runs={runs:,} eligible={eligible:,} completed={completed:,} pending={pending:,}")
    if skipped:
        print("excluded: " + ", ".join(f"{key}={value:,}" for key, value in skipped.items()))


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
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.max_attempts <= 0:
        raise SystemExit("--max-attempts must be positive")
    if args.limit < 0:
        raise SystemExit("--limit must be nonnegative")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    if args.retry_delay < 0:
        raise SystemExit("--retry-delay must be nonnegative")

    datasets = resolve_datasets(args)
    print(f"inputs: {len(datasets)} file(s)")
    if len(datasets) <= 3:
        for dataset in datasets:
            print(f"  {dataset.input_path} -> {dataset.output_path}")

    candidates: list[Candidate] = []
    completed: set[tuple[str, str, str, str, int]] = set()
    skipped: Counter[str] = Counter()
    total_rows = 0
    for dataset in datasets:
        rows = load_jsonl(dataset.input_path)
        total_rows += len(rows)
        case_candidates, case_skipped = select_candidates(rows, dataset.input_sha256, dataset.output_path)
        skipped.update(case_skipped)
        case_completed = set() if args.dry_run else completed_keys(dataset.output_path, args, dataset.input_sha256)
        case_keys = {c.key for c in case_candidates}
        unknown_completed = case_completed - case_keys
        if unknown_completed:
            raise SystemExit(
                f"refusing to resume {dataset.output_path}: it contains "
                f"{len(unknown_completed)} completed row(s) absent from its input"
            )
        candidates.extend(case_candidates)
        completed.update(case_completed)

    candidates = round_robin_cases(candidates)
    runs = len({c.key[:3] for c in candidates})
    pending = [c for c in candidates if c.key not in completed]
    if args.limit:
        pending = pending[: args.limit]
    print_selection_summary(total_rows, runs, len(candidates), len(pending), len(completed), skipped)

    if not candidates:
        return 2
    if args.dry_run:
        print("\n--- first eligible prompt ---\n")
        print(build_prompt(candidates[0].row, args.prompt_root))
        return 0
    if not pending:
        print("nothing to do")
        return 0

    if not args.skip_server_check:
        try:
            check_server(args)
        except (OSError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            raise SystemExit(f"server preflight failed: {exc}") from exc

    render = harmony_renderer()
    started = time.perf_counter()
    succeeded = failed = 0
    failures: list[JudgeResult] = []

    output_paths = sorted({c.output_path for c in pending})
    with contextlib.ExitStack() as stack:
        handles = {}
        for output_path in output_paths:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if output_path.exists() else "w"
            handles[output_path] = stack.enter_context(output_path.open(mode, encoding="utf-8", newline="\n"))
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending_iter = iter(pending)
            in_flight: set[concurrent.futures.Future[JudgeResult]] = set()
            for _ in range(min(len(pending), args.workers * 2)):
                in_flight.add(pool.submit(judge_candidate, args, render, args.prompt_root, next(pending_iter)))
            done = 0
            try:
                while in_flight:
                    finished, in_flight = concurrent.futures.wait(in_flight, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in finished:
                        result = future.result()
                        done += 1
                        try:
                            candidate = next(pending_iter)
                        except StopIteration:
                            pass
                        else:
                            in_flight.add(pool.submit(judge_candidate, args, render, args.prompt_root, candidate))
                        if result.judgement is None:
                            failed += 1
                            failures.append(result)
                            print(f"failed {result.candidate.key}: {result.error}", file=sys.stderr)
                        else:
                            handle = handles[result.candidate.output_path]
                            handle.write(json.dumps(output_row(result, args), ensure_ascii=False, separators=(",", ":")) + "\n")
                            handle.flush()
                            succeeded += 1
                        if done % 10 == 0 or done == len(pending):
                            elapsed = time.perf_counter() - started
                            rate = elapsed / done
                            remaining = rate * (len(pending) - done) / 60.0
                            print(f"{done:,}/{len(pending):,} processed; ok={succeeded:,} failed={failed:,}; ~{remaining:.1f} min left", flush=True)
            except KeyboardInterrupt:
                for future in in_flight:
                    future.cancel()
                print("interrupted; successful rows are saved and the command can resume", file=sys.stderr)
                return 130

    elapsed = time.perf_counter() - started
    labels: Counter[str] = Counter()
    for dataset in datasets:
        if dataset.output_path.is_file() and dataset.output_path.stat().st_size:
            labels.update(str(row.get("judge_label")) for row in load_jsonl(dataset.output_path))
    print(
        f"wrote {succeeded:,} new judgements across {len(output_paths):,} case file(s) in {elapsed / 60.0:.1f} min; "
        + ", ".join(f"{label}={count:,}" for label, count in labels.most_common())
    )
    if failures:
        print(f"{len(failures):,} rows failed after {args.max_attempts} attempts; re-run the same command to retry them", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
