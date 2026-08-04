#!/usr/bin/env python3
"""Label extracted lossy-only tokens with the baseline GPT-OSS-20B server.

This is a screening pass, not independent ground truth: GPT-OSS-20B is judging
text produced by GPT-OSS-20B.  The prompt is deliberately blind to trace
metrics and asks whether the marked token itself introduces or completes a
visible degradation/error in its local context.

Each eligible prompt contains exactly 64 emitted tokens before the candidate,
the candidate token, and exactly 64 emitted tokens after it. Rows without the
full window are excluded.

Run every case in ``runs/aime24_fresh``. Missing per-case extractions are created
automatically, then judged through one round-robin queue across cases::

    python scripts/judge_lossy_only_tokens.py

Generated files are grouped by arm, seed, context size, and case::

    data/lossy_only_tokens/lenience0p2/seed_0/context_064/
        case_001/
            tokens.jsonl
            summary.json
            judgements/gpt-oss-20b/medium.jsonl
        case_002/
            ...

Preview all-case eligibility and the first prompt without calling the server::

    python scripts/judge_lossy_only_tokens.py --dry-run

Select cases, override the generated-data directory, or use an existing extract::

    python scripts/judge_lossy_only_tokens.py --cases 004 005 006
    python scripts/judge_lossy_only_tokens.py --work-dir data/my_token_screen
    python scripts/judge_lossy_only_tokens.py path/to/tokens.jsonl --out path/to/judged.jsonl

The endpoint must be a plain, non-speculative GPT-OSS-20B vLLM server::

    PYTHON=$PWD/.venv-vllm/bin/python bash remote/run_server_vllm.sh baseline
    python scripts/judge_lossy_only_tokens.py

Successful rows are appended and flushed immediately to their case-specific
judgment files. Re-running the same command resumes compatible rows. Failed
calls are not written, so a later run retries them.
"""

from __future__ import annotations

import argparse
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
FINAL_MARKER = "<|channel|>final<|message|>"
PROMPT_VERSION = "lossy-token-v1"
LABELS = {"benign", "degradation", "ambiguous"}
CATEGORIES = {
    "syntax_error",
    "semantic_nonsense",
    "computation_error",
    "repetition",
    "reasoning_loop",
    "other",
}

RUBRIC = """You are auditing ONE marked token emitted in a mathematical reasoning trace.

Most marked tokens are expected to be benign. Judge only the visible text. You
are not given token probabilities or decoding metadata, and must not speculate
about why the token was selected for review.

ATTRIBUTION RULE
Decide whether the TARGET TOKEN itself introduces, completes, or is an essential
part of a concrete defect. The surrounding window may contain an unrelated
mistake; do not blame the target token merely because it is nearby. If removing
or replacing the target token would not address the visible defect, label it
benign. Use ambiguous only when the local text really cannot resolve attribution.

The snippets contain exactly {WINDOW_TOKENS} model tokens on each side, but their
outer edges still begin and end mid-thought. Do not penalize missing material
outside those edges. Normal scratchpad behavior is benign: terse fragments,
exploratory cases, self-correction, an abandoned approach, repetition needed to
check work, ordinary punctuation/whitespace, and locally incomplete reasoning.

LABELS
- benign: the target token is locally grammatical and meaningful, or is not
  responsible for any visible defect.
- degradation: the target token itself creates/completes a concrete error.
- ambiguous: a plausible defect involves the token, but 64 tokens on each side
  are insufficient to decide whether the token is actually erroneous.

CATEGORY (use null unless label is degradation)
- syntax_error: broken grammar, malformed notation, or a token splice that no
  longer parses.
- semantic_nonsense: an out-of-context word, non-sequitur, contradiction, or
  locally meaningless claim.
- computation_error: a locally checkable false arithmetic or mathematical value.
- repetition: an unnecessary immediate duplication introduced by the token.
- reasoning_loop: the token visibly continues a repeated derivation with no
  progress; ordinary checking or one self-correction is not a loop.
- other: a concrete token-attributable degradation not covered above.

Reply with ONE JSON object and nothing else:
{"label":"benign|degradation|ambiguous","category":null,"confidence":0.0,
"note":"one concise sentence tied specifically to the target token"}
For a degradation label, replace null with exactly one category string from the
list above."""


@dataclass(frozen=True)
class Candidate:
    row: dict[str, Any]
    key: tuple[str, str, str, int]
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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=pathlib.Path,
        help="Existing token JSONL or directory containing tokens.jsonl.",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        help="Judged JSONL override for a single input file only.",
    )
    parser.add_argument(
        "--runs-root",
        type=pathlib.Path,
        help=f"Auto-extract from runs here; defaults to {DEFAULT_RUNS_ROOT}.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        help="Cases to auto-extract, e.g. 004 case_005. Default: all cases.",
    )
    parser.add_argument("--run-seed", default="seed_0", help="Run seed directory, or 'all'.")
    parser.add_argument("--tag", default="lenience0p2", help="Run arm directory to extract.")
    parser.add_argument(
        "--work-dir",
        type=pathlib.Path,
        help="Root containing case_NNN dataset directories.",
    )
    parser.add_argument(
        "--refresh-extraction",
        action="store_true",
        help="Rebuild every selected per-case tokens.jsonl before judging.",
    )
    parser.add_argument("--window-tokens", type=int, default=64)
    parser.add_argument("--server-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="gpt-oss-20b")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="Optional bearer token; defaults to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--judge-effort", choices=("low", "medium", "high"), default="medium"
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument(
        "--limit", type=int, default=0, help="Judge at most N pending rows; 0 means all."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare input if needed and print counts/first prompt; do not call the judge.",
    )
    parser.add_argument(
        "--skip-server-check",
        action="store_true",
        help="Skip the /v1/models preflight request.",
    )
    return parser.parse_args()


def normalize_case(value: str) -> str:
    value = value.strip()
    suffix = value[5:] if value.startswith("case_") else value
    if not suffix.isdigit():
        raise SystemExit(
            f"case must be numeric or formatted as case_NNN, got {value!r}"
        )
    return f"case_{int(suffix):03d}"


def safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned or "unnamed"


def default_work_dir(args: argparse.Namespace) -> pathlib.Path:
    seed_scope = "all_seeds" if args.run_seed.lower() == "all" else args.run_seed
    return (
        pathlib.Path("data/lossy_only_tokens")
        / safe_component(args.tag)
        / safe_component(seed_scope)
        / f"context_{args.window_tokens:03d}"
    )


def extraction_command(
    args: argparse.Namespace,
    cases: list[str] | None,
    input_path: pathlib.Path,
) -> list[str]:
    script = pathlib.Path(__file__).resolve().with_name("extract_lossy_only_tokens.py")
    command = [
        sys.executable,
        str(script),
        "--runs-root",
        str(args.runs_root or DEFAULT_RUNS_ROOT),
        "--tag",
        args.tag,
        "--context-tokens",
        str(args.window_tokens),
        "--out",
        str(input_path),
        "--summary-out",
        str(input_path.parent / "summary.json"),
    ]
    command.extend(["--seed", args.run_seed])
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


def validate_cached_extraction(
    args: argparse.Namespace,
    expected_case: str,
    input_path: pathlib.Path,
) -> None:
    summary_path = input_path.parent / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"cannot validate cached extraction {input_path}: {exc}; "
            "pass --refresh-extraction"
        ) from exc
    actual_cases = sorted(str(case) for case in summary.get("cases", []))
    errors = []
    if summary.get("context_tokens_each_side") != args.window_tokens:
        errors.append("context size")
    if actual_cases != [expected_case]:
        errors.append("case set")
    runs = summary.get("runs") or []
    if any(run.get("tag") != args.tag for run in runs if isinstance(run, dict)):
        errors.append("tag")
    if args.run_seed.lower() != "all" and any(
        run.get("seed") != args.run_seed for run in runs if isinstance(run, dict)
    ):
        errors.append("seed")
    if errors:
        raise SystemExit(
            f"cached extraction {input_path} has incompatible {', '.join(errors)}; "
            "pass --refresh-extraction or choose another --work-dir"
        )


def default_output_path(input_path: pathlib.Path, args: argparse.Namespace) -> pathlib.Path:
    return (
        input_path.parent
        / "judgements"
        / safe_component(args.model)
        / f"{args.judge_effort}.jsonl"
    )


def explicit_input_paths(path: pathlib.Path) -> list[pathlib.Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise SystemExit(f"input path does not exist: {path}")
    direct = path / "tokens.jsonl"
    if direct.is_file():
        return [direct]
    split = sorted(path.glob("case_*/tokens.jsonl"))
    if not split:
        raise SystemExit(
            f"input directory has neither tokens.jsonl nor case_*/tokens.jsonl: {path}"
        )
    return split


def prepare_split_inputs(args: argparse.Namespace) -> list[pathlib.Path]:
    args.runs_root = args.runs_root or DEFAULT_RUNS_ROOT
    cases = (
        sorted({normalize_case(case) for case in args.cases})
        if args.cases
        else discovered_run_cases(args)
    )
    if not cases:
        raise SystemExit(
            f"no matching cases under {args.runs_root} for "
            f"seed={args.run_seed} tag={args.tag}"
        )
    root = args.work_dir or default_work_dir(args)
    inputs = []
    for case in cases:
        input_path = root / case / "tokens.jsonl"
        if args.refresh_extraction or not input_path.is_file():
            print(f"preparing {case} from {args.runs_root} in {input_path.parent}")
            input_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                subprocess.run(
                    extraction_command(args, [case], input_path), check=True
                )
            except subprocess.CalledProcessError as exc:
                raise SystemExit(
                    f"token extraction for {case} failed with exit code {exc.returncode}"
                ) from exc
        else:
            validate_cached_extraction(args, case, input_path)
        inputs.append(input_path)
    print(f"using {len(inputs)} split case files under {root}")
    return inputs


def resolve_datasets(args: argparse.Namespace) -> list[Dataset]:
    if args.input:
        incompatible = []
        if args.runs_root is not None:
            incompatible.append("--runs-root")
        if args.cases:
            incompatible.append("--cases")
        if args.work_dir is not None:
            incompatible.append("--work-dir")
        if args.refresh_extraction:
            incompatible.append("--refresh-extraction")
        if incompatible:
            raise SystemExit(
                "an explicit input cannot be combined with " + ", ".join(incompatible)
            )
        input_paths = explicit_input_paths(args.input)
    else:
        input_paths = prepare_split_inputs(args)
    if args.out and len(input_paths) != 1:
        raise SystemExit(
            "--out is only valid for one input file; split inputs write judgments "
            "inside each case directory"
        )
    datasets = []
    for input_path in input_paths:
        output_path = args.out or default_output_path(input_path, args)
        if input_path.resolve() == output_path.resolve():
            raise SystemExit("input and output paths must be different")
        datasets.append(
            Dataset(
                input_path=input_path,
                output_path=output_path,
                input_sha256=file_sha256(input_path),
            )
        )
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


def identity(row: dict[str, Any]) -> tuple[str, str, str, int]:
    try:
        return (
            str(row["case"]),
            str(row["seed"]),
            str(row["tag"]),
            int(row["token_index"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("row lacks a valid case/seed/tag/token_index identity") from exc


def context_counts(row: dict[str, Any]) -> tuple[int, int]:
    try:
        token_index = int(row["token_index"])
        left = token_index - int(row["context_token_start"])
        right = int(row["context_token_end"]) - token_index
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("row lacks valid context token indices") from exc
    return left, right


def build_prompt(row: dict[str, Any], window_tokens: int) -> str:
    """Build a metric-blind prompt around one exact extracted token."""
    left, right = context_counts(row)
    if (left, right) != (window_tokens, window_tokens):
        raise ValueError(
            f"requires exactly {window_tokens}+{window_tokens} context tokens, "
            f"found {left}+{right}"
        )
    try:
        before = str(row["context_before"])
        token = str(row["token_text"])
        after = str(row["context_after"])
    except KeyError as exc:
        raise ValueError(f"row lacks extracted text field {exc.args[0]!r}") from exc

    token_json = json.dumps(token, ensure_ascii=False)
    joined = before + "[TARGET_TOKEN_START]" + token + "[TARGET_TOKEN_END]" + after
    rubric = RUBRIC.replace("{WINDOW_TOKENS}", str(window_tokens))
    return (
        f"{rubric}\n\n"
        f"TARGET TOKEN (exact JSON string, preserving whitespace):\n{token_json}\n\n"
        f"WINDOW: {window_tokens} TOKENS BEFORE + TARGET + "
        f"{window_tokens} TOKENS AFTER:\n<window>\n{joined}\n</window>"
    )


def select_candidates(
    rows: list[dict[str, Any]],
    window_tokens: int,
    input_sha256: str,
    output_path: pathlib.Path,
) -> tuple[list[Candidate], Counter[str]]:
    candidates: list[Candidate] = []
    skipped: Counter[str] = Counter()
    seen: set[tuple[str, str, str, int]] = set()
    for row_number, row in enumerate(rows, 1):
        if row.get("lossy_only_accepted") is not True:
            skipped["not_lossy_only"] += 1
            continue
        try:
            key = identity(row)
            left, right = context_counts(row)
        except ValueError as exc:
            print(f"warning: input row {row_number}: {exc}", file=sys.stderr)
            skipped["invalid"] += 1
            continue
        if key in seen:
            print(f"warning: duplicate input identity {key}; keeping first", file=sys.stderr)
            skipped["duplicate"] += 1
            continue
        seen.add(key)
        if left < window_tokens or right < window_tokens:
            skipped["incomplete_window"] += 1
            continue
        if left != window_tokens or right != window_tokens:
            # Re-tokenizing decoded context is unsafe because BPE encoding is not
            # round-trip stable. Re-extract at the requested size instead.
            skipped["wrong_window_size"] += 1
            continue
        missing_text = [
            field
            for field in ("context_before", "token_text", "context_after")
            if field not in row
        ]
        if missing_text:
            print(
                f"warning: input row {row_number}: missing text fields {missing_text}",
                file=sys.stderr,
            )
            skipped["invalid"] += 1
            continue
        candidates.append(
            Candidate(
                row=row,
                key=key,
                input_sha256=input_sha256,
                output_path=output_path,
            )
        )
    return candidates, skipped


def round_robin_cases(candidates: list[Candidate]) -> list[Candidate]:
    """Interleave cases so concurrent workers do not drain one case at a time."""
    buckets: dict[tuple[str, str, str], deque[Candidate]] = defaultdict(deque)
    for candidate in candidates:
        case, seed, tag, _ = candidate.key
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


def compatible_provenance(
    row: dict[str, Any], args: argparse.Namespace, input_sha256: str
) -> bool:
    return (
        row.get("judge_prompt_version") == PROMPT_VERSION
        and row.get("judge_window_tokens") == args.window_tokens
        and row.get("judge_model") == args.model
        and row.get("judge_effort") == args.judge_effort
        and row.get("judge_temperature") == args.temperature
        and row.get("judge_seed") == args.seed
        and row.get("judge_input_sha256") == input_sha256
        and row.get("judge_parse_ok") is True
    )


def completed_keys(
    path: pathlib.Path, args: argparse.Namespace, input_sha256: str
) -> set[tuple[str, str, str, int]]:
    if not path.exists():
        return set()
    if path.stat().st_size == 0:
        return set()
    rows = load_jsonl(path)
    compatible: set[tuple[str, str, str, int]] = set()
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
            "by this model/window/prompt configuration; choose a new --out path"
        )
    return compatible


def harmony_renderer() -> Callable[[str, str], str]:
    try:
        from openai_harmony import (
            Conversation,
            HarmonyEncodingName,
            Message,
            ReasoningEffort,
            Role,
            SystemContent,
            load_harmony_encoding,
        )
    except ImportError as exc:
        raise SystemExit(
            "openai-harmony is required; install requirements-tokenizer.txt"
        ) from exc

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    def render(user_text: str, effort: str) -> str:
        system = SystemContent.new().with_reasoning_effort(
            ReasoningEffort(effort.capitalize())
        )
        conversation = Conversation.from_messages(
            [
                Message.from_role_and_content(Role.SYSTEM, system),
                Message.from_role_and_content(Role.USER, user_text),
            ]
        )
        tokens = encoding.render_conversation_for_completion(
            conversation, Role.ASSISTANT
        )
        return encoding.decode(tokens)

    return render


def request_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def http_json(
    url: str,
    *,
    payload: dict[str, Any] | None,
    timeout: float,
    api_key: str | None,
) -> Any:
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
    response = http_json(
        f"{args.server_url.rstrip('/')}/v1/models",
        payload=None,
        timeout=min(args.timeout, 30.0),
        api_key=args.api_key,
    )
    model_ids = {
        str(item.get("id"))
        for item in (response.get("data") or [])
        if isinstance(item, dict) and item.get("id")
    }
    if args.model not in model_ids:
        raise RuntimeError(
            f"model {args.model!r} is not served at {args.server_url}; "
            f"available models: {sorted(model_ids) or 'none reported'}"
        )


def parse_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def validate_judgement(value: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str | None]:
    if value is None:
        return None, "no JSON object in the final response"
    label = value.get("label")
    if label not in LABELS:
        return None, f"invalid label {label!r}"
    category = value.get("category")
    if isinstance(category, str) and category.lower() == "null":
        category = None
    if label == "degradation":
        if category not in CATEGORIES:
            return None, f"degradation needs one of {sorted(CATEGORIES)}, got {category!r}"
    elif category is not None:
        return None, f"{label} requires category null, got {category!r}"
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
        "confidence": confidence,
        "note": note.strip(),
    }, None


def ask_once(
    args: argparse.Namespace,
    render: Callable[[str, str], str],
    user_prompt: str,
) -> tuple[dict[str, Any] | None, str, str | None]:
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
    response = http_json(
        f"{args.server_url.rstrip('/')}/v1/completions",
        payload=payload,
        timeout=args.timeout,
        api_key=args.api_key,
    )
    choices = response.get("choices") or [{}]
    text = str(choices[0].get("text", "")) if isinstance(choices[0], dict) else ""
    final = text.split(FINAL_MARKER)[-1] if FINAL_MARKER in text else text
    parsed = parse_json_object(final)
    judgement, error = validate_judgement(parsed)
    return judgement, final, error


def judge_candidate(
    args: argparse.Namespace,
    render: Callable[[str, str], str],
    candidate: Candidate,
) -> JudgeResult:
    error: str | None = None
    raw_final = ""
    base_prompt = build_prompt(candidate.row, args.window_tokens)
    for attempt in range(1, args.max_attempts + 1):
        prompt = base_prompt
        if error:
            prompt += (
                "\n\nRETRY INSTRUCTION: The previous response was invalid: "
                f"{error}. Return exactly the requested JSON schema."
            )
        try:
            judgement, raw_final, error = ask_once(args, render, prompt)
        except (OSError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            judgement = None
            error = f"{type(exc).__name__}: {exc}"
            raw_final = ""
        if judgement is not None:
            return JudgeResult(candidate, judgement, raw_final, attempt, None)
        if attempt < args.max_attempts:
            delay = min(30.0, args.retry_delay * 2 ** (attempt - 1))
            time.sleep(delay)
    return JudgeResult(candidate, None, raw_final, args.max_attempts, error)


def output_row(result: JudgeResult, args: argparse.Namespace) -> dict[str, Any]:
    assert result.judgement is not None
    # A judged input can be re-used safely: remove stale judgement fields before
    # attaching this run's validated result and provenance.
    row = {
        key: value
        for key, value in result.candidate.row.items()
        if not key.startswith("judge_")
    }
    row.update(
        {
            "judge_label": result.judgement["label"],
            "judge_category": result.judgement["category"],
            "judge_confidence": result.judgement["confidence"],
            "judge_note": result.judgement["note"],
            "judge_model": args.model,
            "judge_effort": args.judge_effort,
            "judge_prompt_version": PROMPT_VERSION,
            "judge_window_tokens": args.window_tokens,
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


def print_selection_summary(
    total: int,
    runs: int,
    eligible: int,
    pending: int,
    completed: int,
    skipped: Counter[str],
    args: argparse.Namespace,
) -> None:
    print(
        f"input={total:,} runs={runs:,} "
        f"eligible_full_{args.window_tokens}+{args.window_tokens}={eligible:,} "
        f"completed={completed:,} pending={pending:,}"
    )
    if skipped:
        print("excluded: " + ", ".join(f"{key}={value:,}" for key, value in skipped.items()))


def main() -> int:
    args = parse_args()
    if args.window_tokens <= 0:
        raise SystemExit("--window-tokens must be positive")
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
    completed: set[tuple[str, str, str, int]] = set()
    skipped: Counter[str] = Counter()
    total_rows = 0
    for dataset in datasets:
        rows = load_jsonl(dataset.input_path)
        total_rows += len(rows)
        case_candidates, case_skipped = select_candidates(
            rows,
            args.window_tokens,
            dataset.input_sha256,
            dataset.output_path,
        )
        skipped.update(case_skipped)
        case_completed = (
            set()
            if args.dry_run
            else completed_keys(dataset.output_path, args, dataset.input_sha256)
        )
        case_keys = {candidate.key for candidate in case_candidates}
        unknown_completed = case_completed - case_keys
        if unknown_completed:
            raise SystemExit(
                f"refusing to resume {dataset.output_path}: it contains "
                f"{len(unknown_completed)} completed row(s) absent from its input"
            )
        candidates.extend(case_candidates)
        completed.update(case_completed)

    candidates = round_robin_cases(candidates)
    runs = len({candidate.key[:3] for candidate in candidates})
    pending = [candidate for candidate in candidates if candidate.key not in completed]
    if args.limit:
        pending = pending[: args.limit]
    print_selection_summary(
        total_rows, runs, len(candidates), len(pending), len(completed), skipped, args
    )

    if not candidates:
        if skipped.get("incomplete_window"):
            print(
                f"no full windows; re-run extract_lossy_only_tokens.py with "
                f"--context-tokens {args.window_tokens}",
                file=sys.stderr,
            )
        return 2
    if args.dry_run:
        print("\n--- first eligible prompt ---\n")
        print(build_prompt(candidates[0].row, args.window_tokens))
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

    output_paths = sorted({candidate.output_path for candidate in pending})
    with contextlib.ExitStack() as stack:
        handles = {}
        for output_path in output_paths:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if output_path.exists() else "w"
            handles[output_path] = stack.enter_context(
                output_path.open(mode, encoding="utf-8", newline="\n")
            )
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending_iter = iter(pending)
            in_flight: set[concurrent.futures.Future[JudgeResult]] = set()
            for _ in range(min(len(pending), args.workers * 2)):
                in_flight.add(
                    pool.submit(judge_candidate, args, render, next(pending_iter))
                )
            done = 0
            try:
                while in_flight:
                    finished, in_flight = concurrent.futures.wait(
                        in_flight,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in finished:
                        result = future.result()
                        done += 1
                        try:
                            candidate = next(pending_iter)
                        except StopIteration:
                            pass
                        else:
                            in_flight.add(
                                pool.submit(judge_candidate, args, render, candidate)
                            )
                        if result.judgement is None:
                            failed += 1
                            failures.append(result)
                            print(
                                f"failed {result.candidate.key}: {result.error}",
                                file=sys.stderr,
                            )
                        else:
                            handle = handles[result.candidate.output_path]
                            handle.write(
                                json.dumps(
                                    output_row(result, args),
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )
                            handle.flush()
                            succeeded += 1
                        if done % 10 == 0 or done == len(pending):
                            elapsed = time.perf_counter() - started
                            rate = elapsed / done
                            remaining = rate * (len(pending) - done) / 60.0
                            print(
                                f"{done:,}/{len(pending):,} processed; "
                                f"ok={succeeded:,} failed={failed:,}; "
                                f"~{remaining:.1f} min left",
                                flush=True,
                            )
            except KeyboardInterrupt:
                for future in in_flight:
                    future.cancel()
                print(
                    "interrupted; successful rows are saved and the command can resume",
                    file=sys.stderr,
                )
                return 130

    elapsed = time.perf_counter() - started
    labels: Counter[str] = Counter()
    for dataset in datasets:
        if dataset.output_path.is_file() and dataset.output_path.stat().st_size:
            labels.update(
                str(row.get("judge_label"))
                for row in load_jsonl(dataset.output_path)
            )
    print(
        f"wrote {succeeded:,} new judgements across {len(output_paths):,} "
        f"case file(s) in {elapsed / 60.0:.1f} min; "
        + ", ".join(f"{label}={count:,}" for label, count in labels.most_common())
    )
    if failures:
        print(
            f"{len(failures):,} rows failed after {args.max_attempts} attempts; "
            "re-run the same command to retry them",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
