#!/usr/bin/env python3
"""Scan whole output traces for repetitive reasoning, with the LLM as detector.

extract_repetition_clusters.py / judge_repetition_clusters.py take the
opposite approach: an algorithmic exact-token-match pass finds candidates
across the whole trace, then the LLM classifies each isolated pair. On real
traces that flooded the candidate set with legitimate reuse -- restating a
derived value while naming a variable, moving to the next step -- and the LLM
had to relitigate the same "is this actually fine" call thousands of times
with only a two-excerpt view. In an 8-row live sample, 75% of algorithmic hits
were benign despite being exact repeats, and no algorithmic threshold (gap
size, match length) separated them from the genuine stuck loops.

This script instead has the LLM read each trace itself, in the order it was
generated, and flag repetitive-reasoning sections directly -- the same way a
human skimming the trace would notice "it's stuck here." That gives it real
narrative context to judge necessity, and it can catch near-duplicate
(paraphrased) loops the exact-matcher is blind to. The tradeoff: a ~20B judge
cannot reliably compare text across a 30k-token span in one read, so this
only sees repetition within a bounded window -- which is where the actual
degenerate loops live (9 of 30 lenience-arm cases hit the 32,768-token
generation cap, which looks like local stuck loops, not sparse distant
echoes; distant echoes were the mostly-benign restatements above).

Wall-time design: the trace is walked in non-overlapping OWNED windows
(--step-tokens each) with some earlier OWNED text carried in as leading
CONTEXT (--lookback-tokens) so the judge can recognize "I already did this."
Every token is screened by exactly one chunk (its owner), so there is no
double-counting across chunks despite the context overlap. This means far
fewer, larger, more information-dense calls than the pair-matching approach
(~450 vs ~11,500 across the full case set at the defaults), which more than
offsets the larger per-call prompt.

Onset resolution follows the same verbatim-quote discipline as
scripts/record_label.py and judge_repetition_clusters.py: the judge must
return onset_quote as an exact substring of the shown text, which this script
locates and maps to a token_index -- never a token offset the judge would
have to count itself.

Run every case in ``runs/aime24_fresh``::

    python scripts/scan_repetitive_sections.py

Preview chunking and the first prompt without calling the server::

    python scripts/scan_repetitive_sections.py --dry-run

The endpoint must be a plain, non-speculative GPT-OSS-20B vLLM server::

    PYTHON=$PWD/.venv-vllm/bin/python bash remote/run_server_vllm.sh baseline
    python scripts/scan_repetitive_sections.py
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
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lib_trace_align import align  # noqa: E402


DEFAULT_RUNS_ROOT = pathlib.Path("runs/aime24_fresh")
DEFAULT_PROMPT_ROOT = pathlib.Path("prompts/aime24")
FINAL_MARKER = "<|channel|>final<|message|>"
PROMPT_VERSION = "repetitive-section-scan-v1"
LABELS = {"unproductive_repetition", "necessary_repetition", "ambiguous"}
CATEGORIES = {"verbatim_loop", "re_derivation", "unresolved_cycle", "other"}
MIN_QUOTE_CHARS = 6
MAX_QUOTE_CHARS = 140

RUBRIC = """You are screening ONE CHUNK of a language model's chain-of-thought on a
competition maths problem, read in the order it was generated. The chunk may
begin with earlier material shown as CONTEXT, followed by a NEW SECTION --
you are screening ONLY the NEW SECTION for repetitive reasoning, using the
context (and earlier parts of the NEW SECTION itself) as what has already
been established.

Find every stretch in the NEW SECTION where the model re-derives, re-states,
or re-checks something it has already done, without making new progress.
Most things that look repetitive at a glance are actually fine and must NOT
be flagged: a deliberate one-off check, restating a value while naming a new
variable or moving to the next step, a brief recap before the final answer,
or exploring a genuinely different case that happens to look similar. Flag
only a stretch where the model is stuck: cycling through the same
derivation, claim, or small set of candidate values again with no visible
progress.

For every stretch you flag, quote 6-15 consecutive WORDS from the NEW SECTION
marking where that stretch begins -- never more than one sentence, and never
the whole stretch. Do not deliberate over how much to include: the first
words of the repeat are always enough, even if the repetition continues well
past them. Copy those words EXACTLY (character for character, including
punctuation and spacing). Never quote from the CONTEXT block -- only from the
NEW SECTION.

LABELS
- unproductive_repetition: this stretch mostly repeats prior work with no
  new progress -- a stalled derivation, a stuck loop, or restating the same
  claim again for no visible reason.
- necessary_repetition: this stretch reuses earlier content but is doing new
  work with it (a deliberate one-off check, a definition, a summary before
  the final answer).
- ambiguous: you can tell something is being repeated but cannot tell from
  what's shown whether it is stuck or purposeful.

CATEGORY (use null unless label is unproductive_repetition)
- verbatim_loop: the same short passage repeats with essentially no new
  surrounding content.
- re_derivation: the full computation reruns from scratch to reach a value
  already established, instead of reusing it.
- unresolved_cycle: this is one step of a back-and-forth between the same
  small set of candidate values or approaches that never converges.
- other: a concrete repetition-driven waste of effort not covered above.

If the NEW SECTION contains no repetitive reasoning at all, report an empty
list.

Reply with ONE JSON object and nothing else:
{"sections":[{"onset_quote":"...","label":"unproductive_repetition|necessary_repetition|ambiguous",
"category":null,"confidence":0.0,"note":"one concise sentence"}]}
For category, use exactly one of the listed strings when label is
unproductive_repetition, otherwise null. If there is nothing to flag, reply
{"sections":[]}."""


@dataclass(frozen=True)
class Chunk:
    case: str
    seed: str
    tag: str
    chunk_index: int
    context_text: str
    new_section_text: str
    token_boundaries: list[list[int]]  # [rel_char_offset, token_index], over context+new_section
    window_token_start: int
    window_token_end: int
    owned_token_start: int
    owned_token_end: int
    run_sha256: str
    output_path: pathlib.Path

    @property
    def key(self) -> tuple[str, str, str, int]:
        return (self.case, self.seed, self.tag, self.chunk_index)


@dataclass(frozen=True)
class JudgeResult:
    chunk: Chunk
    sections: list[dict[str, Any]] | None
    raw_final: str
    attempts: int
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cases = parser.add_mutually_exclusive_group()
    cases.add_argument("--case", dest="cases", action="append", type=lambda v: normalize_case(v),
                        help="Case to scan, e.g. 004 or case_004. Repeat for multiple cases. Default: all cases.")
    parser.add_argument("--runs-root", type=pathlib.Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--prompt-root", type=pathlib.Path, default=DEFAULT_PROMPT_ROOT,
                         help="Directory of case_NNN/source.json problem statements.")
    parser.add_argument("--run-seed", default="seed_0", help="Run seed directory, or 'all'.")
    parser.add_argument("--tag", default="lenience0p2", help="Run arm directory to scan.")
    parser.add_argument("--step-tokens", type=int, default=1000,
                         help="Tokens each chunk owns (screened for onsets); the trace is fully covered "
                              "by non-overlapping owned windows.")
    parser.add_argument("--lookback-tokens", type=int, default=1000,
                         help="Earlier owned tokens carried in as CONTEXT so the judge can recognize reuse.")
    parser.add_argument("--work-dir", type=pathlib.Path, help="Root containing case_NNN output directories.")
    parser.add_argument("--server-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="gpt-oss-20b")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"), help="Optional bearer token.")
    parser.add_argument("--judge-effort", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=0, help="Judge at most N pending chunks; 0 means all.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Build the chunk list and print counts/first prompt; do not call the judge.")
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
        pathlib.Path("data/repetitive_sections")
        / safe_component(args.tag)
        / safe_component(seed_scope)
        / f"step_{args.step_tokens:04d}_lookback_{args.lookback_tokens:04d}"
    )


def output_path_for_case(args: argparse.Namespace, case: str) -> pathlib.Path:
    work_dir = args.work_dir or default_work_dir(args)
    return work_dir / case / "judgements" / safe_component(args.model) / f"{args.judge_effort}.jsonl"


def discover_runs(args: argparse.Namespace) -> list[pathlib.Path]:
    # output.txt, not proposals.jsonl: the strict arm has no per-token proposal
    # trace (that file only exists for the speculative-decoding arms), and this
    # script only needs the committed text, not accept/reject metadata -- see
    # load_run's direct-tokenize fallback for arms without a trace.
    seed_pattern = "seed_*" if args.run_seed.lower() == "all" else args.run_seed
    if args.cases:
        runs = []
        for case in sorted(set(args.cases)):
            runs.extend(sorted((args.runs_root / case).glob(f"{seed_pattern}/{args.tag}")))
        return [run for run in runs if (run / "output.txt").is_file()]
    pattern = f"case_*/{seed_pattern}/{args.tag}/output.txt"
    return sorted(p.parent for p in args.runs_root.glob(pattern))


def direct_tokenize(run_dir: pathlib.Path) -> tuple[bytes | None, list[dict] | None]:
    """Fallback for runs.jsonl without a proposals.jsonl (e.g. the strict arm):
    tokenize output.txt directly instead of aligning to a per-token trace.

    Token boundaries need not match the original generation's token stream
    exactly -- this script only uses them to define chunk/context byte
    spans, not to recover per-token accept/reject metadata -- so a fresh,
    self-consistent tokenization of the committed text is sufficient.
    """
    import tiktoken

    out_path = run_dir / "output.txt"
    if not out_path.is_file():
        return None, None
    enc = tiktoken.get_encoding("o200k_harmony")
    raw = out_path.read_bytes()
    ids = enc.encode(raw.decode("utf-8", errors="replace"), allowed_special="all")
    records = []
    cursor = 0
    for k, token_id in enumerate(ids):
        piece = enc.decode_single_token_bytes(token_id)
        records.append(
            {
                "emitted_token_id": token_id,
                "token_index": k + 1,
                "byte_start": cursor,
                "byte_end": cursor + len(piece),
                "text": piece.decode("utf-8", errors="replace"),
            }
        )
        cursor += len(piece)
    return raw, records


def load_run(run_dir: pathlib.Path) -> tuple[bytes | None, list[dict] | None]:
    if (run_dir / "proposals.jsonl").is_file():
        return align(run_dir)
    return direct_tokenize(run_dir)


# --- byte-exact text helpers (mirrors extract_lossy_only_tokens.py / extract_repetition_clusters.py) --


def char_position(raw: bytes, byte_position: int) -> int:
    return len(raw[:byte_position].decode("utf-8", errors="replace"))


def decode_span(raw: bytes, start: int, end: int) -> str:
    return raw[start:end].decode("utf-8", errors="replace").replace("\r", "")


def iter_chunk_bounds(n_records: int, step_tokens: int, lookback_tokens: int):
    """Yield (window_start_idx, owned_start_idx, owned_end_idx), 0-based record indices.

    owned ranges partition [0, n_records) exactly once each; window_start_idx
    only pulls in extra leading context, never extends the owned range.
    """
    owned_start = 0
    while owned_start < n_records:
        owned_end = min(n_records, owned_start + step_tokens)
        window_start = max(0, owned_start - lookback_tokens)
        yield window_start, owned_start, owned_end
        owned_start = owned_end


def build_chunk(
    case: str, seed: str, tag: str, chunk_index: int,
    raw: bytes, records: list[dict], window_start_idx: int, owned_start_idx: int, owned_end_idx: int,
    run_sha256: str, output_path: pathlib.Path,
) -> Chunk:
    window_left_byte = records[window_start_idx]["byte_start"]
    window_right_byte = records[owned_end_idx - 1]["byte_end"]
    base_char = char_position(raw, window_left_byte)

    owned_byte_start = records[owned_start_idx]["byte_start"]
    rel_owned_start = char_position(raw, owned_byte_start) - base_char
    full_text = decode_span(raw, window_left_byte, window_right_byte)

    token_boundaries = [
        [char_position(raw, records[i]["byte_start"]) - base_char, int(records[i]["token_index"])]
        for i in range(window_start_idx, owned_end_idx)
    ]

    return Chunk(
        case=case, seed=seed, tag=tag, chunk_index=chunk_index,
        context_text=full_text[:rel_owned_start],
        new_section_text=full_text[rel_owned_start:],
        token_boundaries=token_boundaries,
        window_token_start=int(records[window_start_idx]["token_index"]),
        window_token_end=int(records[owned_end_idx - 1]["token_index"]),
        owned_token_start=int(records[owned_start_idx]["token_index"]),
        owned_token_end=int(records[owned_end_idx - 1]["token_index"]),
        run_sha256=run_sha256,
        output_path=output_path,
    )


def build_run_chunks(run_dir: pathlib.Path, args: argparse.Namespace) -> list[Chunk]:
    raw, records = load_run(run_dir)
    if raw is None or records is None:
        raise RuntimeError(f"cannot read/tokenize run: {run_dir}")
    case, seed, tag = run_dir.parts[-3:]
    run_sha256 = hashlib.sha256(raw).hexdigest()
    output_path = output_path_for_case(args, case)
    chunks = []
    for chunk_index, (window_start_idx, owned_start_idx, owned_end_idx) in enumerate(
        iter_chunk_bounds(len(records), args.step_tokens, args.lookback_tokens)
    ):
        chunks.append(
            build_chunk(
                case, seed, tag, chunk_index, raw, records,
                window_start_idx, owned_start_idx, owned_end_idx, run_sha256, output_path,
            )
        )
    return chunks


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


def build_prompt(chunk: Chunk, prompt_root: pathlib.Path) -> str:
    problem = load_problem(prompt_root, chunk.case)
    context_block = (
        f"CONTEXT (earlier material; do not flag anything in here):\n<<<\n{chunk.context_text}\n>>>\n\n"
        if chunk.context_text
        else ""
    )
    return (
        f"{RUBRIC}\n\nPROBLEM:\n{problem}\n\n"
        f"{context_block}"
        f"NEW SECTION (screen this for repetitive reasoning):\n<<<\n{chunk.new_section_text}\n>>>"
    )


def _whitespace_tolerant_find(text: str, words: list[str]) -> int:
    if not words:
        return -1
    pattern = r"\s+".join(re.escape(word) for word in words)
    match = re.search(pattern, text)
    return match.start() if match else -1


def find_quote(text: str, quote: str, min_words: int = 3) -> int:
    """Locate quote in text, tolerating whitespace differences and a
    hallucinated/paraphrased tail.

    The judge reliably reproduces the FIRST few words of a quote from real
    context, then sometimes drifts into paraphrase or blends in nearby text
    as it continues -- observed directly: "So 110 is less. Wait 110 seems
    smaller." reproduced as "110 is less. So 110 is smaller." (reordered),
    and a quote that silently spliced in a different nearby sentence. Both
    had an exact, correctly-placed prefix. So after a full-quote match fails
    (exact, then whitespace-tolerant), progressively drop trailing words and
    retry -- this uses whatever prefix the judge actually got right instead
    of discarding the whole quote over a drift near the end, while a
    genuinely wrong quote (bad prefix) still fails to match at any length.
    """
    pos = text.find(quote)
    if pos >= 0:
        return pos
    words = quote.split()
    pos = _whitespace_tolerant_find(text, words)
    if pos >= 0:
        return pos
    for n in range(len(words) - 1, min_words - 1, -1):
        pos = _whitespace_tolerant_find(text, words[:n])
        if pos >= 0:
            return pos
    return -1


def resolve_onset_token(chunk: Chunk, quote: str) -> int | None:
    text = chunk.context_text + chunk.new_section_text
    pos = find_quote(text, quote)
    if pos < 0:
        return None
    offsets = [b[0] for b in chunk.token_boundaries]
    idx = bisect.bisect_right(offsets, pos) - 1
    if idx < 0:
        idx = 0
    token_index = int(chunk.token_boundaries[idx][1])
    if token_index < chunk.owned_token_start:
        return None  # resolved into the CONTEXT zone; the judge must only flag the NEW SECTION
    return token_index


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


def validate_response(value: dict[str, Any] | None, chunk: Chunk) -> tuple[list[dict[str, Any]] | None, str | None]:
    if value is None:
        return None, "no JSON object in the final response"
    sections = value.get("sections")
    if not isinstance(sections, list):
        return None, f"sections must be a list, got {type(sections).__name__}"

    parsed: list[dict[str, Any]] = []
    for i, item in enumerate(sections):
        if not isinstance(item, dict):
            return None, f"sections[{i}] must be an object"
        label = item.get("label")
        if label not in LABELS:
            return None, f"sections[{i}]: invalid label {label!r}"
        category = item.get("category")
        if isinstance(category, str) and category.lower() == "null":
            category = None
        if label == "unproductive_repetition":
            if category not in CATEGORIES:
                return None, f"sections[{i}]: unproductive_repetition needs one of {sorted(CATEGORIES)}, got {category!r}"
        elif category is not None:
            return None, f"sections[{i}]: {label} requires category null, got {category!r}"

        onset_quote = item.get("onset_quote")
        if not isinstance(onset_quote, str) or len(onset_quote.strip()) < MIN_QUOTE_CHARS:
            return None, f"sections[{i}]: onset_quote must be a string of at least {MIN_QUOTE_CHARS} characters"
        if len(onset_quote) > MAX_QUOTE_CHARS:
            return None, (
                f"sections[{i}]: onset_quote is {len(onset_quote)} characters, over the "
                f"{MAX_QUOTE_CHARS}-character limit; quote only the first 6-15 words of where "
                "the repeat begins, not the whole stretch"
            )
        onset_token_index = resolve_onset_token(chunk, onset_quote)
        if onset_token_index is None:
            return None, f"sections[{i}]: onset_quote not found verbatim inside the NEW SECTION: {onset_quote!r}"

        confidence = item.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            return None, f"sections[{i}]: confidence must be numeric, got {confidence!r}"
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            return None, f"sections[{i}]: confidence must be in [0, 1], got {confidence}"

        note = item.get("note")
        if not isinstance(note, str) or not note.strip():
            return None, f"sections[{i}]: note must be a non-empty string"

        parsed.append(
            {
                "label": label, "category": category, "onset_quote": onset_quote,
                "onset_token_index": onset_token_index, "confidence": confidence, "note": note.strip(),
            }
        )
    return parsed, None


def ask_once(args: argparse.Namespace, render: Callable[[str, str], str], chunk: Chunk, user_prompt: str) -> tuple[list[dict[str, Any]] | None, str, str | None]:
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
    sections, error = validate_response(parsed, chunk)
    return sections, final, error


def judge_chunk(args: argparse.Namespace, render: Callable[[str, str], str], prompt_root: pathlib.Path, chunk: Chunk) -> JudgeResult:
    error: str | None = None
    raw_final = ""
    base_prompt = build_prompt(chunk, prompt_root)
    for attempt in range(1, args.max_attempts + 1):
        prompt = base_prompt
        if error:
            prompt += f"\n\nRETRY INSTRUCTION: The previous response was invalid: {error}. Return exactly the requested JSON schema."
        try:
            sections, raw_final, error = ask_once(args, render, chunk, prompt)
        except (OSError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            sections = None
            error = f"{type(exc).__name__}: {exc}"
            raw_final = ""
        if sections is not None:
            return JudgeResult(chunk, sections, raw_final, attempt, None)
        if attempt < args.max_attempts:
            time.sleep(min(30.0, args.retry_delay * 2 ** (attempt - 1)))
    return JudgeResult(chunk, None, raw_final, args.max_attempts, error)


def output_row(result: JudgeResult, args: argparse.Namespace) -> dict[str, Any]:
    assert result.sections is not None
    chunk = result.chunk
    return {
        "case": chunk.case,
        "seed": chunk.seed,
        "tag": chunk.tag,
        "chunk_index": chunk.chunk_index,
        "window_token_start": chunk.window_token_start,
        "window_token_end": chunk.window_token_end,
        "owned_token_start": chunk.owned_token_start,
        "owned_token_end": chunk.owned_token_end,
        "sections": result.sections,
        "judge_model": args.model,
        "judge_effort": args.judge_effort,
        "judge_prompt_version": PROMPT_VERSION,
        "judge_step_tokens": args.step_tokens,
        "judge_lookback_tokens": args.lookback_tokens,
        "judge_temperature": args.temperature,
        "judge_seed": args.seed,
        "judge_run_sha256": chunk.run_sha256,
        "judge_attempts": result.attempts,
        "judge_parse_ok": True,
        "judge_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "judge_raw_final": result.raw_final[:4000],
    }


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            rows.append(value)
    return rows


def compatible_provenance(row: dict[str, Any], args: argparse.Namespace, run_sha256: str) -> bool:
    return (
        row.get("judge_prompt_version") == PROMPT_VERSION
        and row.get("judge_model") == args.model
        and row.get("judge_effort") == args.judge_effort
        and row.get("judge_step_tokens") == args.step_tokens
        and row.get("judge_lookback_tokens") == args.lookback_tokens
        and row.get("judge_temperature") == args.temperature
        and row.get("judge_seed") == args.seed
        and row.get("judge_run_sha256") == run_sha256
        and row.get("judge_parse_ok") is True
    )


def completed_chunk_indices(path: pathlib.Path, args: argparse.Namespace, run_sha256: str) -> set[int]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    rows = load_jsonl(path)
    compatible: set[int] = set()
    incompatible = 0
    for row in rows:
        if compatible_provenance(row, args, run_sha256) and isinstance(row.get("chunk_index"), int):
            compatible.add(row["chunk_index"])
        else:
            incompatible += 1
    if incompatible:
        raise SystemExit(
            f"refusing to append to {path}: found {incompatible} row(s) not produced "
            "by this model/prompt/chunking configuration; choose a new --work-dir"
        )
    return compatible


def round_robin_cases(chunks: list[Chunk]) -> list[Chunk]:
    buckets: dict[tuple[str, str, str], deque[Chunk]] = defaultdict(deque)
    for chunk in chunks:
        buckets[(chunk.case, chunk.seed, chunk.tag)].append(chunk)
    ordered: list[Chunk] = []
    active = sorted(buckets)
    while active:
        next_active = []
        for key in active:
            ordered.append(buckets[key].popleft())
            if buckets[key]:
                next_active.append(key)
        active = next_active
    return ordered


def main() -> int:
    args = parse_args()
    if args.step_tokens <= 0:
        raise SystemExit("--step-tokens must be positive")
    if args.lookback_tokens < 0:
        raise SystemExit("--lookback-tokens must be nonnegative")
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

    run_dirs = discover_runs(args)
    if not run_dirs:
        raise SystemExit(f"no matching proposal runs under {args.runs_root} for seed={args.run_seed} tag={args.tag}")

    all_chunks: list[Chunk] = []
    completed: set[tuple[str, str, str, int]] = set()
    unreadable_runs: list[tuple[pathlib.Path, str]] = []
    for run_dir in run_dirs:
        try:
            chunks = build_run_chunks(run_dir, args)
        except (RuntimeError, OSError, json.JSONDecodeError, KeyError) as exc:
            # One case's proposals.jsonl not aligning to output.txt (seen on a
            # LongBench-v2 run: only the trailing ~2/3 of the text matched a
            # suffix of any token-id decoding, an alignment failure lib_trace_align
            # cannot recover from) must not take down every other case's scan --
            # they are independent, and losing 51 good runs over 1 bad one is a
            # worse outcome than skipping the bad one and saying so.
            unreadable_runs.append((run_dir, f"{type(exc).__name__}: {exc}"))
            print(f"warning: skipping unreadable run {run_dir}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if not chunks:
            continue
        run_sha256 = chunks[0].run_sha256
        output_path = chunks[0].output_path
        case_completed = set() if args.dry_run else completed_chunk_indices(output_path, args, run_sha256)
        valid_indices = {c.chunk_index for c in chunks}
        unknown = case_completed - valid_indices
        if unknown:
            raise SystemExit(f"refusing to resume {output_path}: it contains {len(unknown)} completed chunk(s) absent from the current chunking")
        for c in chunks:
            if c.chunk_index in case_completed:
                completed.add(c.key)
        all_chunks.extend(chunks)

    all_chunks = round_robin_cases(all_chunks)
    runs = len({c.key[:3] for c in all_chunks})
    pending = [c for c in all_chunks if c.key not in completed]
    if args.limit:
        pending = pending[: args.limit]

    print(f"chunks={len(all_chunks):,} runs={runs:,} completed={len(completed):,} pending={len(pending):,}")
    if unreadable_runs:
        print(f"skipped {len(unreadable_runs)} unreadable run(s) out of {len(run_dirs)}: "
              + ", ".join(str(p) for p, _ in unreadable_runs), file=sys.stderr)
    if not all_chunks:
        return 2
    if args.dry_run:
        print("\n--- first prompt ---\n")
        print(build_prompt(all_chunks[0], args.prompt_root))
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
                in_flight.add(pool.submit(judge_chunk, args, render, args.prompt_root, next(pending_iter)))
            done = 0
            try:
                while in_flight:
                    finished, in_flight = concurrent.futures.wait(in_flight, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in finished:
                        result = future.result()
                        done += 1
                        try:
                            nxt = next(pending_iter)
                        except StopIteration:
                            pass
                        else:
                            in_flight.add(pool.submit(judge_chunk, args, render, args.prompt_root, nxt))
                        if result.sections is None:
                            failed += 1
                            failures.append(result)
                            print(f"failed {result.chunk.key}: {result.error}", file=sys.stderr)
                        else:
                            handle = handles[result.chunk.output_path]
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
    flagged: Counter[str] = Counter()
    total_sections = 0
    for output_path in output_paths:
        if output_path.is_file() and output_path.stat().st_size:
            for row in load_jsonl(output_path):
                for section in row.get("sections") or []:
                    flagged[str(section.get("label"))] += 1
                    total_sections += 1
    print(
        f"wrote {succeeded:,} judged chunks across {len(output_paths):,} case file(s) in {elapsed / 60.0:.1f} min; "
        f"{total_sections:,} flagged sections: " + ", ".join(f"{label}={count:,}" for label, count in flagged.most_common())
    )
    if failures:
        print(f"{len(failures):,} chunks failed after {args.max_attempts} attempts; re-run the same command to retry them", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
