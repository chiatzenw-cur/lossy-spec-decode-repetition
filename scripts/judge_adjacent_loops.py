#!/usr/bin/env python3
"""LLM-judge candidate ADJACENT (back-to-back, uninterrupted) exact-repeat loops.

Consumes the consolidated loop-event JSON produced by extract_repetition_clusters.py
+ a gap==0 adjacency filter + interval-merge (see the analysis this session ran
inline) -- NOT a fresh extraction. Each event already carries an EXACT token
position (loop_token_start/loop_token_end), recovered algorithmically from the
shingle-match + union-find detector in extract_repetition_clusters.py, which in
turn comes from lib_trace_align.align()'s byte-exact alignment between
proposals.jsonl and output.txt.

This is the key design choice that answers "how do I get the token position
from an LLM judgement": don't. Never ask the LLM to report a position or quote
to resolve -- positions are found by the ALGORITHMIC step, which is exact by
construction (token-id equality), before the LLM ever sees the text. The LLM's
only job here is the qualitative call an algorithm can't make: is this
particular already-located adjacent repeat actually pathological (a stuck
loop with no new information) or legitimate (e.g. a numbered list, correct
repeated boilerplate, a short connector phrase that's supposed to recur)?
Each judged row is the SAME row the extractor produced, with a verdict field
appended -- position and judgement are never re-joined after the fact.

The judge server must be non-speculative (no --speculative-config): the judge
model must not itself be verified against a drafter while it labels text
produced under speculative decoding, to keep the two concerns separate.
Start one with:

    PYTHON=$PWD/.venv-vllm/bin/python bash remote/run_server_vllm.sh baseline
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable

FINAL_MARKER = "<|channel|>final<|message|>"

RUBRIC = """You are auditing a candidate ADJACENT REPEATED PASSAGE found by an automated
scanner in a language model's output. The scanner works purely mechanically
(exact token-id matching): it found that an identical run of text repeats
immediately back-to-back, {chain_repeat_count} times in a row, with NOTHING
else in between the repeats. It has no understanding of the task; most
mechanical hits ARE genuine stuck loops (that is what "adjacent, uninterrupted,
exact" almost always means), but a few are legitimate -- e.g. a numbered list
whose separators repeat, or correct boilerplate code that is meant to look
like this.

REPEATED UNIT (shown once; this exact text recurs {chain_repeat_count} times
back-to-back with no other content between copies):
---
{unit_text}
---

CONTEXT BEFORE the loop starts:
---
{context_before}
---

CONTEXT AFTER the loop ends (what came right after, if anything -- empty means
the loop was still running when this excerpt was cut off, e.g. the generation
hit its token budget mid-loop):
---
{context_after}
---

Task: [1] mark this repeat, and [2] classify why.
Respond with a JSON object and nothing else:

{{"verdict": "abnormal" | "legitimate" | "ambiguous",
  "category": "verbatim_loop" | "re_derivation" | "list_or_boilerplate" | "other" | null,
  "reasoning": "one or two sentences"}}

verdict=abnormal: the model is stuck producing the same text with no new
  information -- reasoning stalled, not progressing toward an answer.
verdict=legitimate: the repeated text is doing real, intended work (e.g. a
  correctly repeated separator/marker in a list or code structure).
verdict=ambiguous: genuinely unclear from this excerpt alone.
category is required when verdict=abnormal, null otherwise.
  verbatim_loop: same sentence/clause repeated with no variation.
  re_derivation: repeatedly restating the same computed value/step.
  list_or_boilerplate: repeats a structural marker but still stuck (rare --
    most list/boilerplate repeats are legitimate, only mark this if the
    repeat count is clearly excessive for the structure it claims to be).
  other: abnormal but doesn't fit the above.
"""


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
        raise RuntimeError(f"model {args.model!r} is not served at {args.server_url}; available: {sorted(model_ids) or 'none'}")
    # Refuse a speculative-decoding server: the whole point of this judge is a
    # plain baseline verifier, not one itself being verified against a drafter.
    props = http_json(f"{args.server_url.rstrip('/')}/v1/models", payload=None, timeout=min(args.timeout, 30.0), api_key=args.api_key)
    _ = props  # vLLM's /v1/models does not report spec-decode config; see --skip-server-check to bypass this check entirely if needed.


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


VERDICTS = {"abnormal", "legitimate", "ambiguous"}
CATEGORIES = {"verbatim_loop", "re_derivation", "list_or_boilerplate", "other"}


def validate_response(value: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str | None]:
    if value is None:
        return None, "no JSON object in the final response"
    verdict = value.get("verdict")
    if verdict not in VERDICTS:
        return None, f"verdict must be one of {sorted(VERDICTS)}, got {verdict!r}"
    category = value.get("category")
    if isinstance(category, str) and category.lower() == "null":
        category = None
    if verdict == "abnormal":
        if category not in CATEGORIES:
            return None, f"verdict=abnormal needs category in {sorted(CATEGORIES)}, got {category!r}"
    elif category is not None:
        return None, f"verdict={verdict} requires category null, got {category!r}"
    reasoning = value.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        return None, "reasoning must be a non-empty string"
    return {"verdict": verdict, "category": category, "reasoning": reasoning.strip()}, None


def build_context(raw: bytes, records: list[dict], start_idx: int, end_idx: int, context_tokens: int) -> dict[str, str]:
    """start_idx/end_idx: 0-based, inclusive-exclusive over records (loop span)."""
    left_offset = max(0, start_idx - context_tokens)
    right_offset = min(len(records), end_idx + context_tokens)
    left_byte = records[left_offset]["byte_start"]
    right_byte = records[right_offset - 1]["byte_end"]
    match_byte_start = records[start_idx]["byte_start"]
    match_byte_end = records[end_idx - 1]["byte_end"]
    full = raw[left_byte:right_byte].decode("utf-8", errors="replace").replace("\r", "")
    match_start_rel = len(raw[left_byte:match_byte_start].decode("utf-8", errors="replace"))
    match_end_rel = len(raw[left_byte:match_byte_end].decode("utf-8", errors="replace"))
    return {
        "context_before": full[:match_start_rel],
        "unit_text": full[match_start_rel:match_end_rel],
        "context_after": full[match_end_rel:],
    }


def load_run_records(run_dir: pathlib.Path) -> tuple[bytes | None, list[dict] | None]:
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from lib_trace_align import align  # noqa: E402
    return align(run_dir)


def judge_one(args: argparse.Namespace, render: Callable[[str, str], str], event: dict[str, Any], unit_text: str, context_before: str, context_after: str) -> dict[str, Any]:
    prompt = RUBRIC.format(
        chain_repeat_count=event["chain_repeat_count"],
        unit_text=unit_text[:2000],
        context_before=context_before[-1500:],
        context_after=context_after[:800] or "(nothing -- excerpt ends here, e.g. generation was cut off mid-loop)",
    )
    error: str | None = None
    parsed: dict[str, Any] | None = None
    raw_final = ""
    for attempt in range(1, args.max_attempts + 1):
        p = prompt
        if error:
            p += f"\n\nRETRY INSTRUCTION: previous response invalid: {error}. Return exactly the requested JSON schema, nothing else."
        try:
            payload = {
                "model": args.model,
                "prompt": render(p, args.judge_effort),
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
            raw_final = text.split(FINAL_MARKER)[-1] if FINAL_MARKER in text else text
            value = parse_json_object(raw_final)
            parsed, error = validate_response(value)
        except (OSError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            parsed = None
            error = f"{type(exc).__name__}: {exc}"
        if parsed is not None:
            break
        if attempt < args.max_attempts:
            time.sleep(min(30.0, args.retry_delay * 2 ** (attempt - 1)))

    row = dict(event)
    row.pop("cluster_id", None)
    row["unit_text_preview"] = unit_text[:300]
    if parsed is not None:
        row.update(parsed)
        row["judge_error"] = None
    else:
        row["verdict"] = None
        row["category"] = None
        row["reasoning"] = None
        row["judge_error"] = error or "unknown"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("events_json", type=pathlib.Path, help="Consolidated loop-event JSON (list of dicts with case/tag/loop_token_start/loop_token_end/...).")
    parser.add_argument("--runs-root", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--context-tokens", type=int, default=60)
    parser.add_argument("--server-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="gpt-oss-20b")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--judge-effort", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--skip-server-check", action="store_true")
    args = parser.parse_args()

    events = json.loads(args.events_json.read_text())
    if not events:
        raise SystemExit(f"no events in {args.events_json}")

    if not args.skip_server_check:
        check_server(args)
    render = harmony_renderer()

    # Cache raw/records per (case, tag) run dir -- many events share a run.
    run_cache: dict[tuple[str, str], tuple[bytes, list[dict]]] = {}

    def prepare(event: dict[str, Any]) -> tuple[dict[str, Any], str, str, str] | None:
        key = (event["case"], event["tag"])
        if key not in run_cache:
            run_dir = args.runs_root / event["case"] / "seed_0" / event["tag"]
            raw, records = load_run_records(run_dir)
            if raw is None or records is None:
                return None
            run_cache[key] = (raw, records)
        raw, records = run_cache[key]
        start_idx = event["loop_token_start"] - 1  # token_index is 1-based
        end_idx = event["loop_token_end"]  # exclusive end == last token_index (already 1-based -> exclusive)
        ctx = build_context(raw, records, start_idx, end_idx, args.context_tokens)
        return event, ctx["unit_text"], ctx["context_before"], ctx["context_after"]

    prepared = []
    for e in events:
        p = prepare(e)
        if p is not None:
            prepared.append(p)
    print(f"prepared {len(prepared)}/{len(events)} events (missing run dirs skipped)")

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(judge_one, args, render, e, unit, before, after) for e, unit, before, after in prepared]
        for i, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
            row = fut.result()
            results.append(row)
            if i % 20 == 0 or i == len(futures):
                print(f"  judged {i}/{len(futures)}")

    results.sort(key=lambda r: (r["case"], r["loop_token_start"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    verdicts = Counter(r["verdict"] for r in results)
    print(f"\nwrote {len(results)} judged rows to {args.out}")
    print(f"verdicts: {dict(verdicts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
