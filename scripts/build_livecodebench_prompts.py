#!/usr/bin/env python3
"""Render LiveCodeBench problems into Harmony prompts for trace collection.

Mirrors scripts/build_aime24_prompts.py's structure and artifact contract.

livecodebench/code_generation_lite ships a custom loading script, so the HF
datasets-server preview/rows API (used by the other build_*_prompts.py
scripts) refuses it outright. The raw data is plain JSONL directly in the
repo (test.jsonl, ~1.25GB), but almost all of that size is
public/private_test_cases -- base64-compressed I/O blobs this script never
needs, since it only renders the problem statement. Range-fetching the first
--fetch-bytes of the file (default 20MB, comfortably dozens of complete rows
given each is a few KB before its test cases) avoids the full download.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import urllib.request

from openai_harmony import (
    Conversation,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
    load_harmony_encoding,
)

RAW_URL = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/{filename}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--filename", default="test.jsonl", help="Which release file to sample from (test.jsonl is the earliest/smallest).")
    parser.add_argument("--fetch-bytes", type=int, default=20_000_000, help="Byte range to download from the start of the file.")
    parser.add_argument("--limit", type=int, default=12, help="Number of cases to render; 0 renders every complete row fetched.")
    parser.add_argument("--difficulty", nargs="+", default=None, help="Restrict to these difficulties (easy/medium/hard); default keeps all.")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--conversation-date", default="2026-08-01")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("prompts/livecodebench"))
    parser.add_argument("--replace-output", action="store_true")
    parser.add_argument("--rows-json", type=pathlib.Path, default=None, help="Read rows from this file instead of fetching them.")
    parser.add_argument("--save-rows-json", type=pathlib.Path, default=None)
    return parser.parse_args()


def fetch_rows(args: argparse.Namespace) -> list[dict]:
    if args.rows_json is not None:
        lines = args.rows_json.read_text(encoding="utf-8").splitlines()
    else:
        url = RAW_URL.format(filename=args.filename)
        request = urllib.request.Request(url, headers={"Range": f"bytes=0-{args.fetch_bytes - 1}"})
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read()
        text = raw.decode("utf-8", errors="ignore")
        lines = text.splitlines()
        if args.save_rows_json:
            args.save_rows_json.parent.mkdir(parents=True, exist_ok=True)
            args.save_rows_json.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break  # a range fetch's last line is very likely mid-row; complete rows come first
    return rows


def make_user_prompt(row: dict) -> str:
    starter = str(row.get("starter_code") or "").strip()
    if starter:
        return (
            f"{row['question_content']}\n\n"
            f"Complete the following function:\n```python\n{starter}\n```\n\n"
            "Please reason step by step, then provide the complete solution code in a "
            "```python ... ``` block."
        )
    return (
        f"{row['question_content']}\n\n"
        "Write a complete Python program that reads input from stdin and writes output "
        "to stdout.\n\nPlease reason step by step, then provide the complete solution "
        "code in a ```python ... ``` block."
    )


def make_conversation(prompt: str, reasoning_effort: str, conversation_date: str) -> Conversation:
    effort = ReasoningEffort(reasoning_effort.capitalize())
    system = SystemContent.new().with_reasoning_effort(effort).with_conversation_start_date(conversation_date)
    return Conversation.from_messages(
        [Message.from_role_and_content(Role.SYSTEM, system), Message.from_role_and_content(Role.USER, prompt)]
    )


def main() -> int:
    args = parse_args()
    if args.output.exists():
        if not args.replace_output:
            raise SystemExit(f"Output already exists: {args.output} (use --replace-output)")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    rows = fetch_rows(args)
    print(f"fetched {len(rows)} complete rows from {args.filename}")

    if args.difficulty:
        wanted = {d.lower() for d in args.difficulty}
        rows = [r for r in rows if str(r.get("difficulty", "")).lower() in wanted]
        print(f"{len(rows)} rows with difficulty in {sorted(wanted)}")

    count = len(rows) if args.limit <= 0 else min(args.limit, len(rows))
    index_rows = []
    for position in range(count):
        row = rows[position]
        if not row.get("question_content"):
            continue
        user_prompt = make_user_prompt(row)
        conversation = make_conversation(user_prompt, args.reasoning_effort, args.conversation_date)
        tokens = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
        rendered = encoding.decode(tokens)

        case = f"case_{position + 1:03d}"
        case_dir = args.output / case
        case_dir.mkdir()

        metadata = {
            "source": "LiveCodeBench",
            "source_dataset": "livecodebench/code_generation_lite",
            "source_file": args.filename,
            "source_id": f"livecodebench/code_generation_lite:{args.filename}:{row.get('question_id')}",
            "question_id": row.get("question_id"),
            "question_title": row.get("question_title"),
            "platform": row.get("platform"),
            "difficulty": row.get("difficulty"),
            "contest_date": row.get("contest_date"),
            "tokenizer": "o200k_harmony",
            "harmony_encoding": "HARMONY_GPT_OSS",
            "reasoning_effort": args.reasoning_effort,
            "conversation_start_date": args.conversation_date,
            "input_tokens": len(tokens),
            "reference_answer": None,
            "selected_for_pilot": True,
        }
        (case_dir / "rendered_prompt.txt").write_text(rendered, encoding="utf-8")
        (case_dir / "token_count.txt").write_text(f"{len(tokens)}\n", encoding="utf-8")
        (case_dir / "reference_output.txt").write_text("", encoding="utf-8")
        (case_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (case_dir / "source.json").write_text(
            json.dumps({"problem": user_prompt, "answer": None}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        index_rows.append({"case": case, **metadata})
        print(f"{case}: {len(tokens):>5} input tokens  {row.get('platform')}/{row.get('difficulty')} {row.get('question_id')}")

    if not index_rows:
        raise SystemExit("no rows selected -- check --difficulty filter or increase --fetch-bytes")

    with (args.output / "candidate_index.jsonl").open("w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = [row["input_tokens"] for row in index_rows]
    (args.output / "selection_summary.json").write_text(
        json.dumps(
            {
                "dataset": "livecodebench/code_generation_lite",
                "filename": args.filename,
                "difficulty": args.difficulty,
                "selected": len(index_rows),
                "reasoning_effort": args.reasoning_effort,
                "tokenizer": "o200k_harmony",
                "harmony_encoding": "HARMONY_GPT_OSS",
                "min_input_tokens": min(counts),
                "max_input_tokens": max(counts),
            },
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {len(index_rows)} cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
