#!/usr/bin/env python3
"""Render LongBench-v2 questions into Harmony prompts for trace collection.

Mirrors scripts/build_aime24_prompts.py's structure and artifact contract
(rendered_prompt.txt / metadata.json / source.json / candidate_index.jsonl) so
run_experiment_vllm.py and scripts/fresh_server_replay.py work unchanged.

LongBench-v2 is a long-context multiple-choice benchmark (zai-org/LongBench-v2):
context + question + 4 choices, single correct answer. Contexts range from a
few thousand to well over a million tokens (the dataset's own "length" field
buckets rows as short/medium/long); the server here runs with
--max-model-len 65536, so only rows small enough to leave real room for
reasoning output are usable. This script fetches the "short" bucket (already
the smallest) and further filters by actual Harmony-encoded token count.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import urllib.parse
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

CHOICE_KEYS = ("choice_A", "choice_B", "choice_C", "choice_D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="zai-org/LongBench-v2")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--length-bucket", default="short", choices=("short", "medium", "long", "any"))
    parser.add_argument("--max-input-tokens", type=int, default=45000,
                         help="Upper bound on rendered-prompt tokens, leaving room under --max-model-len 65536 for reasoning output.")
    parser.add_argument("--limit", type=int, default=12, help="Number of qualifying cases to render; 0 renders all that fit.")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--conversation-date", default="2026-08-01")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("prompts/longbench_v2"))
    parser.add_argument("--replace-output", action="store_true")
    parser.add_argument("--rows-json", type=pathlib.Path, default=None,
                         help="Read rows from this file instead of fetching them.")
    parser.add_argument("--save-rows-json", type=pathlib.Path, default=None,
                         help="Archive the fetched rows here so the build can be repeated offline.")
    return parser.parse_args()


def fetch_all_rows(args: argparse.Namespace) -> list[dict]:
    if args.rows_json is not None:
        text = args.rows_json.read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        rows = payload["rows"] if isinstance(payload, dict) and "rows" in payload else payload
        return [r["row"] if isinstance(r, dict) and "row" in r else r for r in rows]

    rows: list[dict] = []
    offset = 0
    page = 100
    total = None
    while total is None or offset < total:
        query = urllib.parse.urlencode(
            {"dataset": args.dataset, "config": args.config, "split": args.split, "offset": offset, "length": page}
        )
        url = f"https://datasets-server.huggingface.co/rows?{query}"
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        total = payload.get("num_rows_total")
        batch = [entry["row"] if isinstance(entry, dict) and "row" in entry else entry for entry in payload["rows"]]
        rows.extend(batch)
        offset += len(batch)
        if not batch:
            break
    if args.save_rows_json:
        args.save_rows_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_rows_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rows


def make_user_prompt(row: dict) -> str:
    choices = "\n".join(f"{letter}. {row[f'choice_{letter}']}" for letter in "ABCD")
    return (
        f"{row['context']}\n\n"
        f"Question: {row['question']}\n\n"
        f"{choices}\n\n"
        "Please reason step by step, then give your final answer as a single "
        "letter (A, B, C, or D) within \\boxed{}."
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
    rows = fetch_all_rows(args)
    print(f"fetched {len(rows)} rows from {args.dataset}")

    if args.length_bucket != "any":
        rows = [r for r in rows if r.get("length") == args.length_bucket]
        print(f"{len(rows)} rows in length bucket {args.length_bucket!r}")

    index_rows = []
    position = 0
    for row in rows:
        if args.limit and len(index_rows) >= args.limit:
            break
        if not all(row.get(k) for k in CHOICE_KEYS) or not row.get("context") or not row.get("question"):
            continue
        user_prompt = make_user_prompt(row)
        conversation = make_conversation(user_prompt, args.reasoning_effort, args.conversation_date)
        tokens = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
        if len(tokens) > args.max_input_tokens:
            continue
        rendered = encoding.decode(tokens)

        position += 1
        case = f"case_{position:03d}"
        case_dir = args.output / case
        case_dir.mkdir()

        metadata = {
            "source": "LongBench-v2",
            "source_dataset": args.dataset,
            "source_split": args.split,
            "source_id": f"{args.dataset}:{args.split}:{row.get('_id')}",
            "domain": row.get("domain"),
            "sub_domain": row.get("sub_domain"),
            "difficulty": row.get("difficulty"),
            "length_bucket": row.get("length"),
            "tokenizer": "o200k_harmony",
            "harmony_encoding": "HARMONY_GPT_OSS",
            "reasoning_effort": args.reasoning_effort,
            "conversation_start_date": args.conversation_date,
            "input_tokens": len(tokens),
            "reference_answer": row.get("answer"),
            "selected_for_pilot": True,
        }
        (case_dir / "rendered_prompt.txt").write_text(rendered, encoding="utf-8")
        (case_dir / "token_count.txt").write_text(f"{len(tokens)}\n", encoding="utf-8")
        (case_dir / "reference_output.txt").write_text(f"{row.get('answer')}\n", encoding="utf-8")
        (case_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (case_dir / "source.json").write_text(
            json.dumps(
                {
                    "problem": f"{row['question']}\n\n" + "\n".join(f"{l}. {row[f'choice_{l}']}" for l in "ABCD"),
                    "answer": row.get("answer"),
                },
                ensure_ascii=False, indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        index_rows.append({"case": case, **metadata})
        print(f"{case}: {len(tokens):>6} input tokens  domain={row.get('domain')!r} answer={row.get('answer')}")

    if not index_rows:
        raise SystemExit(f"no rows fit under --max-input-tokens {args.max_input_tokens}")

    with (args.output / "candidate_index.jsonl").open("w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = [row["input_tokens"] for row in index_rows]
    (args.output / "selection_summary.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "config": args.config,
                "split": args.split,
                "length_bucket": args.length_bucket,
                "max_input_tokens": args.max_input_tokens,
                "selected": len(index_rows),
                "reasoning_effort": args.reasoning_effort,
                "tokenizer": "o200k_harmony",
                "harmony_encoding": "HARMONY_GPT_OSS",
                "min_input_tokens": min(counts),
                "max_input_tokens_actual": max(counts),
            },
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {len(index_rows)} cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
