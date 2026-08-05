#!/usr/bin/env python3
"""Render MT-Bench first-turn prompts into Harmony prompts for trace collection.

Mirrors scripts/build_aime24_prompts.py's structure and artifact contract.
The lmsys/mt-bench Space is a judging UI, not raw data; HuggingFaceH4/mt_bench_prompts
is the canonical 80-question set it's built from (category, two-turn prompt
sequence, optional reference answers for math/reasoning/coding).

Only the first turn is rendered: this repo's harness (run_experiment_vllm.py)
is single-turn, and turn 2 depends on the model's own turn-1 response, which
would need a second request per case. MT-Bench's value here is category
diversity (writing/roleplay/math/coding/extraction/stem/humanities/reasoning),
not the multi-turn mechanic specifically.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="HuggingFaceH4/mt_bench_prompts")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--categories", nargs="+", default=None,
                         help="Restrict to these categories; default keeps all 8.")
    parser.add_argument("--limit", type=int, default=0, help="Number of prompts to render; 0 renders the whole (filtered) split.")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--conversation-date", default="2026-08-01")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("prompts/mtbench"))
    parser.add_argument("--replace-output", action="store_true")
    parser.add_argument("--rows-json", type=pathlib.Path, default=None)
    parser.add_argument("--save-rows-json", type=pathlib.Path, default=None)
    return parser.parse_args()


def load_rows(args: argparse.Namespace) -> list[dict]:
    if args.rows_json is not None:
        text = args.rows_json.read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        query = urllib.parse.urlencode(
            {"dataset": args.dataset, "config": args.config, "split": args.split, "offset": 0, "length": 100}
        )
        url = f"https://datasets-server.huggingface.co/rows?{query}"
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if args.save_rows_json:
            args.save_rows_json.parent.mkdir(parents=True, exist_ok=True)
            args.save_rows_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if isinstance(payload, dict) and "rows" in payload:
        total = payload.get("num_rows_total")
        rows = [entry["row"] if isinstance(entry, dict) and "row" in entry else entry for entry in payload["rows"]]
        if total is not None and len(rows) < total:
            raise SystemExit(f"fetched {len(rows)} of {total} rows; paginate before rendering a partial split")
        return rows
    if isinstance(payload, list):
        return payload
    raise SystemExit(f"unrecognised row source: {args.rows_json or args.dataset}")


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
    rows = load_rows(args)
    print(f"fetched {len(rows)} rows from {args.dataset}")

    if args.categories:
        wanted = {c.lower() for c in args.categories}
        rows = [r for r in rows if str(r.get("category", "")).lower() in wanted]
        print(f"{len(rows)} rows in categories {sorted(wanted)}")

    count = len(rows) if args.limit <= 0 else min(args.limit, len(rows))
    index_rows = []
    for position in range(count):
        record = rows[position]
        turns = record.get("prompt") or []
        if not turns:
            continue
        first_turn = str(turns[0])
        references = record.get("reference") or []

        conversation = make_conversation(first_turn, args.reasoning_effort, args.conversation_date)
        tokens = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
        rendered = encoding.decode(tokens)

        case = f"case_{position + 1:03d}"
        case_dir = args.output / case
        case_dir.mkdir()

        metadata = {
            "source": "MT-Bench",
            "source_dataset": args.dataset,
            "source_split": args.split,
            "source_id": f"{args.dataset}:{args.split}:{record.get('prompt_id')}",
            "category": record.get("category"),
            "prompt_id": record.get("prompt_id"),
            "second_turn_prompt": str(turns[1]) if len(turns) > 1 else None,
            "tokenizer": "o200k_harmony",
            "harmony_encoding": "HARMONY_GPT_OSS",
            "reasoning_effort": args.reasoning_effort,
            "conversation_start_date": args.conversation_date,
            "input_tokens": len(tokens),
            "reference_answer": references[0] if references else None,
            "selected_for_pilot": True,
        }
        (case_dir / "rendered_prompt.txt").write_text(rendered, encoding="utf-8")
        (case_dir / "token_count.txt").write_text(f"{len(tokens)}\n", encoding="utf-8")
        (case_dir / "reference_output.txt").write_text(f"{references[0] if references else ''}\n", encoding="utf-8")
        (case_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (case_dir / "source.json").write_text(
            json.dumps({"problem": first_turn, "answer": references[0] if references else None}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        index_rows.append({"case": case, **metadata})
        print(f"{case}: {len(tokens):>5} input tokens  category={record.get('category')}")

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
                "categories": args.categories,
                "selected": len(index_rows),
                "reasoning_effort": args.reasoning_effort,
                "tokenizer": "o200k_harmony",
                "harmony_encoding": "HARMONY_GPT_OSS",
                "min_input_tokens": min(counts) if counts else None,
                "max_input_tokens": max(counts) if counts else None,
            },
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {len(index_rows)} cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
