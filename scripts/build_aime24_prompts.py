#!/usr/bin/env python3
"""Render AIME24 problems into Harmony prompts for the lossy-verification pilot.

Uses the same openai-harmony path as scripts/filter_leval.py so the prompt files
are format-identical to prompts/leval_9k_11k/ and the existing runners work
unchanged. Reasoning effort is medium, not the high used for the L-Eval corpus:
effort moves output length by more than the relaxation under test, so it is
fixed here and recorded in every metadata file.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil

from datasets import load_dataset
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="HuggingFaceH4/aime_2024")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--conversation-date", default="2026-08-01")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("prompts/aime24"))
    parser.add_argument("--replace-output", action="store_true")
    return parser.parse_args()


def make_user_prompt(problem: str) -> str:
    # AIME answers are integers in [0, 999]; asking for the boxed form gives a
    # deterministic target to grade against without constraining the reasoning.
    return (
        f"{problem}\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )


def make_conversation(prompt: str, reasoning_effort: str, conversation_date: str) -> Conversation:
    effort = ReasoningEffort(reasoning_effort.capitalize())
    system = (
        SystemContent.new()
        .with_reasoning_effort(effort)
        .with_conversation_start_date(conversation_date)
    )
    return Conversation.from_messages(
        [
            Message.from_role_and_content(Role.SYSTEM, system),
            Message.from_role_and_content(Role.USER, prompt),
        ]
    )


def field(record: dict, *names: str) -> str:
    for name in names:
        if name in record and record[name] is not None:
            return str(record[name])
    raise KeyError(f"none of {names} in record with keys {sorted(record)}")


def main() -> None:
    args = parse_args()
    if args.output.exists():
        if not args.replace_output:
            raise SystemExit(f"Output already exists: {args.output} (use --replace-output)")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    dataset = load_dataset(args.dataset, split=args.split)

    index_rows = []
    for position in range(min(args.limit, len(dataset))):
        record = dataset[position]
        problem = field(record, "problem", "question")
        answer = field(record, "answer", "solution")
        problem_id = field(record, "id", "problem_id") if ("id" in record or "problem_id" in record) else str(position)

        user_prompt = make_user_prompt(problem)
        conversation = make_conversation(
            user_prompt,
            reasoning_effort=args.reasoning_effort,
            conversation_date=args.conversation_date,
        )
        tokens = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
        rendered = encoding.decode(tokens)

        case = f"case_{position + 1:03d}"
        case_dir = args.output / case
        case_dir.mkdir()

        metadata = {
            "source": "AIME24",
            "source_dataset": args.dataset,
            "source_split": args.split,
            "source_id": f"{args.dataset}:{args.split}:{position}",
            "problem_id": problem_id,
            "position": position,
            "tokenizer": "o200k_harmony",
            "harmony_encoding": "HARMONY_GPT_OSS",
            "reasoning_effort": args.reasoning_effort,
            "conversation_start_date": args.conversation_date,
            "input_tokens": len(tokens),
            "reference_answer": answer,
            "selected_for_pilot": True,
        }
        (case_dir / "rendered_prompt.txt").write_text(rendered, encoding="utf-8")
        (case_dir / "token_count.txt").write_text(f"{len(tokens)}\n", encoding="utf-8")
        (case_dir / "reference_output.txt").write_text(f"{answer}\n", encoding="utf-8")
        (case_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (case_dir / "source.json").write_text(
            json.dumps({"problem": problem, "answer": answer}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        index_rows.append({"case": case, **metadata})
        print(f"{case}: {len(tokens):>5} input tokens  answer={answer}")

    with (args.output / "candidate_index.jsonl").open("w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = [row["input_tokens"] for row in index_rows]
    (args.output / "selection_summary.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "split": args.split,
                "selected": len(index_rows),
                "reasoning_effort": args.reasoning_effort,
                "conversation_start_date": args.conversation_date,
                "tokenizer": "o200k_harmony",
                "harmony_encoding": "HARMONY_GPT_OSS",
                "min_input_tokens": min(counts),
                "max_input_tokens": max(counts),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {len(index_rows)} cases to {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
