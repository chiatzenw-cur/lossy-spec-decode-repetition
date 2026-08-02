#!/usr/bin/env python3
"""Select L-Eval prompts by exact GPT-OSS Harmony token count."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
from pathlib import Path
from typing import Any

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
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=[
            Path("data/long_context/leval/paper_assistant.jsonl"),
            Path("data/long_context/leval/multidoc_qa.jsonl"),
            Path("data/long_context/leval/scientific_qa.jsonl"),
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("prompts/leval_9k_11k"),
    )
    parser.add_argument("--min-tokens", type=int, default=9_000)
    parser.add_argument("--max-tokens", type=int, default=11_000)
    parser.add_argument("--target-tokens", type=int, default=10_000)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="high")
    parser.add_argument("--conversation-date", default="2026-08-01")
    parser.add_argument("--pilot-count", type=int, default=5)
    parser.add_argument("--replace-output", action="store_true")
    return parser.parse_args()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def make_user_prompt(document: str, instruction: str, output: str) -> str:
    # This is the open-ended prompt format from L-Eval's official
    # Baselines/turbo16k-test.py, preserved so the source task remains recognizable.
    suggested_words = len(output.split())
    return (
        f"Document is as follows. {document} Instruction: {instruction} "
        f"The suggested output length is around {suggested_words} words. Output: "
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


def main() -> None:
    args = parse_args()
    if args.min_tokens > args.max_tokens:
        raise SystemExit("--min-tokens must not exceed --max-tokens")
    if args.output.exists():
        if not args.replace_output:
            raise SystemExit(f"Output already exists: {args.output} (use --replace-output)")
        shutil.rmtree(args.output)

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    all_rows: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []

    for input_path in args.inputs:
        for document_position, line in enumerate(
            input_path.read_text(encoding="utf-8").splitlines()
        ):
            record = json.loads(line)
            for instruction_position, (instruction, output) in enumerate(
                zip(record["instructions"], record["outputs"])
            ):
                user_prompt = make_user_prompt(record["input"], instruction, output)
                conversation = make_conversation(
                    user_prompt,
                    reasoning_effort=args.reasoning_effort,
                    conversation_date=args.conversation_date,
                )
                tokens = encoding.render_conversation_for_completion(
                    conversation, Role.ASSISTANT
                )
                token_count = len(tokens)
                source_id = f"{input_path.stem}:{document_position}:{instruction_position}"
                row = {
                    "source": "L-Eval",
                    "source_file": input_path.as_posix(),
                    "source_task": input_path.stem,
                    "source_id": source_id,
                    "document_position": document_position,
                    "instruction_position": instruction_position,
                    "input_tokens": token_count,
                    "distance_from_target": abs(token_count - args.target_tokens),
                    "instruction": instruction,
                    "reference_output": output,
                    "reference_output_words": len(output.split()),
                    "evaluation": record.get("evaluation"),
                    "user_prompt": user_prompt,
                    "tokens": tokens,
                    "source_record": record,
                }
                all_rows.append(row)
                if args.min_tokens <= token_count <= args.max_tokens:
                    selected.append(row)

    # Longer expected answers are more useful for provoking generation degeneration.
    # Distance from 10k breaks ties without discarding otherwise valid candidates.
    selected.sort(
        key=lambda row: (-row["reference_output_words"], row["distance_from_target"])
    )
    args.output.mkdir(parents=True)

    with (args.output / "all_token_counts.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as count_file:
        for row in all_rows:
            compact = {key: value for key, value in row.items() if key not in {"tokens", "source_record", "user_prompt", "reference_output"}}
            count_file.write(json.dumps(compact, ensure_ascii=False) + "\n")

    with (args.output / "candidate_index.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as index_file:
        for rank, row in enumerate(selected, start=1):
            case_name = f"case_{rank:03d}"
            case_dir = args.output / case_name
            case_dir.mkdir()
            selected_for_pilot = rank <= args.pilot_count
            metadata = {
                "case": case_name,
                "source": row["source"],
                "source_file": row["source_file"],
                "source_task": row["source_task"],
                "source_id": row["source_id"],
                "document_position": row["document_position"],
                "instruction_position": row["instruction_position"],
                "tokenizer": "o200k_harmony",
                "harmony_encoding": "HARMONY_GPT_OSS",
                "reasoning_effort": args.reasoning_effort,
                "conversation_start_date": args.conversation_date,
                "input_tokens": row["input_tokens"],
                "target_tokens": args.target_tokens,
                "distance_from_target": row["distance_from_target"],
                "instruction": row["instruction"],
                "reference_output_words": row["reference_output_words"],
                "evaluation": row["evaluation"],
                "selected_for_pilot": selected_for_pilot,
            }
            messages = [
                {
                    "role": "system",
                    "content": {
                        "reasoning_effort": args.reasoning_effort,
                        "conversation_start_date": args.conversation_date,
                        "knowledge_cutoff": "2024-06",
                        "valid_channels": ["analysis", "commentary", "final"],
                    },
                },
                {"role": "user", "content": row["user_prompt"]},
            ]

            json_dump(case_dir / "source.json", row["source_record"])
            json_dump(case_dir / "messages.json", messages)
            json_dump(case_dir / "metadata.json", metadata)
            (case_dir / "reference_output.txt").write_text(
                row["reference_output"] + "\n", encoding="utf-8", newline="\n"
            )
            (case_dir / "rendered_prompt.txt").write_text(
                encoding.decode(row["tokens"]), encoding="utf-8", newline="\n"
            )
            (case_dir / "token_count.txt").write_text(
                f"{row['input_tokens']}\n", encoding="ascii", newline="\n"
            )
            index_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")

    counts = [row["input_tokens"] for row in all_rows]
    summary = {
        "sources": [path.as_posix() for path in args.inputs],
        "total_instruction_prompts": len(all_rows),
        "selected_prompts": len(selected),
        "selected_for_pilot": min(args.pilot_count, len(selected)),
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "target_tokens": args.target_tokens,
        "observed_min_tokens": min(counts),
        "observed_median_tokens": statistics.median(counts),
        "observed_max_tokens": max(counts),
        "tokenizer": "o200k_harmony",
        "harmony_encoding": "HARMONY_GPT_OSS",
        "reasoning_effort": args.reasoning_effort,
        "conversation_start_date": args.conversation_date,
        "prompt_format_source": "L-Eval/Baselines/turbo16k-test.py open-ended format",
    }
    json_dump(args.output / "selection_summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for rank, row in enumerate(selected, start=1):
        marker = "pilot" if rank <= args.pilot_count else "reserve"
        print(
            f"case_{rank:03d}\t{row['input_tokens']} tokens\t"
            f"{row['reference_output_words']} reference words\t{marker}\t{row['source_id']}"
        )


if __name__ == "__main__":
    main()
