#!/usr/bin/env python3
"""Select LongBench-Chat prompts by exact GPT-OSS Harmony token count."""

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
        "--input",
        type=Path,
        default=Path("data/long_context/longbench_chat/test_cases.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("prompts/longbench_chat_9k_11k"),
    )
    parser.add_argument("--min-tokens", type=int, default=9_000)
    parser.add_argument("--max-tokens", type=int, default=11_000)
    parser.add_argument("--target-tokens", type=int, default=10_000)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="high")
    parser.add_argument("--conversation-date", default="2026-08-01")
    parser.add_argument(
        "--replace-output",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
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

    samples = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise SystemExit(f"Expected a JSON array in {args.input}")

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    selected: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    for position, sample in enumerate(samples):
        prompt = sample.get("prompt")
        if not isinstance(prompt, str):
            raise SystemExit(f"Sample at position {position} has no string 'prompt'")

        conversation = make_conversation(
            prompt,
            reasoning_effort=args.reasoning_effort,
            conversation_date=args.conversation_date,
        )
        tokens = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
        token_count = len(tokens)
        all_rows.append(
            {
                "source_id": sample.get("idx", position),
                "source_position": position,
                "input_tokens": token_count,
                "distance_from_target": abs(token_count - args.target_tokens),
                "query": sample.get("query"),
            }
        )

        if args.min_tokens <= token_count <= args.max_tokens:
            selected.append(
                {
                    "source": "LongBench-Chat",
                    "source_file": args.input.as_posix(),
                    "source_id": sample.get("idx", position),
                    "source_position": position,
                    "input_tokens": token_count,
                    "distance_from_target": abs(token_count - args.target_tokens),
                    "query": sample.get("query"),
                    "answer": sample.get("answer"),
                    "prompt": prompt,
                    "tokens": tokens,
                    "source_record": sample,
                }
            )

    selected.sort(key=lambda row: (row["distance_from_target"], row["source_position"]))
    args.output.mkdir(parents=True)

    with (args.output / "all_token_counts.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as count_file:
        for row in sorted(all_rows, key=lambda item: item["source_position"]):
            count_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    index_path = args.output / "candidate_index.jsonl"
    with index_path.open("w", encoding="utf-8", newline="\n") as index_file:
        for rank, row in enumerate(selected, start=1):
            case_name = f"case_{rank:03d}"
            case_dir = args.output / case_name
            case_dir.mkdir()

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
                {"role": "user", "content": row["prompt"]},
            ]
            metadata = {
                "case": case_name,
                "source": row["source"],
                "source_file": row["source_file"],
                "source_id": row["source_id"],
                "source_position": row["source_position"],
                "tokenizer": "o200k_harmony",
                "harmony_encoding": "HARMONY_GPT_OSS",
                "reasoning_effort": args.reasoning_effort,
                "conversation_start_date": args.conversation_date,
                "input_tokens": row["input_tokens"],
                "target_tokens": args.target_tokens,
                "distance_from_target": row["distance_from_target"],
                "query": row["query"],
                "reference_answer": row["answer"],
            }

            json_dump(case_dir / "source.json", row["source_record"])
            json_dump(case_dir / "messages.json", messages)
            json_dump(case_dir / "metadata.json", metadata)
            (case_dir / "rendered_prompt.txt").write_text(
                encoding.decode(row["tokens"]), encoding="utf-8", newline="\n"
            )
            (case_dir / "token_count.txt").write_text(
                f"{row['input_tokens']}\n", encoding="ascii", newline="\n"
            )

            index_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")

    counts = [row["input_tokens"] for row in all_rows]
    nearest = sorted(all_rows, key=lambda row: row["distance_from_target"])[:5]
    summary = {
        "source": args.input.as_posix(),
        "total_samples": len(samples),
        "selected_samples": len(selected),
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "target_tokens": args.target_tokens,
        "tokenizer": "o200k_harmony",
        "harmony_encoding": "HARMONY_GPT_OSS",
        "reasoning_effort": args.reasoning_effort,
        "conversation_start_date": args.conversation_date,
        "observed_min_tokens": min(counts),
        "observed_median_tokens": statistics.median(counts),
        "observed_max_tokens": max(counts),
        "nearest_to_target": nearest,
    }
    json_dump(args.output / "selection_summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for rank, row in enumerate(selected, start=1):
        print(f"case_{rank:03d}\t{row['input_tokens']} tokens\tsource_id={row['source_id']}")


if __name__ == "__main__":
    main()
