#!/usr/bin/env python3
"""Run and archive paired GPT-OSS generations against SGLang's native API."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any


DEFAULT_PROMPT_ROOT = pathlib.Path("prompts/leval_9k_11k")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send the same archived Harmony prompts to baseline/strict/lossy SGLang servers."
    )
    parser.add_argument("--mode", required=True, choices=("baseline", "strict", "lossy"))
    parser.add_argument("--tag", help="Output directory label; defaults to mode or lossy_l<value>.")
    parser.add_argument("--server-url", default="http://127.0.0.1:30000")
    parser.add_argument("--prompt-root", type=pathlib.Path, default=DEFAULT_PROMPT_ROOT)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Case directory names. Default: metadata entries marked selected_for_pilot.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--runs-root", type=pathlib.Path, default=pathlib.Path("runs"))
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--draft-model", default="nebius/EAGLE3-gpt-oss-20b")
    parser.add_argument(
        "--lenience",
        type=float,
        default=None,
        help="Required for lossy mode; must match server LENIENCE/threshold_acc.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def http_json(url: str, *, payload: dict[str, Any] | None, timeout: float) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="GET" if payload is None else "POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_server_info(base_url: str) -> dict[str, Any]:
    try:
        value = http_json(f"{base_url.rstrip('/')}/get_server_info", payload=None, timeout=30)
        return value if isinstance(value, dict) else {"raw": value}
    except Exception as exc:  # The generation endpoint can still work on older builds.
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def selected_cases(prompt_root: pathlib.Path) -> list[str]:
    index_path = prompt_root / "candidate_index.jsonl"
    selected: list[str] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("selected_for_pilot"):
            selected.append(str(item["case"]))
    if not selected:
        raise ValueError(f"No selected_for_pilot cases in {index_path}")
    return selected


def first_consecutive_repeat(
    tokens: list[int], n_values: tuple[int, ...] = (32, 16, 8), repeats: int = 3
) -> dict[str, int] | None:
    best: dict[str, int] | None = None
    for n in n_values:
        span = n * repeats
        for start in range(0, len(tokens) - span + 1):
            block = tokens[start : start + n]
            if all(tokens[start + i * n : start + (i + 1) * n] == block for i in range(1, repeats)):
                candidate = {"start_token": start, "ngram_tokens": n, "consecutive_repeats": repeats}
                if best is None or candidate["start_token"] < best["start_token"]:
                    best = candidate
                break
    return best


def finish_fields(meta: dict[str, Any]) -> tuple[str | None, bool]:
    finish = meta.get("finish_reason")
    if isinstance(finish, dict):
        finish_type = finish.get("type")
    else:
        finish_type = finish
    eos_reached = finish_type == "stop"
    return finish_type, eos_reached


def safe_tag(args: argparse.Namespace) -> str:
    if args.tag:
        tag = args.tag
    elif args.mode == "lossy":
        if args.lenience is None:
            raise ValueError("--lenience is required in lossy mode")
        tag = f"lossy_l{args.lenience:g}".replace(".", "p")
    else:
        tag = args.mode
    if not tag or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in tag):
        raise ValueError(f"Unsafe tag: {tag!r}")
    return tag


def validate_args(args: argparse.Namespace) -> None:
    if args.temperature <= 0:
        raise ValueError(
            "temperature must be > 0: SGLang's EAGLE verifier uses the greedy path at temperature=0, "
            "where threshold_acc is not applied"
        )
    if args.mode == "lossy" and (args.lenience is None or not 0 < args.lenience < 1):
        raise ValueError("lossy mode requires --lenience in (0, 1)")
    if args.mode != "lossy" and args.lenience is not None:
        raise ValueError("--lenience is only valid with --mode lossy")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")


def run_one(
    args: argparse.Namespace,
    case: str,
    seed: int,
    tag: str,
    server_info: dict[str, Any],
    commit: str | None,
) -> dict[str, Any]:
    case_dir = args.prompt_root / case
    prompt_path = case_dir / "rendered_prompt.txt"
    metadata_path = case_dir / "metadata.json"
    if not prompt_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Incomplete prompt case: {case_dir}")

    prompt = prompt_path.read_text(encoding="utf-8")
    prompt_metadata = read_json(metadata_path)
    output_dir = args.runs_root / case / f"seed_{seed}" / tag
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_dir}; pass --overwrite to replace files")
    output_dir.mkdir(parents=True, exist_ok=True)

    request_payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "sampling_seed": seed,
            "repetition_penalty": 1.0,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
        },
    }

    config = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": commit,
        "mode": args.mode,
        "tag": tag,
        "lossy_method": "lenience" if args.mode == "lossy" else None,
        "lossy_parameters": {"threshold_acc": args.lenience} if args.mode == "lossy" else {},
        "model": args.model,
        "draft_model": None if args.mode == "baseline" else args.draft_model,
        "seed": seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "input_tokens_archived": prompt_metadata.get("input_tokens"),
        "prompt_case": case,
        "prompt_source_id": prompt_metadata.get("source_id"),
        "endpoint": f"{args.server_url.rstrip('/')}/generate",
    }
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "request.json", request_payload)
    write_json(output_dir / "server_info.json", server_info)
    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    started = time.perf_counter()
    try:
        response = http_json(
            f"{args.server_url.rstrip('/')}/generate",
            payload=request_payload,
            timeout=args.timeout,
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        elapsed = time.perf_counter() - started
        error_record = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "wall_time_seconds": elapsed,
        }
        write_json(output_dir / "run.json", error_record)
        raise
    elapsed = time.perf_counter() - started

    write_json(output_dir / "response.json", response)
    output_text = response.get("text", "") if isinstance(response, dict) else ""
    output_ids = response.get("output_ids", []) if isinstance(response, dict) else []
    meta = response.get("meta_info", {}) if isinstance(response, dict) else {}
    if not isinstance(output_ids, list):
        output_ids = []
    if not isinstance(meta, dict):
        meta = {}
    (output_dir / "output.txt").write_text(str(output_text), encoding="utf-8")

    finish_reason, eos_reached = finish_fields(meta)
    repeat = first_consecutive_repeat(output_ids)
    run_record = {
        "status": "ok",
        "wall_time_seconds": elapsed,
        "input_tokens": meta.get("prompt_tokens", prompt_metadata.get("input_tokens")),
        "output_tokens": meta.get("completion_tokens", len(output_ids) if output_ids else None),
        "finish_reason": finish_reason,
        "eos_reached": eos_reached,
        "reached_max_new_tokens": finish_reason == "length",
        "consecutive_repeat_signal": repeat,
        "spec_accept_rate": meta.get("spec_accept_rate"),
        "spec_accept_length": meta.get("spec_accept_length"),
        "spec_verify_ct": meta.get("spec_verify_ct"),
        "meta_info": meta,
    }
    write_json(output_dir / "run.json", run_record)
    print(
        f"{case} seed={seed} mode={tag}: output_tokens={run_record['output_tokens']} "
        f"finish={finish_reason} eos={eos_reached} repeat={repeat} wall={elapsed:.2f}s",
        flush=True,
    )
    return run_record


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        tag = safe_tag(args)
        cases = args.cases or selected_cases(args.prompt_root)
        unknown = [case for case in cases if not (args.prompt_root / case).is_dir()]
        if unknown:
            raise ValueError(f"Unknown cases under {args.prompt_root}: {', '.join(unknown)}")
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    server_info = get_server_info(args.server_url)
    commit = git_commit()
    failures = 0
    for case in cases:
        for seed in args.seeds:
            try:
                run_one(args, case, seed, tag, server_info, commit)
            except Exception as exc:
                failures += 1
                print(f"{case} seed={seed} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
