#!/usr/bin/env python3
"""Run and archive paired GPT-OSS generations against vLLM's OpenAI-compatible API.

Mirrors scripts/run_lossy_experiment.py (SGLang) and writes the same artifact
contract, so scripts/summarize_runs.py works unchanged across both backends.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any


DEFAULT_PROMPT_ROOT = pathlib.Path("prompts/leval_9k_11k")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("baseline", "strict", "lossy"))
    parser.add_argument("--tag", help="Output directory label; defaults to mode or lossy_a<value>.")
    parser.add_argument("--server-url", default="http://127.0.0.1:30000")
    parser.add_argument("--prompt-root", type=pathlib.Path, default=DEFAULT_PROMPT_ROOT)
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--runs-root", type=pathlib.Path, default=pathlib.Path("runs"))
    parser.add_argument("--model", default="gpt-oss-20b", help="Served model name.")
    parser.add_argument("--draft-model", default="nebius/EAGLE3-gpt-oss-20b")
    parser.add_argument(
        "--synthetic-acceptance-length",
        type=float,
        default=None,
        help="Required in lossy mode; must match the server's SYNTH_LEN.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


def spec_counters(base_url: str) -> dict[str, float]:
    """Cumulative speculative-decode counters from /metrics.

    vLLM reports these per engine, not per request, so a request's own counts
    come from differencing a snapshot taken either side of it. Valid only while
    requests are issued one at a time.
    """
    wanted = {
        "vllm:spec_decode_num_draft_tokens_total": "draft_tokens",
        "vllm:spec_decode_num_accepted_tokens_total": "accepted_tokens",
        "vllm:spec_decode_num_drafts_total": "drafts",
    }
    out: dict[str, float] = {}
    try:
        request = urllib.request.Request(f"{base_url.rstrip('/')}/metrics")
        text = urllib.request.urlopen(request, timeout=30).read().decode("utf-8")
    except Exception:
        return out
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for metric, key in wanted.items():
            if line.startswith(metric):
                try:
                    out[key] = float(line.rsplit(" ", 1)[1])
                except (ValueError, IndexError):
                    pass
    return out


def acceptance_stats(before: dict[str, float], after: dict[str, float]) -> dict[str, Any]:
    """Per-request acceptance, from the counter delta.

    l_bar is mean accepted DRAFT tokens per verification round, so the mean
    accepted length including the always-kept bonus token is l_bar + 1.
    """
    drafted = after.get("draft_tokens", 0.0) - before.get("draft_tokens", 0.0)
    accepted = after.get("accepted_tokens", 0.0) - before.get("accepted_tokens", 0.0)
    drafts = after.get("drafts", 0.0) - before.get("drafts", 0.0)
    stats: dict[str, Any] = {
        "draft_tokens": drafted or None,
        "accepted_tokens": accepted or None,
        "draft_rounds": drafts or None,
        "draft_acceptance_rate": (accepted / drafted) if drafted else None,
        "l_bar": (accepted / drafts) if drafts else None,
    }
    stats["mean_accept_length"] = (stats["l_bar"] + 1) if stats["l_bar"] is not None else None
    return stats


def server_info(base_url: str) -> dict[str, Any]:
    """vLLM has no /get_server_info; record what the OpenAI surface exposes."""
    info: dict[str, Any] = {}
    for name, path in (("models", "/v1/models"), ("version", "/version")):
        try:
            info[name] = http_json(f"{base_url.rstrip('/')}{path}", payload=None, timeout=30)
        except Exception as exc:
            info[name] = {"unavailable": f"{type(exc).__name__}: {exc}"}
    return info


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


def safe_tag(args: argparse.Namespace) -> str:
    if args.tag:
        tag = args.tag
    elif args.mode == "lossy":
        if args.synthetic_acceptance_length is None:
            raise ValueError("--synthetic-acceptance-length is required in lossy mode")
        tag = f"lossy_a{args.synthetic_acceptance_length:g}".replace(".", "p")
    else:
        tag = args.mode
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    if not tag or any(ch not in allowed for ch in tag):
        raise ValueError(f"Unsafe tag: {tag!r}")
    return tag


def validate_args(args: argparse.Namespace) -> None:
    if args.temperature <= 0:
        raise ValueError(
            "temperature must be > 0: at temperature 0 the verifier takes a greedy path and the "
            "probabilistic acceptance rule under test is not exercised"
        )
    if args.mode == "lossy" and args.synthetic_acceptance_length is None:
        raise ValueError("lossy mode requires --synthetic-acceptance-length")
    if args.mode != "lossy" and args.synthetic_acceptance_length is not None:
        raise ValueError("--synthetic-acceptance-length is only valid with --mode lossy")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")


def run_one(
    args: argparse.Namespace,
    case: str,
    seed: int,
    tag: str,
    info: dict[str, Any],
    commit: str | None,
) -> dict[str, Any]:
    case_dir = args.prompt_root / case
    prompt_path = case_dir / "rendered_prompt.txt"
    metadata_path = case_dir / "metadata.json"
    if not prompt_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Incomplete prompt case: {case_dir}")

    prompt = prompt_path.read_text(encoding="utf-8")
    prompt_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    output_dir = args.runs_root / case / f"seed_{seed}" / tag
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_dir}; pass --overwrite to replace files")
    output_dir.mkdir(parents=True, exist_ok=True)

    request_payload = {
        "model": args.model,
        "prompt": prompt,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_new_tokens,
        "seed": seed,
        "repetition_penalty": 1.0,
        # The archived prompts are already rendered Harmony text carrying their own
        # special tokens; letting the tokenizer add more would change the input.
        "add_special_tokens": False,
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        "stream": False,
    }

    config = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": commit,
        "backend": "vllm",
        "mode": args.mode,
        "tag": tag,
        "lossy_method": "synthetic_acceptance" if args.mode == "lossy" else None,
        "lossy_parameters": (
            {"synthetic_acceptance_length": args.synthetic_acceptance_length}
            if args.mode == "lossy"
            else {}
        ),
        "model": args.model,
        "draft_model": None if args.mode == "baseline" else args.draft_model,
        "seed": seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "input_tokens_archived": prompt_metadata.get("input_tokens"),
        "prompt_case": case,
        "prompt_source_id": prompt_metadata.get("source_id"),
        "endpoint": f"{args.server_url.rstrip('/')}/v1/completions",
    }
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "request.json", request_payload)
    write_json(output_dir / "server_info.json", info)
    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    counters_before = spec_counters(args.server_url)
    started = time.perf_counter()
    try:
        response = http_json(
            f"{args.server_url.rstrip('/')}/v1/completions",
            payload=request_payload,
            timeout=args.timeout,
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        elapsed = time.perf_counter() - started
        write_json(
            output_dir / "run.json",
            {"status": "error", "error": f"{type(exc).__name__}: {exc}", "wall_time_seconds": elapsed},
        )
        raise
    elapsed = time.perf_counter() - started
    spec = acceptance_stats(counters_before, spec_counters(args.server_url))

    write_json(output_dir / "response.json", response)
    choices = response.get("choices") or [{}]
    choice = choices[0] if isinstance(choices[0], dict) else {}
    output_text = choice.get("text", "")
    usage = response.get("usage") or {}
    finish_reason = choice.get("finish_reason")
    (output_dir / "output.txt").write_text(str(output_text), encoding="utf-8")

    # vLLM returns token ids only when echo/logprobs are requested; the repeat
    # detector falls back to logprob token ids when they are present.
    logprobs = choice.get("logprobs") or {}
    output_ids = logprobs.get("tokens") if isinstance(logprobs, dict) else None
    output_ids = output_ids if isinstance(output_ids, list) else []

    run_record = {
        "status": "ok",
        "backend": "vllm",
        "wall_time_seconds": elapsed,
        "input_tokens": usage.get("prompt_tokens", prompt_metadata.get("input_tokens")),
        "output_tokens": usage.get("completion_tokens"),
        "finish_reason": finish_reason,
        "eos_reached": finish_reason == "stop",
        "reached_max_new_tokens": finish_reason == "length",
        "consecutive_repeat_signal": first_consecutive_repeat(output_ids) if output_ids else None,
        "usage": usage,
        # Harmony channels: a degenerate loop lives in `analysis` and never
        # reaches `final`, so length has to be attributed per channel or a
        # truncated run reads as rambling.
        "analysis_chars": len(str(output_text).split("<|channel|>final")[0]),
        "final_chars": (
            len(str(output_text).split("<|channel|>final<|message|>")[-1])
            if "<|channel|>final" in str(output_text)
            else 0
        ),
        "reached_final_channel": "<|channel|>final" in str(output_text),
        **spec,
    }
    L = run_record["output_tokens"]
    l_bar = run_record.get("l_bar")
    run_record["L_over_l_bar"] = (L / l_bar) if (L and l_bar) else None
    write_json(output_dir / "run.json", run_record)
    print(
        f"{case} seed={seed} mode={tag}: L={L} finish={finish_reason} "
        f"l_bar={l_bar if l_bar is None else round(l_bar, 3)} "
        f"L/l_bar={run_record['L_over_l_bar'] if run_record['L_over_l_bar'] is None else round(run_record['L_over_l_bar'], 1)} "
        f"final_ch={run_record['reached_final_channel']} wall={elapsed:.2f}s",
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

    info = server_info(args.server_url)
    commit = git_commit()
    failures = 0
    for case in cases:
        for seed in args.seeds:
            try:
                run_one(args, case, seed, tag, info, commit)
            except Exception as exc:
                failures += 1
                print(f"{case} seed={seed} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
