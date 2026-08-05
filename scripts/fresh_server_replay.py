#!/usr/bin/env python3
"""Replay cases with one freshly started server per (arm, case, seed).

Why this exists
---------------
Output depends on how many requests preceded it on the same engine: case_001
gives 1,711 tokens as a server's first request and 2,485 as its second. The
original pilot handled that by running each arm as a fresh server issuing all
ten cases in the same order -- but "same ordinal position" is not the same as
"same engine state". By case_002 the two arms have already produced different
numbers of tokens, acceptance events and RNG draws, so anything downstream of
the first case carries a request-history confound.

Pinning every measurement to ordinal 1 removes it: each request sees an engine
that has done no work at all, on both arms. The cost is a full server start per
run (minutes), which is why this is a separate driver rather than the default.

Each run is written by scripts/run_experiment_vllm.py with --assert-fresh-server,
so the artifact fails loudly rather than silently recording a warm request.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["strict", "lossy"],
        choices=("strict", "lossy", "baseline", "spec_casc_opt", "cactus"),
        help="Arms to replay. Each (arm, case, seed) gets its own server.",
    )
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--lenience-factor", type=float, default=0.2)
    parser.add_argument(
        "--spec-casc-alpha",
        type=float,
        default=0.05,
        help=(
            "spec-casc-opt's relaxation knob (Narasimhan et al. 2025). 0.05 matches the "
            "value that produced a repetition-loop failure with an MTP drafter in the "
            "reproduction paper (arXiv:2607.08690 Fig. 5) -- the same failure mode this "
            "repo studies, with a comparable EAGLE3 MTP-style drafter."
        ),
    )
    parser.add_argument(
        "--cactus-alpha",
        type=float,
        default=0.25,
        help=(
            "CACTUS's relaxation knob (Hao & Mou 2026), >= 0. Boosts the drafted token's "
            "acceptance as a function of p(x) and alpha only, never q -- unlike every other "
            "relaxed rule here. 0.25 is mid-range among the values arXiv:2607.08690 evaluates."
        ),
    )
    parser.add_argument("--prompt-root", type=pathlib.Path, default=pathlib.Path("prompts/aime24"))
    parser.add_argument("--runs-root", type=pathlib.Path, default=pathlib.Path("runs/fresh"))
    parser.add_argument("--log-root", type=pathlib.Path, default=pathlib.Path("logs/fresh"))
    parser.add_argument("--tag-suffix", default="", help="Appended to the default per-arm tag.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--server-seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    parser.add_argument("--request-timeout", type=float, default=7200.0)
    parser.add_argument(
        "--python",
        default=str(REPO_ROOT / ".venv-vllm" / "bin" / "python"),
        help="Interpreter for both the server and the request client.",
    )
    parser.add_argument(
        "--trace-proposals",
        action="store_true",
        help=(
            "Record every proposal token (p, q, u, strict/lossy counterfactual) to "
            "<run dir>/proposals.jsonl. Observation only; does not alter acceptance."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Redo runs that already exist.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def tag_for(
    arm: str, factor: float, suffix: str,
    spec_casc_alpha: float | None = None, cactus_alpha: float | None = None,
) -> str:
    if arm == "strict":
        base = "strict"
    elif arm == "baseline":
        base = "baseline"
    elif arm == "spec_casc_opt":
        base = f"specCascOpt{spec_casc_alpha:g}".replace(".", "p")
    elif arm == "cactus":
        base = f"cactus{cactus_alpha:g}".replace(".", "p")
    else:
        base = f"lenience{factor:g}".replace(".", "p")
    return base + suffix


def stop_server() -> None:
    subprocess.run(
        ["bash", str(REPO_ROOT / "remote" / "stop_server.sh")],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def health_ok(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


TRACE_PATH_FILE = pathlib.Path(f"/tmp/lossy-spec-decode-trace-{os.getuid()}")


def set_trace_destination(path: pathlib.Path | None) -> None:
    """Tell the patched sampler where to write its proposal trace.

    A file, not an environment variable: EngineCore is spawned with a sanitised
    environment, so env vars never reach the sampler. Must be set before the
    server starts, because the tracer resolves it at import.
    """
    if path is None:
        TRACE_PATH_FILE.write_text("", encoding="utf-8")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        TRACE_PATH_FILE.write_text(str(path), encoding="utf-8")


def start_server(args: argparse.Namespace, arm: str, log_path: pathlib.Path):
    """Start one server and wait for it to serve. Returns the Popen handle."""
    env = dict(os.environ)
    env.pop("BETA", None)  # run_server_vllm.sh refuses to start if this is set
    env["PYTHON"] = args.python
    env["PORT"] = str(args.port)
    env["SEED"] = str(args.server_seed)
    mode = "lossy" if arm in ("spec_casc_opt", "cactus") else arm
    if arm == "lossy":
        env["LOSSY_RULE"] = "lenience"
        env["LENIENCE_FACTOR"] = f"{args.lenience_factor:g}"
    elif arm == "spec_casc_opt":
        env["LOSSY_RULE"] = "spec_casc_opt"
        env["SPEC_CASC_ALPHA"] = f"{args.spec_casc_alpha:g}"
    elif arm == "cactus":
        env["LOSSY_RULE"] = "cactus"
        env["CACTUS_ALPHA"] = f"{args.cactus_alpha:g}"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    # New process group: the server spawns EngineCore children, and stop_server.sh
    # is the thing that knows how to clear the GPU, so this handle is only used
    # for liveness and for a last-resort kill.
    process = subprocess.Popen(
        ["bash", str(REPO_ROOT / "remote" / "run_server_vllm.sh"), mode],
        cwd=REPO_ROOT,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.time() + args.startup_timeout
    while time.time() < deadline:
        if health_ok(args.port):
            return process
        if process.poll() is not None:
            handle.close()
            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:])
            raise RuntimeError(f"server exited with code {process.returncode} during startup:\n{tail}")
        time.sleep(2.0)
    handle.close()
    raise RuntimeError(f"server did not become healthy within {args.startup_timeout:.0f}s; see {log_path}")


def request_once(
    args: argparse.Namespace, arm: str, case: str, seed: int, tag: str, log_path: pathlib.Path
) -> subprocess.CompletedProcess:
    command = [
        args.python,
        str(REPO_ROOT / "scripts" / "run_experiment_vllm.py"),
        "--mode", "lossy" if arm in ("lossy", "spec_casc_opt", "cactus") else arm,
        "--prompt-root", str(args.prompt_root),
        "--runs-root", str(args.runs_root),
        "--cases", case,
        "--seeds", str(seed),
        "--tag", tag,
        "--temperature", str(args.temperature),
        "--top-p", str(args.top_p),
        "--max-new-tokens", str(args.max_new_tokens),
        "--timeout", str(args.request_timeout),
        "--server-url", f"http://127.0.0.1:{args.port}",
        "--server-log", str(log_path),
        "--assert-fresh-server",
    ]
    if arm == "lossy":
        command += ["--lossy-method", "lenience", "--lenience-factor", f"{args.lenience_factor:g}"]
    elif arm == "spec_casc_opt":
        command += ["--lossy-method", "spec_casc_opt", "--spec-casc-alpha", f"{args.spec_casc_alpha:g}"]
    elif arm == "cactus":
        command += ["--lossy-method", "cactus", "--cactus-alpha", f"{args.cactus_alpha:g}"]
    if args.overwrite:
        command.append("--overwrite")
    return subprocess.run(command, cwd=REPO_ROOT, check=False)


def main() -> int:
    args = parse_args()
    unknown = [case for case in args.cases if not (REPO_ROOT / args.prompt_root / case).is_dir()]
    if unknown:
        print(f"unknown cases under {args.prompt_root}: {', '.join(unknown)}", file=sys.stderr)
        return 2

    plan = [
        (case, seed, arm, tag_for(arm, args.lenience_factor, args.tag_suffix, args.spec_casc_alpha, args.cactus_alpha))
        for case in args.cases
        for seed in args.seeds
        for arm in args.arms
    ]
    todo = []
    for case, seed, arm, tag in plan:
        run_json = REPO_ROOT / args.runs_root / case / f"seed_{seed}" / tag / "run.json"
        if run_json.is_file() and not args.overwrite:
            print(f"skip {case} seed={seed} {tag}: already present ({run_json})")
            continue
        todo.append((case, seed, arm, tag))

    print(f"{len(todo)} run(s), one fresh server each")
    for case, seed, arm, tag in todo:
        print(f"  {case} seed={seed} arm={arm} tag={tag}")
    if args.dry_run or not todo:
        return 0

    results = []
    failures = 0
    for index, (case, seed, arm, tag) in enumerate(todo, start=1):
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = REPO_ROOT / args.log_root / f"{tag}_{case}_seed{seed}_{stamp}.log"
        print(f"\n[{index}/{len(todo)}] {case} seed={seed} arm={arm} -> {log_path}", flush=True)
        started = time.perf_counter()
        status = "ok"
        try:
            stop_server()
            # Set before start_server: the tracer resolves its destination at
            # import, inside EngineCore. Staged outside the run directory --
            # run_experiment_vllm.py refuses to write into a directory that
            # already exists, so creating the trace there first would trip its
            # overwrite guard. Moved into the run directory afterwards.
            run_dir = REPO_ROOT / args.runs_root / case / f"seed_{seed}" / tag
            trace_stage = (
                REPO_ROOT / args.log_root / f"{tag}_{case}_seed{seed}_proposals.jsonl"
                if args.trace_proposals
                else None
            )
            set_trace_destination(trace_stage)
            process = start_server(args, arm, log_path)
            try:
                completed = request_once(args, arm, case, seed, tag, log_path)
                if completed.returncode != 0:
                    status = f"request failed (exit {completed.returncode})"
                    failures += 1
            finally:
                stop_server()
                if process.poll() is None:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                if trace_stage is not None and trace_stage.is_file():
                    if run_dir.is_dir():
                        trace_stage.replace(run_dir / "proposals.jsonl")
                    else:
                        print(f"  warning: run dir missing, trace left at {trace_stage}", file=sys.stderr)
        except (RuntimeError, OSError) as exc:
            status = f"{type(exc).__name__}: {exc}"
            failures += 1
            print(status, file=sys.stderr)
            stop_server()
        finally_trace = None
        try:
            set_trace_destination(None)
        except OSError as exc:  # non-fatal: only affects the next run's tracing
            finally_trace = str(exc)
        elapsed = time.perf_counter() - started
        print(f"[{index}/{len(todo)}] {status} in {elapsed:.0f}s", flush=True)
        if finally_trace:
            print(f"  warning: could not clear trace destination: {finally_trace}", file=sys.stderr)
        results.append(
            {
                "case": case,
                "seed": seed,
                "arm": arm,
                "tag": tag,
                "status": status,
                "wall_time_seconds": round(elapsed, 1),
                "server_log": os.path.relpath(log_path, REPO_ROOT),
            }
        )

    manifest = REPO_ROOT / args.runs_root / "fresh_server_replay.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    previous = []
    if manifest.is_file():
        try:
            previous = json.loads(manifest.read_text(encoding="utf-8")).get("batches", [])
        except (OSError, json.JSONDecodeError):
            previous = []
    previous.append(
        {
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "arms": args.arms,
            "lenience_factor": args.lenience_factor,
            "command": sys.argv,
            "runs": results,
        }
    )
    manifest.write_text(json.dumps({"batches": previous}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {manifest}; {len(results) - failures}/{len(results)} ok")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
