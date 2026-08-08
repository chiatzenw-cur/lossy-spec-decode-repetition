#!/usr/bin/env python3
"""For each genuine bonus-escape event (from extract_bonus_escape_events.py),
ask the TARGET model alone (baseline mode, no drafter, temperature=0 =>
argmax) what it would emit next, given the exact prefix (original rendered
prompt + everything generated up to but not including the escape token).

One persistent server, many requests -- safe because prefix caching is off
(--no-enable-prefix-caching), so there is no cross-request KV reuse to cause
the "warm vs cold" divergence this project already documented elsewhere; only
literal per-request recomputation from a cold KV cache each time.

Output: for each event, whether the target's own argmax equals the "expected"
(periodic-continuation) token, the "actual" (bonus-emitted) token, both, or
neither -- the input to classify_bonus_escape_mechanism.py's 4-way table.

Run from repo root. Needs a free GPU (stops/starts its own vLLM server).
"""
from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO = pathlib.Path(__file__).resolve().parent.parent
EVENTS = REPO / "data" / "loop_token_metrics" / "bonus_escape_events.jsonl"
OUT = REPO / "data" / "loop_token_metrics" / "bonus_escape_replay_results.jsonl"
SERVER_LOG = REPO / "logs" / "bonus_escape_replay_server.log"
PORT = 30000
PY = str(REPO / ".venv-vllm" / "bin" / "python")
MODEL_NAME = "gpt-oss-20b"  # served-model-name in remote/run_server_vllm.sh, NOT the HF repo id


def health_ok() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def http_json(url: str, payload: dict, timeout: float = 120.0):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def start_baseline_server(log_path: pathlib.Path):
    env = dict(os.environ)
    env.pop("BETA", None)
    env.pop("DRAFT_MODEL_PATH", None)
    env.pop("PARALLEL_DRAFTING", None)
    env["PYTHON"] = PY
    env["PORT"] = str(PORT)
    env["SEED"] = "0"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        ["bash", str(REPO / "remote" / "run_server_vllm.sh"), "baseline"],
        cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
    )
    deadline = time.time() + 1800
    while time.time() < deadline:
        if health_ok():
            return proc
        if proc.poll() is not None:
            handle.close()
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-30:])
            raise RuntimeError(f"server died during startup:\n{tail}")
        time.sleep(2.0)
    raise RuntimeError("server did not become healthy in time")


def stop_server() -> None:
    subprocess.run(["bash", str(REPO / "remote" / "stop_server.sh")], cwd=REPO, check=False, capture_output=True)


def main() -> int:
    events = [json.loads(l) for l in EVENTS.open()]
    events = [e for e in events if not e["same_token"]]
    print(f"{len(events)} genuine bonus-escape events to replay")

    prompt_cache: dict[tuple, str] = {}

    def get_prompt(e):
        key = (e["runs_root"], e["case"], e["tag"])
        if key not in prompt_cache:
            p = pathlib.Path(e["runs_root"]) / e["case"] / "seed_0" / e["tag"] / "prompt.txt"
            prompt_cache[key] = p.read_text(encoding="utf-8")
        return prompt_cache[key]

    stop_server()
    print("starting baseline server...")
    proc = start_baseline_server(SERVER_LOG)
    print("server up")

    results = []
    errors = 0
    t0 = time.time()
    try:
        for i, e in enumerate(events):
            prompt = get_prompt(e) + bytes.fromhex(e["prefix_b64"]).decode("utf-8", errors="strict")
            payload = {
                "model": MODEL_NAME,
                "prompt": prompt,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 1,
                "seed": 0,
                "repetition_penalty": 1.0,
                "add_special_tokens": False,
                "skip_special_tokens": False,
                "spaces_between_special_tokens": False,
                "stream": False,
            }
            try:
                resp = http_json(f"http://127.0.0.1:{PORT}/v1/completions", payload, timeout=120.0)
                argmax_text = resp["choices"][0]["text"]
            except Exception as exc:
                errors += 1
                print(f"  [{i}] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
                argmax_text = None
            results.append({**e, "argmax_text": argmax_text})
            if (i + 1) % 25 == 0 or i == len(events) - 1:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(events)}] elapsed={elapsed:.0f}s errors={errors}", flush=True)
    finally:
        stop_server()
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(results)} results to {OUT} ({errors} errors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
