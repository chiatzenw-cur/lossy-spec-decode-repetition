#!/usr/bin/env python3
"""Run and archive paired GPT-OSS generations against vLLM's OpenAI-compatible API.

Mirrors scripts/run_lossy_experiment.py (SGLang) and writes the same artifact
contract, so scripts/summarize_runs.py works unchanged across both backends.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import sysconfig
import time
import urllib.error
import urllib.request
from typing import Any


DEFAULT_PROMPT_ROOT = pathlib.Path("prompts/leval_9k_11k")

# Same path the patched sampler reads, derived the same way. Hardcoding a
# repo-relative path here (as an earlier version did) desynchronises writer and
# reader for anyone whose clone is not the one the patch was generated from.
LENIENCE_FACTOR_FILE = pathlib.Path(f"/tmp/lossy-spec-decode-lenience-{os.getuid()}")
SPEC_CASC_ALPHA_FILE = pathlib.Path(f"/tmp/lossy-spec-decode-spec-casc-alpha-{os.getuid()}")
CACTUS_ALPHA_FILE = pathlib.Path(f"/tmp/lossy-spec-decode-cactus-alpha-{os.getuid()}")

# Files the Lenience patch touches, with the sha256 of the patched form. The
# hashes go into every config.json so a run directory alone proves which
# verifier ran, without trusting the directory name or an external server log.
PATCHED_FILES = {
    "vllm/v1/sample/rejection_sampler.py": (
        "81a0947d7263675a07125b714b3093fbd82f91e3211a642a4d0ec448ad2b898d"
    ),
    "vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py": (
        "0ad55a1cb39f2306c78a170fdb6468a36b55643c6610c85cd46c908e0d313112"
    ),
}

# spec-casc-opt touches only the V1 file (mutually exclusive with the
# Lenience patch above -- both start from the same pristine file). The other
# file must be pristine when this arm runs, so its expected hash here is the
# UPSTREAM one, not a patched one, and verified accordingly.
SPEC_CASC_OPT_PATCHED_FILES = {
    "vllm/v1/sample/rejection_sampler.py": (
        "de32559fa494f8b4b88df34874793001d066492cd034f88e046fcd63af0de85d"
    ),
}

# CACTUS is the third patch mutually exclusive with the two above, from the
# same pristine file. v2 (full-residual H_x fix); the v1 accept-only hash is
# 02492f03bdf90c9442bb4bca81c61b82c06ad34b733a295f9305326941a93068 -- data
# collected under that version is tagged cactus_accept_only, not cactus. See
# patches/vllm-0.26.0-cactus.patch's header for what v2 fixes.
CACTUS_PATCHED_FILES = {
    "vllm/v1/sample/rejection_sampler.py": (
        "4fa623a70332075cada34dbe585a05fc941db12fb3dd67edee3bd3923f779074"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("baseline", "strict", "lossy"))
    parser.add_argument(
        "--lossy-method",
        choices=("lenience", "synthetic_acceptance", "spec_casc_opt", "cactus"),
        default=None,
        help="Required in lossy mode. Must match the server's LOSSY_RULE.",
    )
    parser.add_argument(
        "--lenience-factor",
        type=float,
        default=None,
        help=(
            "Required for --lossy-method lenience. Accept iff p/(factor*q) >= u. "
            "Checked against the value the server actually loaded."
        ),
    )
    parser.add_argument(
        "--synthetic-acceptance-length",
        type=float,
        default=None,
        help="Required for --lossy-method synthetic_acceptance; must match SYNTH_LEN.",
    )
    parser.add_argument(
        "--spec-casc-alpha",
        type=float,
        default=None,
        help=(
            "Required for --lossy-method spec_casc_opt (Narasimhan et al. 2025). Defer to "
            "the strict test iff max_u q(u) < max_u p(u) - alpha*TV(p,q), else accept "
            "unconditionally. Checked against the value the server actually loaded."
        ),
    )
    parser.add_argument(
        "--cactus-alpha",
        type=float,
        default=None,
        help=(
            "Required for --lossy-method cactus (Hao & Mou 2026). Boost the drafted "
            "token's acceptance via gamma_x=min(p(x)+sqrt(2*alpha*p(x)*(1-p(x))),1), "
            "alpha>=0. Checked against the value the server actually loaded."
        ),
    )
    parser.add_argument("--tag", help="Output directory label; defaults to a name built from the arm.")
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
        "--assert-fresh-server",
        action="store_true",
        help=(
            "Fail unless the server has served nothing yet. Output depends on how "
            "many requests preceded it on the same engine, so an arm comparison is "
            "only clean if both sides sit at the same position -- which one request "
            "per server makes trivially true."
        ),
    )
    parser.add_argument(
        "--server-log",
        type=pathlib.Path,
        default=None,
        help=(
            "Server stdout/stderr log. The patched sampler announces the factor it "
            "loaded there; recording that line is the only proof that does not "
            "depend on the file still holding the same value."
        ),
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


def engine_totals(base_url: str) -> dict[str, float]:
    """Cumulative work counters, used only to tell a fresh engine from a used one."""
    wanted = (
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:request_success_total",
        "vllm:spec_decode_num_drafts_total",
    )
    out: dict[str, float] = {}
    try:
        request = urllib.request.Request(f"{base_url.rstrip('/')}/metrics")
        text = urllib.request.urlopen(request, timeout=30).read().decode("utf-8")
    except Exception:
        return out
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for metric in wanted:
            if line.startswith(metric):
                try:
                    out[metric] = out.get(metric, 0.0) + float(line.rsplit(" ", 1)[1])
                except (ValueError, IndexError):
                    pass
    return out


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(out.strip())


def sha256_of(path: pathlib.Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def vllm_install_info() -> dict[str, Any]:
    """Version and verifier hashes of the vLLM this interpreter would import.

    Read off disk rather than by importing vLLM: this is the client process and
    importing it costs ~10s per invocation, which the one-server-per-case driver
    pays once per case. Only meaningful when the runner shares a filesystem with
    the server, which is the supported single-box setup.
    """
    info: dict[str, Any] = {"version": None, "commit_id": None, "site_packages": None}
    try:
        purelib = pathlib.Path(sysconfig.get_paths()["purelib"])
    except Exception:
        return info
    info["site_packages"] = str(purelib)
    version_py = purelib / "vllm" / "_version.py"
    try:
        text = version_py.read_text(encoding="utf-8")
    except OSError:
        return info
    for key, pattern in (
        ("version", r"__version__ = version = '([^']+)'"),
        ("commit_id", r"__commit_id__ = commit_id = '([^']+)'"),
    ):
        match = re.search(pattern, text)
        if match:
            info[key] = match.group(1)
    files: dict[str, Any] = {}
    for rel, expected in PATCHED_FILES.items():
        got = sha256_of(purelib / rel)
        files[rel] = {"sha256": got, "matches_lenience_patch": got == expected}
    info["verifier_files"] = files
    info["lenience_patch_applied"] = all(
        entry["matches_lenience_patch"] for entry in files.values()
    )
    # Mutually exclusive with the above: spec-casc-opt patches only the V1
    # file, from the same pristine source the Lenience patch starts from, so
    # its sha256 is checked separately here rather than folded into `files`.
    spec_casc_files: dict[str, Any] = {}
    for rel, expected in SPEC_CASC_OPT_PATCHED_FILES.items():
        got = sha256_of(purelib / rel)
        spec_casc_files[rel] = {"sha256": got, "matches_spec_casc_opt_patch": got == expected}
    info["spec_casc_opt_verifier_files"] = spec_casc_files
    info["spec_casc_opt_patch_applied"] = all(
        entry["matches_spec_casc_opt_patch"] for entry in spec_casc_files.values()
    )
    cactus_files: dict[str, Any] = {}
    for rel, expected in CACTUS_PATCHED_FILES.items():
        got = sha256_of(purelib / rel)
        cactus_files[rel] = {"sha256": got, "matches_cactus_patch": got == expected}
    info["cactus_verifier_files"] = cactus_files
    info["cactus_patch_applied"] = all(
        entry["matches_cactus_patch"] for entry in cactus_files.values()
    )
    return info


def lenience_in_force() -> dict[str, Any]:
    """The factor the patched sampler would load, read from its own channel.

    The server writes this file before starting and the sampler reads it at
    import, so agreement between it and --lenience-factor is what makes a run
    directory self-describing.
    """
    record: dict[str, Any] = {"path": str(LENIENCE_FACTOR_FILE), "value": None}
    try:
        record["value"] = float(LENIENCE_FACTOR_FILE.read_text().strip())
        record["mtime_utc"] = dt.datetime.fromtimestamp(
            LENIENCE_FACTOR_FILE.stat().st_mtime, dt.timezone.utc
        ).isoformat()
    except (OSError, ValueError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def spec_casc_alpha_in_force() -> dict[str, Any]:
    """The alpha the patched sampler would load, read from its own channel.

    Mirrors lenience_in_force(): the server writes this file before starting
    and the sampler reads it at import, so agreement with --spec-casc-alpha
    is what makes a run directory self-describing.
    """
    record: dict[str, Any] = {"path": str(SPEC_CASC_ALPHA_FILE), "value": None}
    try:
        record["value"] = float(SPEC_CASC_ALPHA_FILE.read_text().strip())
        record["mtime_utc"] = dt.datetime.fromtimestamp(
            SPEC_CASC_ALPHA_FILE.stat().st_mtime, dt.timezone.utc
        ).isoformat()
    except (OSError, ValueError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def server_log_spec_casc_alpha(path: pathlib.Path | None) -> dict[str, Any] | None:
    """The alpha the sampler actually announced, scraped from the server log.

    Mirrors server_log_lenience(): the file check above can only say what the
    file held when the client looked; this says what the engine loaded at
    import, which is the value that ran.
    """
    if path is None:
        return None
    record: dict[str, Any] = {"path": str(path), "lines": [], "alphas": []}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    for line in text.splitlines():
        if "[SPEC-CASC-OPT PATCH" not in line:
            continue
        record["lines"].append(line.strip())
        match = re.search(r"alpha=(-?inf|nan|[0-9.eE+-]+)", line)
        if match:
            try:
                record["alphas"].append(float(match.group(1)))
            except ValueError:
                pass
    record["distinct_alphas"] = sorted(set(record["alphas"]))
    return record


def cactus_alpha_in_force() -> dict[str, Any]:
    """The alpha the CACTUS-patched sampler would load, read from its own channel.

    Mirrors lenience_in_force() / spec_casc_alpha_in_force().
    """
    record: dict[str, Any] = {"path": str(CACTUS_ALPHA_FILE), "value": None}
    try:
        record["value"] = float(CACTUS_ALPHA_FILE.read_text().strip())
        record["mtime_utc"] = dt.datetime.fromtimestamp(
            CACTUS_ALPHA_FILE.stat().st_mtime, dt.timezone.utc
        ).isoformat()
    except (OSError, ValueError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def server_log_cactus_alpha(path: pathlib.Path | None) -> dict[str, Any] | None:
    """The alpha the CACTUS-patched sampler actually announced, scraped from the
    server log. Mirrors server_log_spec_casc_alpha()."""
    if path is None:
        return None
    record: dict[str, Any] = {"path": str(path), "lines": [], "alphas": []}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    for line in text.splitlines():
        if "[CACTUS PATCH" not in line:
            continue
        record["lines"].append(line.strip())
        match = re.search(r"alpha=(-?inf|nan|[0-9.eE+-]+)", line)
        if match:
            try:
                record["alphas"].append(float(match.group(1)))
            except ValueError:
                pass
    record["distinct_alphas"] = sorted(set(record["alphas"]))
    return record


def server_log_lenience(path: pathlib.Path | None) -> dict[str, Any] | None:
    """The factor the sampler actually announced, scraped from the server log.

    The file check above can only say what the file held when the client looked;
    this says what the engine loaded at import, which is the value that ran.
    """
    if path is None:
        return None
    record: dict[str, Any] = {"path": str(path), "lines": [], "factors": []}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    for line in text.splitlines():
        if "[LENIENCE PATCH" not in line:
            continue
        record["lines"].append(line.strip())
        match = re.search(r"lenience_factor=([0-9.eE+-]+)", line)
        if match:
            try:
                record["factors"].append(float(match.group(1)))
            except ValueError:
                pass
    record["distinct_factors"] = sorted(set(record["factors"]))
    return record


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


def safe_tag(args: argparse.Namespace) -> str:
    if args.tag:
        tag = args.tag
    elif args.mode != "lossy":
        tag = args.mode
    elif args.lossy_method == "lenience":
        tag = f"lenience{args.lenience_factor:g}".replace(".", "p")
    elif args.lossy_method == "spec_casc_opt":
        # "-" is in the allowed charset below, so a negative alpha (e.g.
        # -0.1) needs no further escaping to stay a safe tag/directory name.
        tag = f"specCascOpt{args.spec_casc_alpha:g}".replace(".", "p")
    elif args.lossy_method == "cactus":
        tag = f"cactus{args.cactus_alpha:g}".replace(".", "p")
    else:
        tag = f"synthetic{args.synthetic_acceptance_length:g}".replace(".", "p")
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
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")

    ALL_METHOD_PARAMS = (
        ("--lenience-factor", "lenience_factor"),
        ("--synthetic-acceptance-length", "synthetic_acceptance_length"),
        ("--spec-casc-alpha", "spec_casc_alpha"),
        ("--cactus-alpha", "cactus_alpha"),
    )

    def require_only(required_flag: str, required_attr: str) -> None:
        for flag, attr in ALL_METHOD_PARAMS:
            value = getattr(args, attr)
            if attr == required_attr:
                if value is None:
                    raise ValueError(f"--lossy-method {args.lossy_method} requires {required_flag}")
            elif value is not None:
                raise ValueError(f"{flag} is not used by --lossy-method {args.lossy_method}")

    if args.mode == "lossy":
        if args.lossy_method is None:
            raise ValueError(
                "lossy mode requires --lossy-method {lenience,synthetic_acceptance,spec_casc_opt,cactus}"
            )
        if args.lossy_method == "lenience":
            require_only("--lenience-factor", "lenience_factor")
            if not 0.0 < args.lenience_factor < 1.0:
                raise ValueError(
                    f"--lenience-factor must be in (0, 1) to be lossy; got {args.lenience_factor} "
                    "(1.0 is the lossless rule: run it as --mode strict)"
                )
        elif args.lossy_method == "spec_casc_opt":
            require_only("--spec-casc-alpha", "spec_casc_alpha")
        elif args.lossy_method == "cactus":
            require_only("--cactus-alpha", "cactus_alpha")
            if args.cactus_alpha < 0.0:
                raise ValueError(f"--cactus-alpha must be >= 0 (it bounds a KL divergence); got {args.cactus_alpha}")
        else:
            require_only("--synthetic-acceptance-length", "synthetic_acceptance_length")
    else:
        for flag, attr in ALL_METHOD_PARAMS:
            if getattr(args, attr) is not None:
                raise ValueError(f"{flag} is only valid with --mode lossy")
        if args.lossy_method is not None:
            raise ValueError("--lossy-method is only valid with --mode lossy")


def acceptance_rule_record(args: argparse.Namespace) -> dict[str, Any]:
    """Everything needed to identify the acceptance rule from the artifact alone.

    Also fails the run when the factor the server loaded disagrees with the one
    requested -- including the strict arm, which must be running at exactly 1.0.
    An arm mislabelled here is invisible afterwards: that is how a 'strict' run
    directory ended up holding lossy output once already.
    """
    in_force = lenience_in_force()
    record: dict[str, Any] = {
        "mode": args.mode,
        "lossy_method": args.lossy_method,
        "lossy_parameters": {},
        "lenience_factor_in_force": in_force,
    }

    if args.mode == "lossy" and args.lossy_method == "lenience":
        factor = args.lenience_factor
        record["lossy_parameters"] = {"lenience_factor": factor}
        record["acceptance_rule"] = f"accept iff p(x) / ({factor:g} * q(x)) >= u"
        # Naming: this factor is lambda. In the mentored-decoding notation of
        # Xia et al. it is 1 - alpha; their beta is a different parameter, fixed
        # at 1 there, so recording this as beta would misidentify the method.
        record["taxonomy"] = {
            "family": "mentored decoding (Xia et al.)",
            "lambda": factor,
            "alpha_equivalent": round(1.0 - factor, 12),
            "note": "lambda = 1 - alpha; residual and bonus sampling are unchanged from stock",
        }
    elif args.mode == "lossy" and args.lossy_method == "spec_casc_opt":
        alpha = args.spec_casc_alpha
        record["lossy_parameters"] = {"spec_casc_alpha": alpha}
        record["acceptance_rule"] = (
            f"defer to strict p/q test iff max_u q(u) < max_u p(u) - {alpha:g}*TV(p,q), "
            "else accept unconditionally"
        )
        record["taxonomy"] = {
            "family": "speculative cascades [OPT] (Narasimhan et al. 2025)",
            "alpha": alpha,
            "note": (
                "training-free relaxed target distribution; see arXiv:2607.08690 "
                "for a practical comparison against mentored decoding / lenience"
            ),
        }
    elif args.mode == "lossy" and args.lossy_method == "cactus":
        alpha = args.cactus_alpha
        record["lossy_parameters"] = {"cactus_alpha": alpha}
        record["acceptance_rule"] = (
            f"accept iff gamma_x / q(x) >= u, where "
            f"gamma_x = min(p(x) + sqrt(2*{alpha:g}*p(x)*(1-p(x))), 1)"
        )
        record["taxonomy"] = {
            "family": "CACTUS (Hao & Mou 2026)",
            "alpha": alpha,
            "note": (
                "training-free relaxed target distribution, boost depends only on p(x) and "
                "alpha, never on q -- see arXiv:2607.08690 Finding 1 for why this behaves "
                "similarly to mentored decoding / lenience. Residual (post-rejection) sampling "
                "is left as unmodified strict p, not CACTUS's own pi_res == pi_rej, matching "
                "this repo's existing lenience patch's beta=1 simplification precedent."
            ),
        }
    elif args.mode == "lossy":
        record["lossy_parameters"] = {
            "synthetic_acceptance_length": args.synthetic_acceptance_length
        }
        record["acceptance_rule"] = (
            f"synthetic: accept at a prescribed rate for mean length "
            f"{args.synthetic_acceptance_length:g}, ignoring p and q"
        )
    else:
        record["acceptance_rule"] = "accept iff p(x) / q(x) >= u"

    # The synthetic rule does not read the factor file, so there is nothing to
    # cross-check; every other arm must agree with what the sampler loaded.
    expected = None if (args.mode == "lossy" and args.lossy_method != "lenience") else (
        args.lenience_factor if args.mode == "lossy" else 1.0
    )
    record["lenience_factor_expected"] = expected

    announced = server_log_lenience(args.server_log)
    if announced is not None:
        record["lenience_factor_announced_by_server"] = announced

    if expected is not None:
        got = in_force["value"]
        if got is None:
            raise ValueError(
                f"expected lenience factor {expected:g} but {LENIENCE_FACTOR_FILE} is unreadable "
                f"({in_force.get('error')}). Start the server with remote/run_server_vllm.sh, "
                "which writes it for every mode."
            )
        if abs(got - expected) > 1e-12:
            raise ValueError(
                f"server loaded lenience factor {got:g}, run was invoked for {expected:g}. "
                "Refusing to write a mislabelled run directory."
            )
        if announced is not None and announced.get("distinct_factors"):
            if any(abs(f - expected) > 1e-12 for f in announced["distinct_factors"]):
                raise ValueError(
                    f"server log {args.server_log} announces lenience factor(s) "
                    f"{announced['distinct_factors']}, run was invoked for {expected:g}"
                )

    # Parallel check for spec-casc-opt's own knob -- only this arm's alpha is
    # verified here; every other arm leaves it at the neutral -inf the server
    # writes for exactly this reason (see run_server_vllm.sh).
    if args.mode == "lossy" and args.lossy_method == "spec_casc_opt":
        alpha_expected = args.spec_casc_alpha
        alpha_in_force = spec_casc_alpha_in_force()
        record["spec_casc_alpha_in_force"] = alpha_in_force
        alpha_announced = server_log_spec_casc_alpha(args.server_log)
        if alpha_announced is not None:
            record["spec_casc_alpha_announced_by_server"] = alpha_announced

        alpha_got = alpha_in_force["value"]
        if alpha_got is None:
            raise ValueError(
                f"expected spec-casc-opt alpha {alpha_expected:g} but {SPEC_CASC_ALPHA_FILE} is "
                f"unreadable ({alpha_in_force.get('error')}). Start the server with "
                "remote/run_server_vllm.sh, which writes it for every mode."
            )
        if abs(alpha_got - alpha_expected) > 1e-12:
            raise ValueError(
                f"server loaded spec-casc-opt alpha {alpha_got:g}, run was invoked for "
                f"{alpha_expected:g}. Refusing to write a mislabelled run directory."
            )
        if alpha_announced is not None and alpha_announced.get("distinct_alphas"):
            if any(abs(a - alpha_expected) > 1e-12 for a in alpha_announced["distinct_alphas"]):
                raise ValueError(
                    f"server log {args.server_log} announces spec-casc-opt alpha(s) "
                    f"{alpha_announced['distinct_alphas']}, run was invoked for {alpha_expected:g}"
                )

    # Same pattern again for CACTUS's own knob.
    if args.mode == "lossy" and args.lossy_method == "cactus":
        cactus_expected = args.cactus_alpha
        cactus_in_force = cactus_alpha_in_force()
        record["cactus_alpha_in_force"] = cactus_in_force
        cactus_announced = server_log_cactus_alpha(args.server_log)
        if cactus_announced is not None:
            record["cactus_alpha_announced_by_server"] = cactus_announced

        cactus_got = cactus_in_force["value"]
        if cactus_got is None:
            raise ValueError(
                f"expected CACTUS alpha {cactus_expected:g} but {CACTUS_ALPHA_FILE} is "
                f"unreadable ({cactus_in_force.get('error')}). Start the server with "
                "remote/run_server_vllm.sh, which writes it for every mode."
            )
        if abs(cactus_got - cactus_expected) > 1e-12:
            raise ValueError(
                f"server loaded CACTUS alpha {cactus_got:g}, run was invoked for "
                f"{cactus_expected:g}. Refusing to write a mislabelled run directory."
            )
        if cactus_announced is not None and cactus_announced.get("distinct_alphas"):
            if any(abs(a - cactus_expected) > 1e-12 for a in cactus_announced["distinct_alphas"]):
                raise ValueError(
                    f"server log {args.server_log} announces CACTUS alpha(s) "
                    f"{cactus_announced['distinct_alphas']}, run was invoked for {cactus_expected:g}"
                )
    return record


def run_one(
    args: argparse.Namespace,
    case: str,
    seed: int,
    tag: str,
    info: dict[str, Any],
    provenance: dict[str, Any],
    ordinal: int,
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
        "backend": "vllm",
        "tag": tag,
        **provenance["acceptance"],
        "model": args.model,
        "draft_model": None if args.mode == "baseline" else args.draft_model,
        "seed": seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "input_tokens_archived": prompt_metadata.get("input_tokens"),
        "prompt_case": case,
        "prompt_source_id": prompt_metadata.get("source_id"),
        "reference_answer": prompt_metadata.get("reference_answer"),
        "endpoint": f"{args.server_url.rstrip('/')}/v1/completions",
        # Request position on this engine. Output depends on it (see README), so
        # a comparison is only clean between runs that share it; the fresh-server
        # driver pins it to 1 on both arms.
        "server_request_ordinal": ordinal,
        "fresh_server_asserted": args.assert_fresh_server,
        "engine_totals_before_first_request": provenance["engine_totals_at_start"],
        "vllm": provenance["vllm"],
        "git_commit": provenance["git_commit"],
        "git_dirty": provenance["git_dirty"],
        "command": provenance["command"],
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

    run_record = {
        "status": "ok",
        "backend": "vllm",
        "wall_time_seconds": elapsed,
        "server_request_ordinal": ordinal,
        "input_tokens": usage.get("prompt_tokens", prompt_metadata.get("input_tokens")),
        "output_tokens": usage.get("completion_tokens"),
        "finish_reason": finish_reason,
        "eos_reached": finish_reason == "stop",
        "reached_max_new_tokens": finish_reason == "length",
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
        acceptance = acceptance_rule_record(args)
        totals = engine_totals(args.server_url)
        if args.assert_fresh_server:
            if not totals:
                raise ValueError(
                    f"cannot verify a fresh server: no usable counters at {args.server_url}/metrics"
                )
            used = {name: value for name, value in totals.items() if value > 0}
            if used:
                raise ValueError(
                    f"server has already served requests ({used}); --assert-fresh-server "
                    "requires an engine that has done no work yet"
                )
            if len(cases) * len(args.seeds) > 1:
                raise ValueError(
                    "--assert-fresh-server takes exactly one case and one seed: the guarantee "
                    "is one request per engine, and only the first request is at ordinal 1"
                )
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    provenance = {
        "acceptance": acceptance,
        "engine_totals_at_start": totals or None,
        "vllm": vllm_install_info(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "command": sys.argv,
    }
    if args.mode == "lossy" and args.lossy_method == "lenience":
        if not provenance["vllm"].get("lenience_patch_applied"):
            print(
                "configuration error: the Lenience patch is not applied to "
                f"{provenance['vllm'].get('site_packages')}; run bash patches/apply.sh",
                file=sys.stderr,
            )
            return 2
    if args.mode == "lossy" and args.lossy_method == "spec_casc_opt":
        if not provenance["vllm"].get("spec_casc_opt_patch_applied"):
            print(
                "configuration error: the spec-casc-opt patch is not applied to "
                f"{provenance['vllm'].get('site_packages')}; run bash patches/apply_spec_casc_opt.sh",
                file=sys.stderr,
            )
            return 2
    if args.mode == "lossy" and args.lossy_method == "cactus":
        if not provenance["vllm"].get("cactus_patch_applied"):
            print(
                "configuration error: the CACTUS patch is not applied to "
                f"{provenance['vllm'].get('site_packages')}; run bash patches/apply_cactus.sh",
                file=sys.stderr,
            )
            return 2

    info = server_info(args.server_url)
    failures = 0
    ordinal = 0
    for case in cases:
        for seed in args.seeds:
            ordinal += 1
            try:
                run_one(args, case, seed, tag, info, provenance, ordinal)
            except Exception as exc:
                failures += 1
                print(f"{case} seed={seed} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
