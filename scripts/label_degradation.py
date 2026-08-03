#!/usr/bin/env python3
"""Locate degradation events in generated text and record them as TOKEN positions.

Why token positions
-------------------
Distribution analysis joins these labels against proposals.jsonl, which is keyed
by token, so a label recorded as a character offset or a line number is useless.
Detectors work on text, then map byte offsets back to token indices through the
same byte-exact alignment scripts/trace_lookup.py uses.

Detectors, strongest evidence first
-----------------------------------
arith        An asserted integer identity that is simply false ("A*B = C" where
             C is wrong). Mechanically checkable, no judgement, no annotator
             bias. Conservative: only pure-integer claims with an unambiguous
             operator and a real '=' are checked, so approximations and modular
             arithmetic are skipped rather than mislabelled.
value_flip   The model asserts a value for some quantity, then later asserts a
             different value for what is evidently the same quantity (near-miss
             cluster). The second assertion is the event. This is the case_006
             signature -- 23 distinct values for one product.
repeat       A token n-gram repeats >=3 times consecutively; the start of the
             second repetition is the event.

Everything here is a *candidate*. Semantic corruption that is none of the above
needs an annotator, and this file is the place those labels land too, tagged with
their method, so automatic and human labels never get silently mixed.

Usage
    scripts/label_degradation.py --runs-root runs/aime24_fresh --tag lenience0p2
    scripts/label_degradation.py ... --out analysis/degradation_labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", type=pathlib.Path, default=pathlib.Path("runs/aime24_fresh"))
    p.add_argument("--tag", default="lenience0p2", help="Arm directory name to scan.")
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("analysis/degradation_labels.jsonl"))
    p.add_argument("--min-digits", type=int, default=4, help="Ignore arithmetic on small numbers.")
    p.add_argument("--repeat-n", type=int, default=16, help="n-gram size for the repeat detector.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------- alignment


def align(run_dir: pathlib.Path):
    """Token records with byte offsets, or None if the trace cannot be trusted."""
    import tiktoken

    trace = run_dir / "proposals.jsonl"
    out_txt = run_dir / "output.txt"
    if not (trace.is_file() and out_txt.is_file()):
        return None, None
    enc = tiktoken.get_encoding("o200k_harmony")
    rows = [json.loads(x) for x in trace.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows.sort(key=lambda r: r["output_position"])
    ids = [r["emitted_token_id"] for r in rows]
    raw = out_txt.read_text(encoding="utf-8").encode("utf-8")

    committed = None
    for drop in range(0, 64):
        end = len(ids) - drop
        if end <= 0:
            break
        blob = b"".join(enc.decode_single_token_bytes(t) for t in ids[:end])
        if raw.endswith(blob):
            committed = (end, len(raw) - len(blob))
            break
    if committed is None:
        return None, None
    n, prefill = committed

    recs, cursor = [], prefill
    for k in range(n):
        piece = enc.decode_single_token_bytes(ids[k])
        recs.append(
            {
                "token_index": k + 1,
                "output_position": rows[k]["output_position"],
                "byte_start": cursor,
                "byte_end": cursor + len(piece),
                "lossy_only_accepted": bool(rows[k].get("lossy_only_accepted")),
            }
        )
        cursor += len(piece)
    return raw, recs


def token_at(recs: list[dict], byte_pos: int) -> dict | None:
    lo, hi = 0, len(recs) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        r = recs[mid]
        if byte_pos < r["byte_start"]:
            hi = mid - 1
        elif byte_pos >= r["byte_end"]:
            lo = mid + 1
        else:
            return r
    return None


# ---------------------------------------------------------------- detectors

# A complete arithmetic statement, not a fragment of one. The first version of
# this detector regex-matched "A op B = C" and was wrong on 6 of 6 sampled hits:
# it matched two terms of a three-term sum, read across '=' into the next
# expression, and pulled the "2" out of the variable "S2". So instead: find
# maximal spans that are *entirely* numeric expressions joined by '=', evaluate
# every side, and only then compare. Anything containing a letter, an
# approximation sign or a stray symbol is skipped rather than guessed at.
_EXPR_CHARS = r"[0-9,\s()*+×\-/^]"
_STATEMENT = re.compile(
    r"(?<![A-Za-z0-9_.])" + _EXPR_CHARS + r"{2,}(?:=" + _EXPR_CHARS + r"{2,})+"
)
_SAFE = re.compile(r"^[0-9\s()*+\-/]+$")


def _int(s: str) -> int:
    return int(s.replace(",", ""))


def _eval_side(expr: str):
    """Value of a pure-integer expression, or None if it is not safely evaluable."""
    e = expr.replace(",", "").replace("×", "*").strip()
    if not e or not _SAFE.match(e):
        return None
    if re.search(r"[*+\-/]\s*$", e) or re.match(r"^\s*[*/]", e):
        return None  # dangling operator: an incomplete fragment
    if "/" in e:
        return None  # integer division is ambiguous here; skip rather than guess
    try:
        value = eval(e, {"__builtins__": {}}, {})  # noqa: S307 - digits/operators only
    except (SyntaxError, ZeroDivisionError, TypeError, ValueError):
        return None
    return value if isinstance(value, int) else None


def detect_arith(text: str, min_digits: int):
    for m in _STATEMENT.finditer(text):
        span = m.group(0)
        # Reject if the character right after the span continues the expression,
        # which is what made the old detector read into the following term.
        after = text[m.end() : m.end() + 2]
        if re.match(r"\s*[*+×/^-]\s*\d", after):
            continue
        sides = [s for s in span.split("=")]
        if len(sides) < 2:
            continue
        values, ok = [], True
        for s in sides:
            v = _eval_side(s)
            if v is None:
                ok = False
                break
            values.append(v)
        if not ok or len(values) < 2:
            continue
        # Require a real computation somewhere, not "5 = 5", and a big operand.
        if not any(re.search(r"[*+×-]", s) for s in sides):
            continue
        if max(abs(v) for v in values) < 10 ** (min_digits - 1):
            continue
        if len(set(values)) > 1:
            yield {
                "method": "arith",
                "byte_start": m.start(),
                "byte_end": m.end(),
                "evidence": {
                    "claim": span.strip()[:120],
                    "sides": values,
                    "expected": values[0],
                    "asserted": values[-1],
                    "delta": values[-1] - values[0],
                },
            }


BIG = re.compile(rf"(?<![\d.])(\d{{1,3}}(?:,\d{{3}}){{2,}})(?![\d.,])")


def detect_value_flip(text: str, rel: float = 1e-3):
    """A second, different value asserted for what is evidently one quantity.

    Grouping is by magnitude proximity, so two values within `rel` of each other
    are treated as competing claims about the same thing. Crude, but it is the
    pattern that produced 23 values for a single product in case_006.
    """
    seen: list[tuple[int, int]] = []  # (value, byte)
    for m in BIG.finditer(text):
        v = _int(m.group(1))
        for prev_v, _prev_b in seen:
            if prev_v != v and abs(v - prev_v) <= max(1, int(prev_v * rel)):
                yield {
                    "method": "value_flip",
                    "byte_start": m.start(1),
                    "byte_end": m.end(1),
                    "evidence": {"previous": prev_v, "asserted": v, "delta": v - prev_v},
                }
                break
        seen.append((v, m.start(1)))


def detect_repeat(recs: list[dict], ids: list[int], n: int, repeats: int = 3):
    """First start of a token n-gram repeated `repeats` times back to back."""
    span = n * repeats
    for start in range(0, len(ids) - span + 1):
        block = ids[start : start + n]
        if all(ids[start + i * n : start + (i + 1) * n] == block for i in range(1, repeats)):
            r = recs[start + n]  # the second repetition is where it goes wrong
            return {
                "method": "repeat",
                "byte_start": r["byte_start"],
                "byte_end": r["byte_end"],
                "evidence": {"ngram_tokens": n, "repeats": repeats, "first_start_token": recs[start]["token_index"]},
            }
    return None


# ---------------------------------------------------------------- driver


def main() -> int:
    args = parse_args()
    runs = sorted(args.runs_root.glob(f"case_*/seed_*/{args.tag}"))
    if not runs:
        print(f"no runs under {args.runs_root} with tag {args.tag}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    findings, skipped, per_case = [], [], defaultdict(lambda: defaultdict(int))
    for run_dir in runs:
        case = run_dir.parts[-3]
        seed = run_dir.parts[-2]
        raw, recs = align(run_dir)
        if recs is None:
            skipped.append(str(run_dir))
            continue
        text = raw.decode("utf-8", errors="replace")
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        ids = [r["output_position"] for r in recs]  # placeholder, replaced below
        tok_ids = [
            json.loads(x)["emitted_token_id"]
            for x in (run_dir / "proposals.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()
        ][: len(recs)]

        hits = list(detect_arith(text, args.min_digits)) + list(detect_value_flip(text))
        rep = detect_repeat(recs, tok_ids, args.repeat_n)
        if rep:
            hits.append(rep)

        for h in hits:
            tok = token_at(recs, h["byte_start"])
            if tok is None:
                continue
            per_case[case][h["method"]] += 1
            findings.append(
                {
                    "case": case,
                    "seed": seed,
                    "tag": args.tag,
                    "method": h["method"],
                    "token_index": tok["token_index"],
                    "output_position": tok["output_position"],
                    "byte_start": h["byte_start"],
                    "byte_end": h["byte_end"],
                    "lossy_only_at_token": tok["lossy_only_accepted"],
                    "run_finish_reason": run.get("finish_reason"),
                    "run_reached_final_channel": run.get("reached_final_channel"),
                    "run_output_tokens": run.get("output_tokens"),
                    "evidence": h["evidence"],
                    "context": text[max(0, h["byte_start"] - 80) : h["byte_end"] + 80].replace("\n", " "),
                }
            )

    findings.sort(key=lambda f: (f["case"], f["token_index"]))
    with args.out.open("w", encoding="utf-8") as fh:
        for f in findings:
            fh.write(json.dumps(f, ensure_ascii=False, separators=(",", ":")) + "\n")

    if not args.quiet:
        print(f"scanned {len(runs)} runs, {len(skipped)} unalignable")
        print(f"wrote {args.out}: {len(findings)} candidate degradation positions\n")
        print(f"  {'case':10} {'arith':>6} {'flip':>6} {'repeat':>7} {'earliest tok':>13} {'finish':>7}")
        for case in sorted(per_case):
            firsts = [f["token_index"] for f in findings if f["case"] == case]
            fin = next(f["run_finish_reason"] for f in findings if f["case"] == case)
            print(
                f"  {case:10} {per_case[case]['arith']:>6} {per_case[case]['value_flip']:>6} "
                f"{per_case[case]['repeat']:>7} {min(firsts):>13} {fin:>7}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
