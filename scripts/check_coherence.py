#!/usr/bin/env python3
"""Test whether relaxed verification degrades *coherence*, or only termination.

Three families of mechanical check, all paired per problem and all normalised,
because the lossy arm emits 1.45x more text and any raw count would just
re-measure length:

  arithmetic  every `a op b = c` claim in the text, verified exactly. This is
              ground truth, not a proxy: it tests the repo's own hypothesis that
              digits get corrupted on emission while the reasoning stays valid.
  repetition  word 10-gram novelty and the longest immediately-repeated span.
              Separates "stuck in a loop" from "exploring for longer".
  surface     non-ASCII rate, LaTeX delimiter imbalance, word length. Catches
              gross illegibility -- garbled tokens, broken markup, code-switching.

Two length controls, because a 32k-token run and a 5k-token run are not
comparable:

  --prefix-tokens N   score only the first N tokens of both arms (default: the
                      shorter of the pair, so every comparison is like-for-like)
  --buckets           report the first and last 2k tokens separately, which
                      distinguishes constant emission-level corruption from a
                      degradation that accumulates

Absolute arithmetic-error rates are noisy: the extractor cannot tell `2 - 3 = 0`
inside an algebraic expression from a real computation. The comparison is
paired on identical problems, so a constant false-positive rate affects both
arms alike and largely cancels in the ratio. Read the ratio, not the level.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import statistics
import sys
from collections import Counter
from typing import Any

# A whole arithmetic chain and its claimed result, on one line. Matching only a
# binary pair is wrong: `13*6*2 = 156` then yields a bogus `6*2 = 156`, and that
# false-positive channel swamped the real signal (~40% apparent error rate).
# The lookarounds reject exponents, subscripts, LaTeX braces, and chained
# equalities, where the visible digits are not the whole computation.
_NUM = r"\d[\d,]*(?:\.\d+)?"
_OP = r"(?:[*+\-/]|\\times|\\cdot|\\div|×)"
ARITHMETIC = re.compile(
    rf"(?<![\w.^_{{}}])({_NUM}(?:[ \t]*{_OP}[ \t]*{_NUM})+)[ \t]*=[ \t]*({_NUM})"
    rf"(?![\w.]|[ \t]*[*+\-/=^])"
)
LATEX_OPEN = re.compile(r"\\\[")
LATEX_CLOSE = re.compile(r"\\\]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=pathlib.Path, required=True)
    parser.add_argument("--arms", nargs=2, default=["strict", "lenience0p2"], metavar=("CONTROL", "TREATMENT"))
    parser.add_argument("--labels", nargs=2, default=["strict", "lossy"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prefix-tokens",
        type=int,
        default=0,
        help="Score only the first N tokens of each output; 0 uses the shorter of each pair.",
    )
    parser.add_argument("--bucket-tokens", type=int, default=2000)
    parser.add_argument("--min-operand-digits", type=int, default=1)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    return parser.parse_args()


def encoder():
    from openai_harmony import HarmonyEncodingName, load_harmony_encoding

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    def encode(text: str) -> list[int]:
        # The archived outputs carry Harmony control tokens; they are part of
        # what the model emitted, so they are counted, not stripped.
        return encoding.encode(text, allowed_special="all")

    return encoding, encode


def evaluate_chain(expression: str) -> float | None:
    """Value of a digits-and-operators chain, or None if it is not decidable.

    eval is safe here only because the string is rebuilt from a whitelist: the
    regex admits digits, commas, dots and operators, and this re-checks after
    normalising the LaTeX spellings.
    """
    normalised = (
        expression.replace(",", "")
        .replace("\\times", "*")
        .replace("\\cdot", "*")
        .replace("\\div", "/")
        .replace("×", "*")
    )
    if not re.fullmatch(r"[\d.\s*+\-/]+", normalised):
        return None
    try:
        return eval(normalised, {"__builtins__": {}}, {})  # noqa: S307 - whitelisted above
    except (SyntaxError, ZeroDivisionError, TypeError, ValueError, OverflowError):
        return None


def check_arithmetic(text: str, min_digits: int) -> dict[str, Any]:
    checked = wrong = 0
    big_checked = big_wrong = 0
    examples: list[str] = []
    for match in ARITHMETIC.finditer(text):
        chain, claimed_text = match.groups()
        truth = evaluate_chain(chain)
        if truth is None:
            continue
        try:
            claimed = float(claimed_text.replace(",", ""))
        except ValueError:
            continue
        operands = [int(x) for x in re.findall(r"\d[\d,]*", chain.replace(",", "")) if x.isdigit()]
        if not operands or max(len(str(x)) for x in operands) < min_digits:
            continue
        # A non-integral truth against an integer claim is usually a rounded or
        # truncated statement, not an error; leave those out.
        if truth != int(truth) and claimed == int(claimed):
            continue
        checked += 1
        is_big = max(operands) >= 1000
        big_checked += is_big
        if abs(truth - claimed) > 1e-9:
            wrong += 1
            big_wrong += is_big
            if len(examples) < 3:
                examples.append(f"{match.group(0).strip()}  (= {truth:g})")
    return {
        "arith_checked": checked,
        "arith_wrong": wrong,
        "arith_error_rate": (wrong / checked) if checked else None,
        "arith_big_checked": big_checked,
        "arith_big_wrong": big_wrong,
        "arith_big_error_rate": (big_wrong / big_checked) if big_checked else None,
        "arith_examples": examples,
    }


def check_repetition(text: str, n: int = 10) -> dict[str, Any]:
    words = text.split()
    if len(words) < n * 2:
        return {"novelty_10gram": None, "longest_repeat_words": 0, "repeated_line_max": 0}
    grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    novelty = len(counts) / len(grams)

    # Longest span that repeats back-to-back, which is what a degenerate loop
    # looks like; a merely repetitive argument re-uses phrases far apart.
    longest = 0
    for size in (200, 100, 50, 25, 12, 6):
        if size * 2 > len(words):
            continue
        step = max(1, size // 4)
        for start in range(0, len(words) - 2 * size + 1, step):
            if words[start : start + size] == words[start + size : start + 2 * size]:
                longest = size
                break
        if longest:
            break

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    line_counts = Counter(lines)
    return {
        "novelty_10gram": novelty,
        "longest_repeat_words": longest,
        "repeated_line_max": max(line_counts.values()) if line_counts else 0,
    }


def check_surface(text: str) -> dict[str, Any]:
    if not text:
        return {}
    words = text.split()
    alpha_words = [w for w in words if any(ch.isalpha() for ch in w)]
    return {
        "nonascii_rate": sum(1 for ch in text if ord(ch) > 127) / len(text),
        "latex_bracket_imbalance": abs(len(LATEX_OPEN.findall(text)) - len(LATEX_CLOSE.findall(text))),
        "brace_imbalance": abs(text.count("{") - text.count("}")),
        "dollar_parity_odd": text.count("$") % 2,
        "mean_word_chars": statistics.fmean(len(w) for w in words) if words else None,
        "alpha_word_frac": len(alpha_words) / len(words) if words else None,
    }


def score(text: str, min_digits: int) -> dict[str, Any]:
    out: dict[str, Any] = {"chars": len(text), "words": len(text.split())}
    out.update(check_arithmetic(text, min_digits))
    out.update(check_repetition(text))
    out.update(check_surface(text))
    return out


def per_1k(value: float | None, tokens: int) -> float | None:
    if value is None or not tokens:
        return None
    return 1000.0 * value / tokens


def paired_table(pairs: list[tuple[dict, dict]], labels: list[str], fields: list[tuple[str, str, bool]]) -> str:
    lines = [f"| metric | {labels[0]} | {labels[1]} | ratio | {labels[1]} higher |", "|---|---:|---:|---:|---:|"]
    for key, name, higher_is_worse in fields:
        a = [p[0][key] for p in pairs if p[0].get(key) is not None and p[1].get(key) is not None]
        b = [p[1][key] for p in pairs if p[0].get(key) is not None and p[1].get(key) is not None]
        if not a:
            continue
        ma, mb = statistics.fmean(a), statistics.fmean(b)
        higher = sum(1 for x, y in zip(a, b) if y > x)
        ratio = f"{mb / ma:.2f}x" if ma else "-"
        lines.append(f"| {name} | {ma:.4g} | {mb:.4g} | {ratio} | {higher}/{len(a)} |")
    return "\n".join(lines)


def sign_test(pairs: list[tuple[float, float]]) -> tuple[int, int, float]:
    up = sum(1 for x, y in pairs if y > x)
    down = sum(1 for x, y in pairs if y < x)
    n = up + down
    if not n:
        return up, down, 1.0
    tail = sum(math.comb(n, k) for k in range(min(up, down) + 1))
    return up, down, min(1.0, 2 * tail / 2**n)


def main() -> int:
    args = parse_args()
    _, encode = encoder()
    control, treatment = args.arms

    cases = sorted({path.parent.parent.parent.name for path in args.runs_root.glob(f"*/seed_{args.seed}/*/output.txt")})
    rows: list[dict[str, Any]] = []
    for case in cases:
        base = args.runs_root / case / f"seed_{args.seed}"
        texts = {}
        for arm in (control, treatment):
            path = base / arm / "output.txt"
            if not path.is_file():
                break
            texts[arm] = path.read_text(encoding="utf-8")
        if len(texts) != 2:
            continue

        tokens = {arm: encode(text) for arm, text in texts.items()}
        limit = args.prefix_tokens or min(len(tokens[control]), len(tokens[treatment]))
        entry: dict[str, Any] = {"case": case, "prefix_tokens": limit}
        for arm in (control, treatment):
            full = texts[arm]
            head = _decode_prefix(full, tokens[arm], limit)
            entry[arm] = {
                "tokens_total": len(tokens[arm]),
                "prefix": score(head, args.min_operand_digits),
                "first_bucket": score(_decode_prefix(full, tokens[arm], args.bucket_tokens), args.min_operand_digits),
                "last_bucket": score(_decode_suffix(full, tokens[arm], args.bucket_tokens), args.min_operand_digits),
            }
        rows.append(entry)

    if not rows:
        print(f"no paired runs under {args.runs_root} for arms {control}/{treatment}", file=sys.stderr)
        return 1

    print(f"# Coherence checks: {control} vs {treatment}, {len(rows)} paired problems, seed {args.seed}\n")
    print(
        f"Matched prefixes: each pair scored over its first "
        f"{'min(len_a, len_b)' if not args.prefix_tokens else args.prefix_tokens} tokens "
        f"(median {int(statistics.median(r['prefix_tokens'] for r in rows)):,}).\n"
    )

    for section, key in (("Matched prefix", "prefix"), (f"First {args.bucket_tokens} tokens", "first_bucket"), (f"Last {args.bucket_tokens} tokens", "last_bucket")):
        pairs = []
        for row in rows:
            a = dict(row[control][key])
            b = dict(row[treatment][key])
            for src in (a, b):
                src["arith_wrong_per_1k_words"] = per_1k(src.get("arith_wrong"), src.get("words") or 0)
                src["arith_checked_per_1k_words"] = per_1k(src.get("arith_checked"), src.get("words") or 0)
            pairs.append((a, b))
        fields = [
            ("arith_error_rate", "arithmetic claims wrong (rate)", True),
            ("arith_big_error_rate", "  ... with an operand >= 1000", True),
            ("arith_checked_per_1k_words", "arithmetic claims made /1k words", False),
            ("novelty_10gram", "10-gram novelty (1.0 = no repeats)", False),
            ("longest_repeat_words", "longest back-to-back repeat (words)", True),
            ("repeated_line_max", "max repeats of any one line", True),
            ("nonascii_rate", "non-ASCII char rate", True),
            ("latex_bracket_imbalance", "unclosed \\[ ... \\]", True),
            ("mean_word_chars", "mean word length (chars)", False),
            ("alpha_word_frac", "fraction of words containing letters", False),
        ]
        print(f"\n## {section}\n")
        print(paired_table(pairs, args.labels, fields))

    print("\n## Sign tests on the matched prefix\n")
    print("| metric | up | down | two-sided p |")
    print("|---|---:|---:|---:|")
    for key, name in (
        ("arith_error_rate", "arithmetic error rate"),
        ("novelty_10gram", "10-gram novelty"),
        ("nonascii_rate", "non-ASCII rate"),
    ):
        values = [
            (row[control]["prefix"][key], row[treatment]["prefix"][key])
            for row in rows
            if row[control]["prefix"].get(key) is not None and row[treatment]["prefix"].get(key) is not None
        ]
        up, down, p = sign_test(values)
        print(f"| {name} | {up} | {down} | {p:.3f} |")

    worst = sorted(
        (r for r in rows if r[treatment]["prefix"].get("arith_error_rate") is not None),
        key=lambda r: -(r[treatment]["prefix"]["arith_error_rate"] - (r[control]["prefix"].get("arith_error_rate") or 0)),
    )[:5]
    print("\n## Largest arithmetic gaps (matched prefix)\n")
    print(f"| case | {args.labels[0]} wrong/checked | {args.labels[1]} wrong/checked | example from {args.labels[1]} |")
    print("|---|---:|---:|---|")
    for row in worst:
        a, b = row[control]["prefix"], row[treatment]["prefix"]
        example = (b["arith_examples"] or ["-"])[0].replace("|", "\\|")
        print(f"| {row['case']} | {a['arith_wrong']}/{a['arith_checked']} | {b['arith_wrong']}/{b['arith_checked']} | `{example}` |")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


def _decode_prefix(text: str, tokens: list[int], limit: int) -> str:
    """First `limit` tokens as text. Slices the string, not the token list, so
    the result is exactly what the model emitted up to that point."""
    if limit >= len(tokens):
        return text
    from openai_harmony import HarmonyEncodingName, load_harmony_encoding

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return encoding.decode(tokens[:limit])


def _decode_suffix(text: str, tokens: list[int], limit: int) -> str:
    if limit >= len(tokens):
        return text
    from openai_harmony import HarmonyEncodingName, load_harmony_encoding

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return encoding.decode(tokens[-limit:])


if __name__ == "__main__":
    raise SystemExit(main())
