#!/usr/bin/env python3
"""Screen archived outputs for incoherence with the target model as judge.

This is a **screening pass for a human**, not a measurement. It ranks excerpts by
how degraded the judge thinks they are and prints file offsets, so the reading
can start with the worst 20 windows instead of 440,000 tokens. Treat every
verdict as a pointer, not a finding.

Design, and why each part is there:

  blind      the judge never sees which arm an excerpt came from, and in
             pairwise mode the A/B order is randomised per item and recorded, so
             a systematic position preference shows up instead of hiding.
  matched    excerpts are taken at the same *relative* position in both arms, so
             a 32k-token run is compared against a 5k-token run at the same
             stage of reasoning rather than at the same token offset.
  null check --null-check pairs two windows from the SAME arm. The judge should
             call those ties. Whatever rate it does not is the floor below which
             a real A/B difference cannot be read.
  rubric     concrete observable defects (non-sequitur, contradiction, garbled
             text, re-derivation) rather than "which is better", which invites
             the judge to reward length or confidence.

Known limitation, stated up front: the judge is the same model that produced the
text, so it is not independent. It can be trusted to spot *garbled or repetitive*
text; it cannot be trusted to certify that reasoning is valid. That is exactly
why the output is a triage queue for a human rather than a score.

Serve the judge on a plain server, with no speculation and no lenience:

    PYTHON=$PWD/.venv-vllm/bin/python bash remote/run_server_vllm.sh baseline
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from typing import Any

FINAL_MARKER = "<|channel|>final<|message|>"

RUBRIC = """You are auditing an excerpt from the middle of a language model's chain of thought \
on a competition maths problem.

The excerpt starts and ends mid-thought. Incompleteness, an unfinished sentence at \
either edge, no visible antecedent for the opening lines, terse note-taking, \
exploratory dead ends, and the absence of a final answer are ALL EXPECTED and are \
NOT defects. Working notes are supposed to look like working notes.

The default score is 2. Score below 2 only for a defect you can quote verbatim. \
If you cannot produce the quote, the score is 2.

Scale for each axis: 2 = no instance of this defect; 1 = one or two localised \
instances; 0 = the defect is pervasive through the excerpt.

- intelligible: garbled or corrupted text -- words or numbers spliced mid-token, \
broken notation, character salad, text that stops parsing as English or maths. \
Terse or abbreviated is not garbled.
- coherent: non-sequiturs, self-contradiction, or claims appearing from nowhere, \
judged ONLY on transitions visible inside the excerpt. Considering a case and \
rejecting it is normal reasoning, not a contradiction.
- progressing: re-deriving the SAME quantity repeatedly without ever resolving it, \
or restating the same plan over and over. Checking a result once or twice is \
normal and scores 2.
- on_task: drifted to an unrelated problem, or talking about itself instead of \
doing the maths. Exploring a sub-case of the stated problem is on task.

Then give:
- verdict: "clean" (all four are 2), "minor" (one axis at 1), or "degraded" \
(any axis at 0, or two or more axes below 2).
- evidence: one short verbatim quote from the excerpt justifying your lowest \
score, or "" if everything scored 2.

Reply with ONE JSON object and nothing else:
{"intelligible": 0-2, "coherent": 0-2, "progressing": 0-2, "on_task": 0-2, \
"verdict": "clean|minor|degraded", "evidence": "..."}"""

PAIRWISE_RUBRIC = """You are comparing two excerpts, A and B, from the middle of two \
language models' chains of thought on the SAME competition maths problem.

Both start and end mid-thought. Incompleteness, no visible antecedent for the \
opening lines, terse note-taking, exploratory dead ends and the absence of a final \
answer are EXPECTED and are NOT defects, in either excerpt. Length and verbosity \
differences are NOT defects. Judge only the reasoning quality visible inside each \
excerpt.

Most pairs are comparable. Answer A or B only if you can quote a specific defect \
in the worse one.

Which excerpt shows more degradation -- garbled or corrupted text, non-sequiturs, \
self-contradiction, or circling over the same derivation without resolving it?

Answer "A" or "B", or "tie" if they are comparably clean OR comparably degraded. \
Prefer "tie" when you are unsure; a wrong call is worse than an honest tie.

Reply with ONE JSON object and nothing else:
{"worse": "A|B|tie", "defect": "garbled|non_sequitur|contradiction|circling|none", \
"evidence": "one short verbatim quote from the worse excerpt, or empty"}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=pathlib.Path, required=True)
    parser.add_argument("--prompt-root", type=pathlib.Path, default=pathlib.Path("prompts/aime24"))
    parser.add_argument("--arms", nargs=2, default=["strict", "lenience0p2"], metavar=("CONTROL", "TREATMENT"))
    parser.add_argument("--labels", nargs=2, default=["strict", "lossy"])
    parser.add_argument("--seed-dir", type=int, default=0, help="Which seed_N directory to read.")
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument(
        "--positions",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75],
        help="Relative positions in each output to sample windows from.",
    )
    parser.add_argument("--window-tokens", type=int, default=700)
    parser.add_argument(
        "--mode",
        choices=("both", "absolute", "pairwise"),
        default="both",
        help="absolute: rate each window alone. pairwise: blind A/B on matched windows.",
    )
    parser.add_argument(
        "--null-check",
        action="store_true",
        help="Also run pairwise on same-arm window pairs, to expose judge A/B bias.",
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="gpt-oss-20b")
    # medium, not low: low effort scores almost everything 2 and misses localised
    # defects, which is the wrong trade for a screen where recall matters.
    parser.add_argument("--judge-effort", default="medium", choices=("low", "medium", "high"))
    parser.add_argument("--temperature", type=float, default=0.0)
    # Generous: the judge reasons in the analysis channel first, and a cap that
    # truncates before the final channel yields an unparseable judgement.
    parser.add_argument("--max-new-tokens", type=int, default=4000)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--rng-seed", type=int, default=0, help="Controls A/B order randomisation.")
    parser.add_argument("--out", type=pathlib.Path, default=None, help="Write judgements as JSONL here.")
    parser.add_argument("--triage", type=int, default=20, help="How many worst windows to list.")
    parser.add_argument("--dry-run", action="store_true", help="Build the work list, call nothing.")
    return parser.parse_args()


def harmony():
    from openai_harmony import (
        Conversation,
        HarmonyEncodingName,
        Message,
        ReasoningEffort,
        Role,
        SystemContent,
        load_harmony_encoding,
    )

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    def render(user_text: str, effort: str) -> str:
        system = SystemContent.new().with_reasoning_effort(ReasoningEffort(effort.capitalize()))
        conversation = Conversation.from_messages(
            [
                Message.from_role_and_content(Role.SYSTEM, system),
                Message.from_role_and_content(Role.USER, user_text),
            ]
        )
        return encoding.decode(encoding.render_conversation_for_completion(conversation, Role.ASSISTANT))

    return encoding, render


def post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ask(args: argparse.Namespace, render, user_text: str, budget_scale: float = 1.0) -> tuple[dict[str, Any] | None, str]:
    """One judgement. Returns (parsed json or None, raw final-channel text).

    budget_scale exists because a pairwise prompt carries two excerpts and the
    judge reasons over both before answering. At the single-excerpt budget it
    ran out mid-analysis on ~17% of pairs -- and that dropout is not random, it
    selects for the pairs that were hardest to judge, which is exactly the
    wrong thing to drop."""
    payload = {
        "model": args.model,
        "prompt": render(user_text, args.judge_effort),
        "temperature": args.temperature,
        "top_p": 1.0,
        "max_tokens": int(args.max_new_tokens * budget_scale),
        "seed": args.rng_seed,
        "add_special_tokens": False,
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        "stream": False,
    }
    response = post(f"{args.server_url.rstrip('/')}/v1/completions", payload, args.timeout)
    text = (response.get("choices") or [{}])[0].get("text", "")
    final = text.split(FINAL_MARKER)[-1] if FINAL_MARKER in text else text
    match = re.search(r"\{.*\}", final, re.DOTALL)
    if not match:
        return None, final
    try:
        return json.loads(match.group(0)), final
    except json.JSONDecodeError:
        return None, final


def window_at(encoding, tokens: list[int], fraction: float, size: int) -> tuple[str, int]:
    """Window centred on `fraction` through the output. Returns (text, start token)."""
    if len(tokens) <= size:
        return encoding.decode(tokens), 0
    centre = int(len(tokens) * fraction)
    start = max(0, min(len(tokens) - size, centre - size // 2))
    return encoding.decode(tokens[start : start + size]), start


def load_problem(prompt_root: pathlib.Path, case: str) -> str:
    try:
        source = json.loads((prompt_root / case / "source.json").read_text(encoding="utf-8"))
        return str(source.get("problem", "")).strip()
    except (OSError, json.JSONDecodeError):
        return ""


def main() -> int:
    args = parse_args()
    encoding, render = harmony()
    control, treatment = args.arms
    label = dict(zip(args.arms, args.labels))
    rng = random.Random(args.rng_seed)

    cases = args.cases or sorted(
        {p.parent.parent.parent.name for p in args.runs_root.glob(f"*/seed_{args.seed_dir}/*/output.txt")}
    )

    # Build every window first, so a run that dies mid-way still has a work list
    # and the judged set does not depend on how far it got.
    windows: list[dict[str, Any]] = []
    for case in cases:
        base = args.runs_root / case / f"seed_{args.seed_dir}"
        texts = {}
        for arm in args.arms:
            path = base / arm / "output.txt"
            if path.is_file():
                texts[arm] = encoding.encode(path.read_text(encoding="utf-8"), allowed_special="all")
        if len(texts) != 2:
            print(f"skip {case}: needs both arms", file=sys.stderr)
            continue
        problem = load_problem(args.prompt_root, case)
        for fraction in args.positions:
            for arm in args.arms:
                text, start = window_at(encoding, texts[arm], fraction, args.window_tokens)
                windows.append(
                    {
                        "case": case,
                        "arm": arm,
                        "label": label[arm],
                        "position": fraction,
                        "start_token": start,
                        "tokens_total": len(texts[arm]),
                        "text": text,
                        "problem": problem,
                    }
                )

    by_key = {(w["case"], w["position"], w["arm"]): w for w in windows}
    pairs = [
        (by_key[(c, f, control)], by_key[(c, f, treatment)])
        for c in cases
        for f in args.positions
        if (c, f, control) in by_key and (c, f, treatment) in by_key
    ]

    n_abs = len(windows) if args.mode in ("both", "absolute") else 0
    n_pair = len(pairs) if args.mode in ("both", "pairwise") else 0
    n_null = (len(pairs) if len(args.positions) > 1 else 0) if args.null_check else 0
    print(f"{len(cases)} cases, {len(args.positions)} positions, {args.window_tokens}-token windows")
    print(f"work: {n_abs} absolute + {n_pair} pairwise + {n_null} null-check = {n_abs + n_pair + n_null} judgements")
    if args.dry_run:
        return 0

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    done = 0
    total = n_abs + n_pair + n_null

    def progress() -> None:
        rate = (time.perf_counter() - started) / max(done, 1)
        print(f"  {done}/{total} judged ({rate:.1f}s each, ~{rate * (total - done) / 60:.0f} min left)", flush=True)

    if n_abs:
        for window in windows:
            user = (
                f"{RUBRIC}\n\nPROBLEM:\n{window['problem']}\n\n"
                f"EXCERPT (mid-reasoning):\n<<<\n{window['text']}\n>>>"
            )
            try:
                parsed, raw = ask(args, render, user)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                print(f"absolute {window['case']} {window['label']} failed: {exc}", file=sys.stderr)
                parsed, raw = None, ""
            records.append({"kind": "absolute", **{k: v for k, v in window.items() if k != "text"},
                            "judgement": parsed, "raw": raw[:2000], "parsed_ok": parsed is not None})
            done += 1
            if done % 10 == 0:
                progress()

    def run_pairwise(first: dict, second: dict, kind: str) -> None:
        nonlocal done
        flip = rng.random() < 0.5
        a, b = (second, first) if flip else (first, second)
        user = (
            f"{PAIRWISE_RUBRIC}\n\nPROBLEM:\n{first['problem']}\n\n"
            f"EXCERPT A:\n<<<\n{a['text']}\n>>>\n\nEXCERPT B:\n<<<\n{b['text']}\n>>>"
        )
        try:
            parsed, raw = ask(args, render, user, budget_scale=2.0)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"{kind} {first['case']} failed: {exc}", file=sys.stderr)
            parsed, raw = None, ""
        worse_label = None
        if parsed and parsed.get("worse") in ("A", "B"):
            worse_label = (a if parsed["worse"] == "A" else b)["label"]
        records.append(
            {
                "kind": kind,
                "case": first["case"],
                "position": first["position"],
                "a_label": a["label"],
                "b_label": b["label"],
                "a_start_token": a["start_token"],
                "b_start_token": b["start_token"],
                "judgement": parsed,
                "worse_label": worse_label,
                "raw": raw[:2000],
                "parsed_ok": parsed is not None,
            }
        )
        done += 1
        if done % 10 == 0:
            progress()

    if n_pair:
        for first, second in pairs:
            run_pairwise(first, second, "pairwise")

    if n_null:
        # Same arm, two different positions: a judge with no A/B bias calls these
        # ties at whatever rate genuinely-similar excerpts deserve.
        for case in cases:
            for fraction in args.positions:
                key = (case, fraction, control)
                other = next(
                    (f for f in args.positions if f != fraction and (case, f, control) in by_key),
                    None,
                )
                if key in by_key and other is not None:
                    run_pairwise(by_key[key], by_key[(case, other, control)], "null")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    report(args, records)
    if args.out:
        print(f"\nwrote {args.out}")
    return 0


def report(args: argparse.Namespace, records: list[dict[str, Any]]) -> None:
    absolute = [r for r in records if r["kind"] == "absolute" and r["parsed_ok"]]
    pairwise = [r for r in records if r["kind"] == "pairwise" and r["parsed_ok"]]
    nulls = [r for r in records if r["kind"] == "null" and r["parsed_ok"]]
    bad_parse = sum(1 for r in records if not r["parsed_ok"])

    print(f"\n# Judge screening: {len(records)} judgements, {bad_parse} unparseable\n")

    if absolute:
        axes = ("intelligible", "coherent", "progressing", "on_task")
        print("## Absolute ratings (0 = severe, 2 = no problem)\n")
        print("| arm | " + " | ".join(axes) + " | clean | minor | degraded |")
        print("|---|" + "|".join("---:" for _ in axes) + "|---:|---:|---:|")
        for label in args.labels:
            group = [r for r in absolute if r["label"] == label]
            if not group:
                continue
            means = []
            for axis in axes:
                values = [r["judgement"].get(axis) for r in group if isinstance(r["judgement"].get(axis), (int, float))]
                means.append(f"{sum(values)/len(values):.2f}" if values else "-")
            verdicts = Counter(str(r["judgement"].get("verdict")) for r in group)
            print(
                f"| {label} | " + " | ".join(means) + " | "
                f"{verdicts.get('clean', 0)} | {verdicts.get('minor', 0)} | {verdicts.get('degraded', 0)} |"
            )

    if pairwise:
        counts = Counter(r["worse_label"] for r in pairwise)
        decided = sum(v for k, v in counts.items() if k)
        ties = len(pairwise) - decided
        print(f"\n## Blind pairwise ({len(pairwise)} matched window pairs)\n")
        for label in args.labels:
            print(f"- judged worse, {label}: {counts.get(label, 0)}")
        print(f"- tie / no difference: {ties}")
        if decided:
            import math

            worse_treat = counts.get(args.labels[1], 0)
            p = 2 * sum(math.comb(decided, k) for k in range(min(worse_treat, decided - worse_treat) + 1)) / 2**decided
            print(f"- two-sided sign test on the {decided} decided pairs: p = {min(p, 1.0):.3f}")
        position = Counter(
            r["judgement"].get("worse") for r in pairwise if r["judgement"].get("worse") in ("A", "B")
        )
        print(f"- raw A/B split (position bias check): A={position.get('A', 0)} B={position.get('B', 0)}")
        defects = Counter(r["judgement"].get("defect") for r in pairwise if r["worse_label"])
        if defects:
            print("- defect cited: " + ", ".join(f"{k}={v}" for k, v in defects.most_common()))

    if nulls:
        decided = sum(1 for r in nulls if r["judgement"].get("worse") in ("A", "B"))
        print(f"\n## Null check (same arm, different positions): {decided}/{len(nulls)} called a winner")
        print("  Anything well above chance here means the judge separates excerpts that")
        print("  differ only in position, and the pairwise result cannot be read as an arm effect.")

    if absolute:
        print(f"\n## Triage queue: worst {args.triage} windows for a human to read\n")
        print("| severity | case | arm | position | tokens | evidence |")
        print("|---:|---|---|---:|---|---|")
        scored = []
        for r in absolute:
            judgement = r["judgement"]
            values = [judgement.get(a) for a in ("intelligible", "coherent", "progressing", "on_task")]
            values = [v for v in values if isinstance(v, (int, float))]
            if values:
                scored.append((sum(values), min(values), r))
        scored.sort(key=lambda t: (t[0], t[1]))
        for total, _, r in scored[: args.triage]:
            evidence = str(r["judgement"].get("evidence", ""))[:110].replace("|", "\\|").replace("\n", " ")
            span = f"{r['start_token']}-{r['start_token'] + args.window_tokens}/{r['tokens_total']}"
            print(f"| {total}/8 | {r['case']} | {r['label']} | {r['position']:.2f} | {span} | {evidence} |")
        print(
            "\nRead one with:\n"
            "  python -c \"import sys;from openai_harmony import *;"
            "e=load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS);"
            "t=e.encode(open(sys.argv[1]).read(),allowed_special='all');"
            "print(e.decode(t[int(sys.argv[2]):int(sys.argv[3])]))\" "
            "<output.txt> <start> <end>"
        )


if __name__ == "__main__":
    raise SystemExit(main())
