#!/usr/bin/env python3
"""Turn bonus_escape_replay_results.jsonl (target argmax at each bonus-escape
prefix) into the 4-way classification table that decides between the two
escape hypotheses:

  Modal switch        : argmax == actual   , argmax != expected
                         -- the target's OWN top pick had already moved off
                         the loop before the bonus sample was drawn.
  Stochastic escape    : argmax == expected , argmax != actual
                         -- the target still preferred the loop; the
                         unverified bonus draw just missed the mode.
  Nonmodal competition : argmax != both
                         -- neither is the target's own top pick.
  Same token            : filtered upstream by extract_bonus_escape_events.py
                          (actual == expected -- not a real escape, the
                          pattern's own boundary detector artifact).

Run from repo root, after extract_bonus_escape_events.py and
replay_bonus_escape_argmax.py. Output:
data/loop_token_metrics/bonus_escape_classification.json
"""
import json
import collections
import pathlib

IN = pathlib.Path("data/loop_token_metrics/bonus_escape_replay_results.jsonl")
OUT = pathlib.Path("data/loop_token_metrics/bonus_escape_classification.json")


def main() -> int:
    rows = [json.loads(l) for l in IN.open()]

    errors = [r for r in rows if r["argmax_text"] is None]
    ok = [r for r in rows if r["argmax_text"] is not None]
    print(f"total replayed: {len(rows)}  errors: {len(errors)}  usable: {len(ok)}")

    cat_counts: collections.Counter = collections.Counter()
    examples: dict[str, list] = collections.defaultdict(list)
    for r in ok:
        a = r["argmax_text"]
        exp_match = a == r["expected_text"]
        act_match = a == r["actual_text"]
        if exp_match and act_match:
            cat = "ambiguous_both_match"  # should be ~0: expected != actual by construction here
        elif act_match:
            cat = "modal_switch"
        elif exp_match:
            cat = "stochastic_escape"
        else:
            cat = "nonmodal_competition"
        cat_counts[cat] += 1
        if len(examples[cat]) < 5:
            examples[cat].append({
                "case": r["case"], "tag": r["tag"], "escape_token_index": r["escape_token_index"],
                "expected_text": r["expected_text"], "actual_text": r["actual_text"], "argmax_text": a,
            })

    n = len(ok)
    print(f"\n{'category':<24}{'n':>6}{'%':>8}")
    for cat in ("modal_switch", "stochastic_escape", "nonmodal_competition", "ambiguous_both_match"):
        c = cat_counts.get(cat, 0)
        print(f"{cat:<24}{c:>6}{100*c/n:>7.1f}%")

    modal = cat_counts.get("modal_switch", 0)
    stoch = cat_counts.get("stochastic_escape", 0)
    nonmodal = cat_counts.get("nonmodal_competition", 0)
    print(f"\nP(modal switch | bonus escape)        = {modal}/{n} = {100*modal/n:.1f}%")
    print(f"P(stochastic override | bonus escape) = {stoch}/{n} = {100*stoch/n:.1f}%")
    print(f"P(neither is target argmax | bonus escape) = {nonmodal}/{n} = {100*nonmodal/n:.1f}%")

    out = {
        "n_total_bonus_escapes_extracted": 463,
        "n_same_token_boundary_artifact": 6,
        "n_genuine_replayed": len(rows),
        "n_replay_errors": len(errors),
        "n_usable": n,
        "counts": dict(cat_counts),
        "shares_pct": {k: round(100 * v / n, 2) for k, v in cat_counts.items()},
        "examples": examples,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
