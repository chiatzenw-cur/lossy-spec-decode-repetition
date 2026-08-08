#!/usr/bin/env python3
"""For every loop whose escape token was recovered-sourced (draft proposal
rejected, recovery resampled from the residual [p-q]+ with the rejected
draft masked out), test whether the escape was STRUCTURALLY FORCED: did the
rejected draft proposal happen to equal the periodic-continuation token the
loop "expected" next?

Why this matters: under greedy drafting, q is ~one-hot on the draft's own
proposal. If the drafter proposes the SAME token the loop is repeating
(draft == expected) and that draft gets rejected anyway (an unlucky u draw,
since acceptance is still probabilistic even when p(x) is high), the residual
sampler's mask (vocab_offset != draft_token_id) makes it IMPOSSIBLE for
recovery to reselect that token -- the escape is then guaranteed by
construction, not by any change in the target's preference. If instead the
drafter proposed something ELSE (draft != expected) and that got rejected,
recovery is free to reselect the loop token and typically does (see
collect_loop_token_metrics's emitted_p/emitted_target_rank on recovered
in-loop rows: rank 0 97.5% of the time) -- so a recovered ESCAPE in that case
means recovery genuinely sampled away from the loop, a different mechanism.

No GPU/replay needed: draft_token_id and emitted_token_id are already in
proposals.jsonl for every event; this is pure re-derivation from existing
traces via lib_trace_align.align().

Run from repo root. Output: data/loop_token_metrics/recovered_escape_mechanism.json
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, "scripts")
from lib_trace_align import align  # noqa: E402

import tiktoken  # noqa: E402
_ENC = tiktoken.get_encoding("o200k_harmony")


def decode_token(token_id: int | None) -> str | None:
    if token_id is None:
        return None
    return _ENC.decode_single_token_bytes(token_id).decode("utf-8", errors="replace")


LOOPS = pathlib.Path("data/loop_token_metrics/all_relaxed_greedy_loops.jsonl")
JUDGED = [
    "data/adjacent_loop_judgements/aime24_lenience0p05Greedy12k_full.jsonl",
    "data/adjacent_loop_judgements/humaneval_lenience0p05Greedy9k_full.jsonl",
    "data/adjacent_loop_judgements/longbench_v2_greedy_relaxed_full.jsonl",
]
OUT = pathlib.Path("data/loop_token_metrics/recovered_escape_mechanism.json")


def main() -> int:
    rows = [json.loads(l) for l in LOOPS.open()]

    runs_root_by_key: dict[tuple, str] = {}
    for jf in JUDGED:
        for line in open(jf):
            r = json.loads(line)
            runs_root_by_key[(r["case"], r["tag"], r["loop_token_start"], r["loop_token_end"])] = r["runs_root"]

    events = []
    run_cache: dict[tuple, tuple] = {}
    missing = 0
    for r in rows:
        esc = r.get("escape_token")
        if esc is None or esc.get("emission_source") != "recovered":
            continue
        key = (r["case"], r["tag"], r["loop_token_start"], r["loop_token_end"])
        runs_root = runs_root_by_key.get(key)
        if runs_root is None:
            missing += 1
            continue
        run_key = (runs_root, r["case"], r["tag"])
        if run_key not in run_cache:
            run_dir = pathlib.Path(runs_root) / r["case"] / "seed_0" / r["tag"]
            raw, records = align(run_dir)
            run_cache[run_key] = (raw, records)
        raw, records = run_cache[run_key]
        if records is None:
            missing += 1
            continue

        escape_token_index = r["loop_token_end"] + 1
        period = max(1, r["match_length_tokens"])
        expected_index = escape_token_index - period
        if expected_index < 1 or expected_index > len(records):
            missing += 1
            continue

        esc_rec = records[escape_token_index - 1]
        exp_rec = records[expected_index - 1]
        if esc_rec.get("token_index") != escape_token_index or exp_rec.get("token_index") != expected_index:
            missing += 1
            continue

        rejected_draft_id = esc_rec.get("draft_token_id")
        emitted_id = esc_rec.get("emitted_token_id")
        expected_id = exp_rec.get("emitted_token_id")

        events.append({
            "case": r["case"], "tag": r["tag"], "runs_root": runs_root,
            "escape_token_index": escape_token_index, "period": period,
            "rejected_draft_token_id": rejected_draft_id,
            "rejected_draft_text": decode_token(rejected_draft_id),
            "emitted_token_id": emitted_id, "emitted_text": esc_rec.get("text"),
            "expected_token_id": expected_id, "expected_text": exp_rec.get("text"),
            "draft_was_expected": rejected_draft_id == expected_id,
            "u": esc_rec.get("u"), "p": esc_rec.get("p"), "q": esc_rec.get("q"),
            "target_rank_of_draft": esc_rec.get("target_rank"),
            "emitted_p": esc_rec.get("emitted_p"), "emitted_target_rank": esc_rec.get("emitted_target_rank"),
        })

    n = len(events)
    forced = sum(1 for e in events if e["draft_was_expected"])
    print(f"recovered-source escapes found: {n}  (skipped/unreadable: {missing})")
    print(f"structurally forced (rejected draft == expected loop token): {forced}/{n} = {100*forced/n:.1f}%" if n else "n=0")
    print(f"NOT structurally forced (rejected draft != expected loop token): {n-forced}/{n} = {100*(n-forced)/n:.1f}%" if n else "")

    if n:
        forced_events = [e for e in events if e["draft_was_expected"]]
        other_events = [e for e in events if not e["draft_was_expected"]]
        print("\n--- forced-escape examples (draft proposal WAS the loop token, got unlucky-rejected) ---")
        for e in forced_events[:5]:
            print(f"  {e['case']}/{e['tag']} @tok{e['escape_token_index']}: "
                  f"expected/draft={e['expected_text']!r} u={e['u']} p={e['p']} q={e['q']} "
                  f"-> emitted={e['emitted_text']!r}")
        print("\n--- non-forced escape examples (draft proposal was something else) ---")
        for e in other_events[:5]:
            print(f"  {e['case']}/{e['tag']} @tok{e['escape_token_index']}: "
                  f"expected={e['expected_text']!r} rejected_draft_id={e['rejected_draft_token_id']} "
                  f"-> emitted={e['emitted_text']!r} (emitted_p={e['emitted_p']}, emitted_rank={e['emitted_target_rank']})")

    out = {
        "n_recovered_escapes": n,
        "n_skipped": missing,
        "n_structurally_forced": forced,
        "share_structurally_forced": round(forced / n, 4) if n else None,
        "events": events,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
