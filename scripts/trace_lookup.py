#!/usr/bin/env python3
"""Map any text in output.txt back to how that token was produced.

Answers, for a span of generated text: was the token drafted and accepted, was
it a recovered token after a rejection -- and if accepted, would the strict rule
have rejected it? Reports the distribution features recorded at that position.

    # metrics for the 14th "wait" in the output
    scripts/trace_lookup.py RUN_DIR --find " wait" --occurrence 14

    # every occurrence, one line each
    scripts/trace_lookup.py RUN_DIR --find " wait" --summary

    # every token strict would have rejected
    scripts/trace_lookup.py RUN_DIR --lossy-only

Alignment
---------
Two things make naive alignment wrong, both found the hard way:

1. Re-encoding output.txt does NOT reproduce the emitted token ids. BPE is not
   round-trip stable -- in case_002 the model emitted tokens (13, 30) which
   re-encode as the single token 100003. The trace ids are ground truth and the
   text is their rendering, never the other way round.
2. Per-token decode does not concatenate to the whole decode, because multi-byte
   UTF-8 characters can straddle a token boundary. Offsets are therefore
   computed in BYTES via tiktoken's decode_single_token_bytes, which is exact.

The first output token comes from prefill and is never verified, so the trace
starts one token in. A run whose trace does not reproduce the tail of output.txt
byte-for-byte is refused rather than mis-attributed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=pathlib.Path)
    p.add_argument("--find", help="Literal text to locate (e.g. ' wait').")
    p.add_argument("--occurrence", type=int, help="1-based: show only this occurrence of --find.")
    p.add_argument("--summary", action="store_true", help="One line per occurrence instead of full detail.")
    p.add_argument("--lossy-only", action="store_true", help="List tokens accepted only because of lenience.")
    p.add_argument("--token-range", nargs=2, type=int, metavar=("START", "END"))
    p.add_argument("--context", type=int, default=4, help="Tokens of context around a hit.")
    p.add_argument("--limit", type=int, default=60)
    p.add_argument("--export", type=pathlib.Path, help="Write the aligned per-token table as JSONL here.")
    return p.parse_args()


def align(run_dir: pathlib.Path):
    """Per-token records with byte offsets into output.txt, or exit if unalignable."""
    import tiktoken

    enc = tiktoken.get_encoding("o200k_harmony")
    trace_path = run_dir / "proposals.jsonl"
    if not trace_path.is_file():
        raise SystemExit(f"no proposals.jsonl in {run_dir}; re-run with --trace-proposals")
    rows = [json.loads(x) for x in trace_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows.sort(key=lambda r: r["output_position"])
    ids = [r["emitted_token_id"] for r in rows]
    raw = (run_dir / "output.txt").read_text(encoding="utf-8").encode("utf-8")

    # A round is verified as a block; if the request stops mid-round the trailing
    # tokens are never emitted. Drop the smallest number that makes it align.
    committed = None
    for drop in range(0, 32):
        end = len(ids) - drop
        if end <= 0:
            break
        blob = b"".join(enc.decode_single_token_bytes(t) for t in ids[:end])
        if raw.endswith(blob):
            committed = (end, len(raw) - len(blob))
            break
    if committed is None:
        raise SystemExit(
            f"{run_dir}: trace does not reproduce the tail of output.txt; refusing to attribute tokens"
        )
    n_committed, prefill_bytes = committed

    out, cursor = [], prefill_bytes
    for k in range(n_committed):
        piece = enc.decode_single_token_bytes(ids[k])
        rec = dict(rows[k])
        rec["byte_start"], rec["byte_end"] = cursor, cursor + len(piece)
        rec["token_index"] = k + 1  # index in output.txt token stream (0 = prefill)
        rec["text"] = piece.decode("utf-8", errors="replace")
        out.append(rec)
        cursor += len(piece)
    return enc, raw, out, len(ids) - n_committed, prefill_bytes


def label(r: dict) -> str:
    if r["emission_source"] == "bonus":
        return "bonus"
    if r.get("lossy_only_accepted"):
        return "LOSSY-ONLY"
    if r.get("actually_accepted"):
        return "draft-acc"
    return "recovered"


def detail(r: dict) -> str:
    head = f"  tok {r['token_index']:>6} [{r['byte_start']:>7}] {r['text']!r:<16} {label(r):<11}"
    if r.get("p") is None:
        return head
    return head + (
        f" p={r['p']:<9.6f} q={r['q']:<9.6f} p/q={r['p_over_q']:<8.4f} u={r['u']:<8.5f}\n"
        f"{'':>14}H(p)={r['target_entropy']:<7.3f} H(q)={_f(r.get('draft_entropy')):<7} "
        f"KL(p||q)={_f(r.get('kl_target_draft')):<7} KL(q||p)={_f(r.get('kl_draft_target')):<7} "
        f"TV={_f(r.get('tv_distance')):<7} rank={r['target_rank']} run={r['consecutive_accepted_length']}"
    )


def _f(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def main() -> int:
    args = parse_args()
    enc, raw, recs, dropped, prefill = align(args.run_dir)
    lossy = [r for r in recs if r.get("lossy_only_accepted")]
    print(f"{args.run_dir}")
    print(f"  {len(recs)} verified tokens ({dropped} trailing rows discarded, {prefill}B prefill prefix)")
    print(f"  lossy-only accepted: {len(lossy)} ({100*len(lossy)/max(len(recs),1):.1f}%)\n")

    if args.export:
        with args.export.open("w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, separators=(",", ":")) + "\n")
        print(f"  wrote {args.export} ({len(recs)} rows)")
        return 0

    if args.lossy_only:
        for r in lossy[: args.limit]:
            print(detail(r))
        if len(lossy) > args.limit:
            print(f"  ... {len(lossy)-args.limit} more")
        return 0

    if args.token_range:
        a, b = args.token_range
        for r in recs:
            if a <= r["token_index"] <= b:
                print(detail(r))
        return 0

    if not args.find:
        print("give one of --find / --token-range / --lossy-only / --export", file=sys.stderr)
        return 2

    needle = args.find.encode("utf-8")
    hits, start = [], 0
    while True:
        k = raw.find(needle, start)
        if k < 0:
            break
        hits.append(k)
        start = k + 1
    if not hits:
        print(f"  not found: {args.find!r}")
        return 1
    print(f"  {len(hits)} occurrence(s) of {args.find!r}\n")

    chosen = range(len(hits)) if args.occurrence is None else [args.occurrence - 1]
    for h in chosen:
        if not 0 <= h < len(hits):
            print(f"  occurrence {h+1} out of range (1..{len(hits)})", file=sys.stderr)
            return 2
        bs = hits[h]
        be = bs + len(needle)
        span = [r for r in recs if r["byte_end"] > bs and r["byte_start"] < be]
        if not span:
            print(f"  #{h+1} at byte {bs}: falls in the prefill prefix (never verified)")
            continue
        if args.summary:
            r = span[0]
            print(f"  #{h+1:<4} tok {r['token_index']:>6} {label(r):<11} "
                  f"p={_f(r.get('p'))} p/q={_f(r.get('p_over_q'))} H(p)={_f(r.get('target_entropy'))} "
                  f"KL(p||q)={_f(r.get('kl_target_draft'))}")
            continue
        lo = max(0, recs.index(span[0]) - args.context)
        hi = min(len(recs), recs.index(span[-1]) + args.context + 1)
        ctx = raw[max(0, bs-60):be+60].decode("utf-8", errors="replace").replace("\n", "\\n")
        print(f"--- occurrence {h+1}/{len(hits)} at byte {bs}, tokens "
              f"{span[0]['token_index']}..{span[-1]['token_index']} ---")
        print(f"  context: ...{ctx}...")
        for r in recs[lo:hi]:
            mark = "  <<<" if r in span else ""
            print(detail(r) + mark)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
