"""Shared byte-exact alignment between output.txt and proposals.jsonl.

Extracted from scripts/trace_lookup.py so scripts/record_label.py uses the exact
same alignment logic rather than a second implementation that could drift from
it. See trace_lookup.py's module docstring for why this is byte-based rather
than a re-encode of output.txt (BPE is not round-trip stable) or a per-token
decode concatenation (multi-byte UTF-8 can straddle a token boundary).
"""

from __future__ import annotations

import json
import pathlib


def align(run_dir: pathlib.Path):
    """Return (raw_bytes, records) for a traced run, or (None, None) if untrustable.

    Each record: token_index (1-based, matches trace_lookup.py), output_position,
    byte_start, byte_end, lossy_only_accepted, actually_accepted, emission_source.
    """
    import tiktoken

    trace_path = run_dir / "proposals.jsonl"
    out_path = run_dir / "output.txt"
    if not (trace_path.is_file() and out_path.is_file()):
        return None, None

    enc = tiktoken.get_encoding("o200k_harmony")
    rows = [json.loads(x) for x in trace_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows.sort(key=lambda r: r["output_position"])
    ids = [r["emitted_token_id"] for r in rows]
    raw = out_path.read_text(encoding="utf-8").encode("utf-8")

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
    n_committed, prefill_bytes = committed

    out, cursor = [], prefill_bytes
    for k in range(n_committed):
        piece = enc.decode_single_token_bytes(ids[k])
        rec = dict(rows[k])
        rec["byte_start"], rec["byte_end"] = cursor, cursor + len(piece)
        rec["token_index"] = k + 1
        rec["text"] = piece.decode("utf-8", errors="replace")
        out.append(rec)
        cursor += len(piece)
    return raw, out


def token_at(recs: list[dict], byte_pos: int) -> dict | None:
    """Binary search: the record whose [byte_start, byte_end) contains byte_pos."""
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
