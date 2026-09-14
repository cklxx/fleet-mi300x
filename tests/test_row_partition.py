#!/usr/bin/env python3
"""Every output row of every GEMV is owned by exactly one wave.

`gemv.h` splits a Chiplet-task's rows across XCDs, then workers, then the
four waves of a workgroup, as a pure function of (xcd, worker, wave) — no row
range is stored anywhere. That arithmetic is mirrored here line for line and
checked against every real matrix shape in the model: a row owned twice is a
race, a row owned by no one is a silent zero in the output, and neither shows
up as a compile error or a hang.

    python3 tests/test_row_partition.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "host"))
from taskgraph import WORKERS_PER_XCD, XCDS  # noqa: E402

WORKERS, WAVES = WORKERS_PER_XCD, 4


def rows_of(N: int, n_xcds: int, n_workers: int, xcd: int, worker: int,
            wave: int, pairs: bool) -> list[int]:
    """Mirror of gemv_rows (pairs=True) / gemv_gate_up_rows (pairs=False)."""
    per = (N + n_xcds - 1) // n_xcds
    begin = xcd * per
    end = min(N, begin + per)
    first = begin + worker
    if first >= end:
        return []
    cnt = (end - first + n_workers - 1) // n_workers
    out = []
    if pairs:
        for k in range(2 * wave, cnt, 2 * WAVES):
            out.append(first + k * n_workers)
            if k + 1 < cnt:
                out.append(first + (k + 1) * n_workers)
    else:
        for k in range(wave, cnt, WAVES):
            out.append(first + k * n_workers)
    return out


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {label:<44} [{'PASS' if ok else 'FAIL'}]{('  ' + detail) if detail else ''}")
    return ok


def covered_once(N: int, n_xcds: int, n_workers: int, pairs: bool) -> tuple[bool, str]:
    seen: dict[int, tuple] = {}
    for xcd in range(n_xcds):
        for worker in range(n_workers):
            for wave in range(WAVES):
                for r in rows_of(N, n_xcds, n_workers, xcd, worker, wave, pairs):
                    if r in seen:
                        return False, f"row {r} owned by {seen[r]} and {(xcd, worker, wave)}"
                    if not 0 <= r < N:
                        return False, f"row {r} outside [0, {N})"
                    seen[r] = (xcd, worker, wave)
    missing = N - len(seen)
    return missing == 0, f"{missing} rows unowned" if missing else f"{len(seen)}/{N} rows, each once"


def main() -> int:
    # (label, N, n_xcds, n_workers, pairs) for every GEMV the kernel runs
    cases = [
        ("qkv fused: 3072+576 over 8 XCDs", 3648, XCDS, WORKERS, True),
        ("o_proj: 2048 over 8 XCDs", 2048, XCDS, WORKERS, True),
        ("lm_head: 102400 over 8 XCDs", 102400, XCDS, WORKERS, True),
        ("dense down: 2048 over 8 XCDs", 2048, XCDS, WORKERS, True),
        ("dense gate_up: 10944 pairs over 8 XCDs", 10944, XCDS, WORKERS, False),
        ("expert gate_up: 1408 pairs, one XCD", 1408, 1, WORKERS, False),
        ("expert down: 2048 rows, one XCD", 2048, 1, WORKERS, True),
        ("router: 64 rows, one workgroup", 64, 1, 1, True),
        ("merge W_UV: 128 rows, one workgroup", 128, 1, 1, True),
        ("odd count: 37 rows, one workgroup", 37, 1, 1, True),
        ("fewer rows than workers: 5 over 8 XCDs", 5, XCDS, WORKERS, True),
    ]
    results = []
    for label, N, nx, nw, pairs in cases:
        ok, detail = covered_once(N, nx, nw, pairs)
        results.append(check(label, ok, detail))

    # Balance: the largest per-worker share must not exceed the smallest by
    # more than one row pair, or one XCD's tail dominates the phase (§9).
    shares = [sum(len(rows_of(102400, XCDS, WORKERS, x, w, v, True)) for v in range(WAVES))
              for x in range(XCDS) for w in range(WORKERS)]
    results.append(check("lm_head per-worker share balanced",
                         max(shares) - min(shares) <= 2,
                         f"{min(shares)}..{max(shares)} rows per worker"))

    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
