#!/usr/bin/env python3
"""Per-layer phase timeline from a --trace dump: for one layer, when each
task kind's first descriptor became ready and its last one signalled,
relative to the layer's first ready descriptor. Shows which phases are
serial, how long each takes, and the gaps between them.

    python3 scripts/trace_timeline.py results/trace_d8.bin build/taskgraph_d8.bin --layer 5
"""
import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "host"))
from taskgraph import PACK_FIELDS, PACK_FORMAT, TaskKind  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("trace", type=Path)
ap.add_argument("graph", type=Path)
ap.add_argument("--layer", type=int, default=5)
a = ap.parse_args()

blob = a.graph.read_bytes()
tasks = [dict(zip(PACK_FIELDS, struct.unpack_from(PACK_FORMAT, blob, 64 * i)))
         for i in range(len(blob) // 64)]
tr = a.trace.read_bytes()
n = len(tr) // 32
stamps = [struct.unpack_from("<4Q", tr, 32 * i) for i in range(n)]
assert n == len(tasks), (n, len(tasks))

rows = [(t, s) for t, s in zip(tasks, stamps) if t["layer"] == a.layer and s[0]]
t0 = min(s[1] for _, s in rows)
by_kind = {}
for t, s in rows:
    k = by_kind.setdefault(t["kind"], [10**30, 0, 0, 0.0, 0, 0.0])
    k[0] = min(k[0], s[1]); k[1] = max(k[1], s[2]); k[2] = max(k[2], s[1])
    k[3] += s[2] - s[1]; k[4] += 1; k[5] += s[3] >> 8
us = 0.01
print(f"layer {a.layer}: {len(rows)} descriptors, span {(max(s[2] for _, s in rows) - t0) * us:.1f} us")
print(f"  {'kind':<15} {'first-ready':>11} {'last-ready':>10} {'last-done':>10} {'avg-busy':>9} {'prologue':>9}  n")
for kind, (fr, ld, lr, busy, cnt, pro) in sorted(by_kind.items(), key=lambda kv: kv[1][0]):
    print(f"  {TaskKind(kind).name:<15} {(fr - t0) * us:>9.1f}us {(lr - t0) * us:>8.1f}us "
          f"{(ld - t0) * us:>8.1f}us {busy * us / cnt:>7.1f}us {pro * us / cnt:>7.1f}us  {cnt}")
