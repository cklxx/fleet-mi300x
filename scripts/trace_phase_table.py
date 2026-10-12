#!/usr/bin/env python3
"""Per-phase microsecond table for one whole decode step, from a --trace dump.

`scripts/trace_timeline.py` shows one layer. This shows the step: every task kind
aggregated over all layers, so the reader gets the same shape of table the ETX
project uses (per-phase microseconds), plus the gap account.

    python3 scripts/trace_phase_table.py <trace> <graph> [--step-seconds]

Columns, all in microseconds and all measured on the device clock:
  n            tasks of this kind in the step
  busy         sum over tasks of (last-signal - first-ready): the work this kind occupies
  stage        sum of the staged-bytes time the task reports
  prologue     sum of the prologue time the task reports
  span         last-signal of the kind minus the step's first ready time (when the kind finishes)

The bottom lines are the account: sum(busy) is the body total, step span is the wall of the
step on the device clock, and the difference is time when the device was not running any of
these bodies, which is where waits, publication and gaps live.

A trace is an attribution screen, not a timing verdict: stamps perturb the schedule.
"""
import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "host"))
from taskgraph import PACK_FIELDS, PACK_FORMAT, TaskKind  # noqa: E402

US = 0.01  # trace unit -> microseconds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("graph", type=Path)
    ap.add_argument("--layers", type=int, default=None, help="limit to the first N layers of the step")
    a = ap.parse_args()

    blob = a.graph.read_bytes()
    tasks = [dict(zip(PACK_FIELDS, struct.unpack_from(PACK_FORMAT, blob, 64 * i)))
             for i in range(len(blob) // 64)]
    tr = a.trace.read_bytes()
    n = len(tr) // 32
    stamps = [struct.unpack_from("<4Q", tr, 32 * i) for i in range(n)]
    if n != len(tasks):
        print(f"trace has {n} records, graph has {len(tasks)} descriptors", file=sys.stderr)
        return 2

    rows = [(t, s) for t, s in zip(tasks, stamps) if s[0]]
    if not rows:
        print("trace has no stamped records", file=sys.stderr)
        return 2
    if a.layers is not None:
        keep = sorted({t["layer"] for t, _ in rows})[: a.layers]
        rows = [(t, s) for t, s in rows if t["layer"] in keep]

    t0 = min(s[1] for _, s in rows)
    step_us = (max(s[2] for _, s in rows) - t0) * US

    agg = {}
    for t, s in rows:
        k = agg.setdefault(t["kind"], dict(n=0, busy=0.0, stage=0.0, prologue=0.0, last=0))
        k["n"] += 1
        k["busy"] += (s[2] - s[1]) * US
        k["prologue"] += ((s[3] >> 32) & 0xFFFFFF) * US
        k["stage"] += ((s[3] >> 8) & 0xFFFFFF) * US
        k["last"] = max(k["last"], s[2])

    wide = max(len(TaskKind(k).name) for k in agg) if agg else 10
    print(f"step: {len(rows)} descriptors, {len({t['layer'] for t, _ in rows})} layers, "
          f"span {step_us:.1f} us")
    print(f"  {'phase'.ljust(wide)} {'n':>6} {'busy_us':>10} {'stage_us':>10} {'prologue_us':>11} {'finishes_at':>11}")
    order = sorted(agg.items(), key=lambda kv: kv[1]["last"])
    for kind, k in order:
        print(f"  {TaskKind(kind).name.ljust(wide)} {k['n']:>6} {k['busy']:>10.1f} "
              f"{k['stage']:>10.1f} {k['prologue']:>11.1f} {(k['last'] - t0) * US:>10.1f}")

    busy = sum(k["busy"] for k in agg.values())
    stage = sum(k["stage"] for k in agg.values())
    pro = sum(k["prologue"] for k in agg.values())
    print()
    print(f"  {'body total':<{wide}} {len(rows):>6} {busy:>10.1f} {stage:>10.1f} {pro:>11.1f}")
    print(f"  {'step span':<{wide}} {'':>6} {step_us:>10.1f}")
    print(f"  {'non-body time':<{wide}} {'':>6} {step_us - busy:>10.1f}   "
          f"(step span - body total; waits, publication, gaps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
