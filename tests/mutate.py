#!/usr/bin/env python3
"""Mutation gate: break the code on purpose, require the suite to notice.

A green suite proves nothing by itself. The audit that prompted this file
found two of our seven tests could not fail at all: both re-implemented the
production formula in Python and then compared the mirror with itself. The
fix was to pin the mirrors to their headers, and this file is how that fix
stays honest.

Each row edits a copy of the tree the way a careless change would, runs one
test, and requires the expected outcome:

    caught      the test must fail (this is the normal expectation)
    redundant   the test must still pass, because a second check in the
                production code covers the same mistake; a companion row
                removes both and must be caught

    python3 tests/mutate.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = "src/runtime/fleet_runtime.h"
GEMV = "src/kernels/gemv.h"
EXPERT = "src/kernels/expert.h"
GRAPH = "src/host/taskgraph.py"

PER_TASK = "if producers - {t.xcd}:"
PER_EVENT = 'if g.events[e]["scope"] == int(Scope.XCD_LOCAL) and len(xs) > 1:'

# (test, [(file, find, replace)], label, expectation)
MUTATIONS = [
    ("kernel_interface", [(RUNTIME, "if (event_release_fenced(rt, event)) fence_release();",
                           "if (event_release_fenced(rt, event)) { }")],
     "drop the producer-side release", "caught"),
    ("kernel_interface", [(RUNTIME, "if (scope == SCOPE_XCD_LOCAL) fence_acquire_local();",
                           "if (scope == SCOPE_XCD_LOCAL) { }")],
     "drop the XCD-local acquire", "caught"),
    ("kernel_interface", [(RUNTIME, r'lgkmcnt(0)\n\tbuffer_inv sc0', r'lgkmcnt(0)\n\tbuffer_inv sc1')],
     "widen the L1-only invalidate to L2", "caught"),
    ("kernel_interface", [(RUNTIME, "return true;   // see above: measured necessary",
                           "return !rt.coherent_acts;")],
     "make the producer writeback conditional", "caught"),
    ("row_partition", [(RUNTIME, "constexpr int kCUsPerXCD = 38;",
                        "constexpr int kCUsPerXCD = 37;")],
     "change CUs per XCD in the header only", "caught"),
    ("row_partition", [(GEMV, "s.begin = xcd * per;", "s.begin = xcd * per + 0;")],
     "alter the xcd_rows split formula", "caught"),
    ("expert_addressing", [(EXPERT, "(int64_t)half * 2 * rows * hidden",
                            "(int64_t)half * rows * hidden")],
     "reintroduce the shared-half offset bug", "caught"),
    ("validator", [(GRAPH, "if (t.flags & Flags.OPROJ_ROW_SPLIT) and t.kind == TaskKind.O_PROJ:", "if False:")],
     "delete the row-split o_proj global-merge check", "caught"),
    ("validator", [(GRAPH, PER_TASK, "if False:")],
     "delete the per-task XCD-locality check", "redundant"),
    ("validator", [(GRAPH, PER_EVENT, "if False:")],
     "delete the per-event XCD-locality check", "redundant"),
    ("validator", [(GRAPH, PER_TASK, "if False:"), (GRAPH, PER_EVENT, "if False:")],
     "delete both XCD-locality checks", "caught"),
    ("validator", [(GRAPH, "signalled[t.local_event + c] = signalled.get(t.local_event + c, 0) + WAVES",
                    "signalled[t.local_event + c] = signalled.get(t.local_event + c, 0) + 1")],
     "miscount the per-wave chunk arrivals", "caught"),
    ("validator", [(GRAPH, "if t.wait_count != ev[\"producers\"] or int(t.wait_scope) != ev[\"scope\"]:",
                    "if False:")],
     "stop cross-checking the wait-side fields", "caught"),
]


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="fleet-mutate-"))
    for d in ("src", "tests", "reference"):
        shutil.copytree(ROOT / d, tmp / d)

    rows = []
    for test, edits, label, expect in MUTATIONS:
        originals = {}
        applied = True
        for rel, old, new in edits:
            f = tmp / rel
            s = f.read_text()
            originals.setdefault(rel, s)
            if old not in s:
                applied = False
                break
            f.write_text(s.replace(old, new, 1))
        if not applied:
            for rel, s in originals.items():
                (tmp / rel).write_text(s)
            rows.append((label, test, "pattern gone",
                         expect == "skip-if-absent"))
            continue
        rc = subprocess.run([sys.executable, str(tmp / "tests" / f"test_{test}.py")],
                            cwd=tmp, capture_output=True, text=True).returncode
        for rel, s in originals.items():
            (tmp / rel).write_text(s)
        got = "caught" if rc != 0 else "survived"
        ok = (got == "caught") if expect in ("caught", "skip-if-absent") else (got == "survived")
        rows.append((label, test, got, ok))

    shutil.rmtree(tmp, ignore_errors=True)
    w = max(len(r[0]) for r in rows)
    print()
    for label, test, got, ok in rows:
        print(f"  {'OK  ' if ok else 'BAD '} {label:<{w}}  {test:<18} {got}")
    bad = [r[0] for r in rows if not r[3]]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} mutations behaved as expected"
          + (f" — unexpected: {', '.join(bad)}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
