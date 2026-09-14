#!/usr/bin/env python3
"""Every gate that runs without a GPU, in one command.

Three kinds of gate, in the order a regression usually trips them:

  1. the seven unit tests (tests/test_*.py, found by glob so a new file is
     picked up without editing a list anywhere);
  2. the task graph's own validator on every variant the flags can build —
     a graph that fails validation would hang the kernel, and each flag
     combination wires different events;
  3. the HIP parse of the three .hip translation units, in both the device
     and the host pass, with and without the non-temporal weight build.

Exit code is 0 only if every gate passed. This is what setup_env.sh and
hotaisle_bootstrap.sh call before any machine time is bought.

    python3 tests/run_all.py            # everything
    python3 tests/run_all.py --quick    # unit tests + graphs, skip the parse
    python3 tests/run_all.py --mutate   # everything + the mutation gate
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# Every flag combination the graph builder supports. Each one wires a
# different set of events, so each one has to be validated separately.
GRAPH_VARIANTS = [
    ("default (16 KV chunks)", ["--kv-chunks", "16"]),
    ("8 KV chunks", ["--kv-chunks", "8"]),
    ("1 KV chunk", ["--kv-chunks", "1"]),
    ("kv_a replicated", ["--kv-chunks", "16", "--kva-replicated"]),
    ("top-k published", ["--kv-chunks", "16", "--topk-published"]),
    ("q_c per task", ["--kv-chunks", "16", "--qc-per-task"]),
    ("idle-worker prefetch", ["--kv-chunks", "16", "--prefetch"]),
    ("o_proj row split", ["--kv-chunks", "16", "--oproj-row-split"]),
    ("split workers + tiling", ["--kv-chunks", "16", "--split-workers", "18",
                                "--k-chunk", "512"]),
]


def run(cmd: list[str], cwd: Path = ROOT) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the HIP parse")
    ap.add_argument("--mutate", action="store_true",
                    help="also break the code on purpose and require the suite to notice "
                         "(tests/mutate.py, about a minute)")
    a = ap.parse_args()

    rows: list[tuple[str, str, str, bool]] = []

    for f in sorted((ROOT / "tests").glob("test_*.py")):
        if f.name == "run_all.py":
            continue
        t0 = time.time()
        rc, out = run([PY, str(f)])
        tail = [l for l in out.splitlines() if "passed" in l or "FAIL" in l]
        detail = tail[-1].strip() if tail else out.strip().splitlines()[-1][:60] if out.strip() else ""
        rows.append((f.name, detail, f"{time.time() - t0:.1f}s", rc == 0))

    for label, flags in GRAPH_VARIANTS:
        rc, out = run([PY, str(ROOT / "src" / "host" / "taskgraph.py"),
                       *flags, "--report"])
        verdict = next((l.strip() for l in out.splitlines()
                        if l.startswith("validation")), "no verdict")
        errs = [l.strip() for l in out.splitlines() if l.strip().startswith("!")]
        rows.append((f"graph: {label}", verdict + (f"  {errs[0][:60]}" if errs else ""),
                     "", rc == 0 and verdict == "validation: PASS"))

    if not a.quick:
        for label, env in (("HIP parse", {}), ("HIP parse (nt weights)",
                                               {"HIP_SYNTAX_DEFS": "-DFLEET_NT_WEIGHTS=1"})):
            import os
            e = {**os.environ, **env}
            p = subprocess.run(["bash", str(ROOT / "scripts" / "hip_syntax_check.sh")],
                               cwd=ROOT, capture_output=True, text=True, env=e)
            out = (p.stdout or "") + (p.stderr or "")
            if "SKIPPED" in out:
                rows.append((label, "skipped: no clang with the AMDGPU backend", "", True))
                continue
            ok = sum(1 for l in out.splitlines() if l.rstrip().endswith(" OK"))
            errs = sum(1 for l in out.splitlines() if "error:" in l)
            rows.append((label, f"{ok}/6 translation units clean, {errs} errors",
                         "", p.returncode == 0 and errs == 0))

    if a.mutate:
        rc, out = run([PY, str(ROOT / "tests" / "mutate.py")])
        tail = [l for l in out.splitlines() if "behaved as expected" in l]
        rows.append(("mutation gate", tail[-1].strip() if tail else "no verdict", "", rc == 0))

    width = max(len(r[0]) for r in rows)
    print()
    for name, detail, secs, ok in rows:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}"
              + (f"   ({secs})" if secs else ""))
    failed = [r[0] for r in rows if not r[3]]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} gates passed"
          + (f" — failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
