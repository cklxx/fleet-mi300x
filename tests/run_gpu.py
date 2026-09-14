#!/usr/bin/env python3
"""Every gate that needs the MI300X, in one command.

run_all.py covers what a laptop can check: descriptor layout, the event
protocol on the CPU, the graph validator, the reference arithmetic, the
parse. Nothing there can fail on memory visibility, on placement, or on a
race, because none of that exists until 304 workgroups are resident.

This is the other half. It expects a machine that scripts/hotaisle_bootstrap.sh
has already prepared (binaries in build/, packed weights, golden tokens and
hidden states), and it runs:

  * the microbenchmark's payload-visibility proofs, which are the only
    evidence the handshake publishes what it claims to publish;
  * the placement probe: 8 XCDs x 38 workgroups, or the whole design is void;
  * the protocol alone, with every task body skipped;
  * a full 32-token greedy decode on *every* graph variant the builder can
    emit, each required to match HuggingFace token for token and to keep all
    27 layer boundaries inside the gate;
  * teacher forcing, which isolates a step from the one before it;
  * bitwise determinism: the same decode twice, layer dumps compared byte
    for byte, which is what the no-float-atomics rule buys;
  * the abort path: a deliberately tiny spin limit must end the launch with
    a reason code instead of hanging the GPU.

    python3 tests/run_gpu.py                 # everything
    python3 tests/run_gpu.py --quick         # skip the variant sweep
"""
from __future__ import annotations

import argparse
import filecmp
import shutil
import struct
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
NT = BUILD / "fleet_decode_nt"
PLAIN = BUILD / "fleet_decode"
DEFAULT_GRAPH = BUILD / "taskgraph_d16.bin"

# label -> (graph file, extra launcher flags)
VARIANTS = {
    "default (16 KV chunks)": ("taskgraph_d16.bin", []),
    "8 KV chunks": ("taskgraph_d8.bin", []),
    "kv_a replicated": ("taskgraph_d16_kvarep.bin", []),
    "kv_a replicated + coherent": ("taskgraph_d16_kvarep.bin", ["--coherent-acts"]),
    "top-k published": ("taskgraph_d16_topkpub.bin", []),
    "idle-worker prefetch": ("taskgraph_d16_prefetch.bin", []),
    "K-chunk tiling": ("taskgraph_d16_k512.bin", []),
    "split workers + tiling": ("taskgraph_d16_split18_k512.bin", []),
}


def run(cmd, timeout=900):
    p = subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True,
                       text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def decode_ok(out: str) -> tuple[bool, str]:
    """A decode is correct only if every token matched and every layer
    boundary stayed inside the gate."""
    tok = re.search(r"tokens: (\d+)/(\d+) match HF greedy", out)
    lay = re.search(r"consecutive layers within the gate: (\d+) of (\d+)", out)
    ms = re.search(r"per-token latency[^:]*: median ([\d.]+) ms", out)
    ok = bool(tok) and tok.group(1) == tok.group(2) and bool(lay) and lay.group(1) == lay.group(2)
    detail = (f"{tok.group(0)[8:] if tok else 'no tokens'}, "
              f"{lay.group(1) if lay else '?'}/{lay.group(2) if lay else '?'} layers"
              + (f", {ms.group(1)} ms" if ms else ""))
    return ok, detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the variant sweep")
    ap.add_argument("--tokens", type=int, default=32)
    a = ap.parse_args()
    rows: list[tuple[str, str, bool]] = []

    def add(name, detail, ok, status=None):
        st = status or ("PASS" if ok else "FAIL")
        rows.append((name, detail, st))
        print(f"  {st:<4}  {name:<34} {detail}", flush=True)

    # 1. the memory model, measured rather than asserted
    if (BUILD / "microbench").exists():
        rc, out = run([BUILD / "microbench"], timeout=1800)
        verdicts = re.findall(r"\((d'*)\)[^\n]*\n\s+stale words: ([^\[]+)\[([^\]]+)\]", out)
        seen = {v[0]: (v[1].strip(), v[2]) for v in verdicts}
        for tag, must in (("d", True), ("d'", True), ("d'''", True), ("d''", False)):
            if tag not in seen:
                add(f"microbench ({tag})", "not reported", not must,
                    None if must else "INFO")
                continue
            words, verdict = seen[tag]
            # the uncached variant is expected to fail on this VM: MTYPE-UC
            # memory is not fence-free here, which is why --coherent-acts
            # uses agent-scope atomics. Report it, do not score it green.
            add(f"microbench ({tag})", f"{words} [{verdict}]",
                verdict.startswith("PASS") if must else True,
                None if must else "INFO")
    else:
        add("microbench", "build/microbench missing", False)

    if not NT.exists() or not DEFAULT_GRAPH.exists():
        add("decode binaries", "build/fleet_decode_nt or taskgraph_d16.bin missing", False)
        print(f"\n{sum(1 for r in rows if r[2] == 'PASS')}/{len(rows)} gates passed")
        return 1

    # 2. placement: the design assumes one workgroup per CU on 8 XCDs
    rc, out = run([NT, "--graph", DEFAULT_GRAPH, "--smoke", "--tokens", 4])
    hist = re.search(r"placement: (\d+) workgroups by XCC_ID: ([^\n]*)", out)
    counts = re.findall(r"\[\d\]=(\d+)", hist.group(2)) if hist else []
    add("placement probe", hist.group(0)[11:60] if hist else "no probe line",
        len(counts) == 8 and all(c == counts[0] for c in counts))
    add("protocol only (--smoke)", 
        (re.search(r"median ([\d.]+) ms", out).group(0) if "median" in out else "no timing")
        + (", abort" if "abort" in out.lower() else ""),
        rc == 0 and "abort" not in out.lower())

    # 3. the cache the launcher loads, against HuggingFace's own
    if (ROOT / "src" / "host" / "kv_convert.py").exists() and (BUILD / "fleet_cache.bin").exists():
        rc, out = run([sys.executable, ROOT / "src" / "host" / "kv_convert.py", "--verify"],
                      timeout=1800)
        add("KV cache vs HF", out.strip().splitlines()[-1][:60] if out.strip() else "", rc == 0)

    # 4. correctness on every graph the builder can emit
    variants = list(VARIANTS.items())[:1] if a.quick else list(VARIANTS.items())
    for label, (graph, flags) in variants:
        g = BUILD / graph
        if not g.exists():
            # never green on a missing input: the bootstrap emits every
            # variant before this runs, so absence means something failed
            add(f"decode: {label}", f"{graph} not emitted", False, "SKIP")
            continue
        rc, out = run([NT, "--graph", g, "--tokens", a.tokens, *flags])
        ok, detail = decode_ok(out)
        add(f"decode: {label}", detail, ok and rc == 0)

    # 5. teacher forcing isolates each step from the previous one
    rc, out = run([NT, "--graph", DEFAULT_GRAPH, "--tokens", a.tokens, "--teacher-force"])
    ok, detail = decode_ok(out)
    add("decode: teacher-forced", detail, ok and rc == 0)

    # 6. the plain build, so the non-temporal loads are not load-bearing
    if PLAIN.exists():
        rc, out = run([PLAIN, "--graph", DEFAULT_GRAPH, "--tokens", a.tokens])
        ok, detail = decode_ok(out)
        add("decode: plain loads", detail, ok and rc == 0)

    # 7. bitwise determinism, which is what banning float atomics buys
    d1, d2 = BUILD / "det_a.bin", BUILD / "det_b.bin"
    for out_path in (d1, d2):
        run([NT, "--graph", DEFAULT_GRAPH, "--tokens", 4, "--teacher-force",
             "--dump-layers", out_path])
    same = d1.exists() and d2.exists() and filecmp.cmp(d1, d2, shallow=False)
    add("bitwise determinism", "two runs, layer dumps identical" if same
        else "layer dumps differ or missing", same)

    # 8. a wait that can never be satisfied must end the launch with a
    #    reason code instead of hanging the GPU. Lowering the spin limit does
    #    not create a stuck wait, it only lowers the threshold for declaring
    #    one, and the loop evaluates that threshold every 1024 polls. So the
    #    graph itself is broken here: one descriptor is given a wait count
    #    that no number of producers can ever reach.
    sys.path.insert(0, str(ROOT / "src" / "host"))
    from taskgraph import PACK_FIELDS, PACK_FORMAT   # noqa: E402

    blob = bytearray(DEFAULT_GRAPH.read_bytes())
    broken_event = None
    for i in range(len(blob) // 64):
        f = dict(zip(PACK_FIELDS, struct.unpack_from(PACK_FORMAT, blob, 64 * i)))
        if f["wait_event"] >= 0 and f["kind"] != 10:      # 10 = EMBED
            f["wait_count"] = 9999
            struct.pack_into(PACK_FORMAT, blob, 64 * i, *[f[k] for k in PACK_FIELDS])
            broken_event = f["wait_event"]
            break
    if broken_event is None:
        add("abort on an unsatisfiable wait", "no waiting descriptor found", False)
    else:
        bg = BUILD / "taskgraph_broken.bin"
        bg.write_bytes(bytes(blob))
        side = Path(str(DEFAULT_GRAPH) + ".events.txt")
        if side.exists():
            shutil.copy(side, str(bg) + ".events.txt")
        rc, out = run([NT, "--graph", bg, "--tokens", 1, "--spin-limit", 1 << 18],
                      timeout=600)
        line = next((l.strip() for l in out.splitlines()
                     if l.lower().startswith("abort")), "no abort line")
        add("abort on an unsatisfiable wait", f"event {broken_event}: {line[:50]}",
            rc != 0 and "abort" in out.lower())

    good = sum(1 for r in rows if r[2] == "PASS")
    bad = [n for n, _, st in rows if st == "FAIL"]
    skipped = [n for n, _, st in rows if st == "SKIP"]
    info = sum(1 for r in rows if r[2] == "INFO")
    # an informational row is neither a pass nor a failure: counting it in the
    # denominator made a clean run read as 18/19 and exit non-zero
    print(f"\n{good}/{len(rows) - info} GPU gates passed"
          + (f", {info} informational" if info else "")
          + (f" — failed: {', '.join(bad)}" if bad else "")
          + (f" — skipped for missing inputs: {', '.join(skipped)}" if skipped else ""))
    return 0 if not bad and not skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
