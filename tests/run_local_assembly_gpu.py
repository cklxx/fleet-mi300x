#!/usr/bin/env python3
"""F/K hardware gates and conservative stop rules; --dry-run needs no GPU.

Requires the model/cache/golden files prepared by hotaisle_bootstrap.sh.
Run F, then K, and FK only if the separate levers survive. Proof binaries
compare every consumer's ordinary local read with the original payload.
Their timings and stamped runs are excluded from the performance verdict.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import statistics
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "host"))
from taskgraph import PACK_FIELDS, PACK_FORMAT, TaskKind  # noqa: E402


def checked_decode(data, output):
    """Exact mandated sizes: equal but truncated gates must fail."""
    for key, want in (("tokens", 32), ("context", 1024), ("tokens_per_launch", 32),
                      ("matched", 32), ("compared", 32),
                      ("layers_within_gate", 27), ("layers", 27)):
        if data.get(key) != want:
            raise RuntimeError(f"{key}={data.get(key)!r}, required {want}")
    if len(data.get("produced", [])) != 32 or len(data.get("ms", [])) != 32:
        raise RuntimeError("incomplete token/timing array")
    if not math.isfinite(data.get("median_ms", float("nan"))) or data["median_ms"] <= 0:
        raise RuntimeError("invalid latency")
    if not re.search(r"tokens: 32/32 match HF greedy", output) or not re.search(
            r"consecutive layers within the gate: 27 of 27", output):
        raise RuntimeError("stdout token/layer gates missing")
    if "abort:" in output.lower():
        raise RuntimeError("kernel aborted")


def reader_tail(graph, trace, keys):
    """Sum phase tails from global producer completion through every reader.

    Retired REDUCE descriptors identify the original global event. Fold
    consumers use their stage stamp; LM_HEAD has only a norm-ready stamp.
    K uses attention body-ready, including the retained q_c payload wait.
    """
    blob, stamps = graph.read_bytes(), trace.read_bytes()
    tasks = [dict(zip(PACK_FIELDS, values)) for values in struct.iter_unpack(PACK_FORMAT, blob)]
    clocks = list(struct.iter_unpack("<4Q", stamps))
    if len(tasks) != len(clocks) or any(not s[0] for s in clocks):
        raise RuntimeError("incomplete trace")
    producers = {}
    for task, stamp in zip(tasks, clocks):
        producers.setdefault(task["signal_event"], []).append((task, stamp))
    tails = {}
    for task, stamp in zip(tasks, clocks):
        key = (task["kind"], task["layer"])
        if key not in keys:
            continue
        source = producers[task["wait_event"]]
        if source[0][0]["kind"] == TaskKind.REDUCE:
            source = producers[source[0][0]["wait_event"]]
        anchor = max(s[2] for _, s in source)
        shift = 32 if task["kind"] in (TaskKind.LM_HEAD, TaskKind.ATTENTION) else 8
        elapsed = (stamp[3] >> shift) & 0xffffff
        if not elapsed:
            raise RuntimeError(f"missing reader stamp for {key}")
        tails[key] = max(tails.get(key, 0), stamp[1] + elapsed - anchor)
    if tails.keys() != keys:
        raise RuntimeError("missing reader phases")
    return sum(tails.values()) * 0.01  # 100 MHz clock, microseconds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lever", choices=("F", "K", "FK"), required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--build", action="store_true", help="build native gfx942 binaries first")
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.pairs < 3:
        ap.error("at least three external pairs are required")
    out = a.out or ROOT / "build" / f"local_assembly_gpu_{a.lever.lower()}"
    if not a.dry_run:
        out.mkdir(parents=True, exist_ok=True)
    graph = lambda suffix: ROOT / "build" / f"taskgraph_d16{suffix}.bin"
    binary = lambda suffix: ROOT / "build" / f"fleet_decode_{suffix}"
    suffix = a.lever.lower()
    candidate = (binary(suffix), graph("_" + suffix))
    control = (binary(suffix), graph("_f" if a.lever == "FK" else ""))
    baseline = (binary("nt"), graph(""))
    proof = (binary(suffix + "_verify"), candidate[1])
    arms = {"B": baseline, a.lever + "0": control, a.lever + "1": candidate}

    def run(cmd, name, timeout=900):
        print("$ " + shlex.join(map(str, cmd)), flush=True)
        if a.dry_run:
            return ""
        proc = subprocess.run(list(map(str, cmd)), cwd=ROOT, capture_output=True,
                              text=True, timeout=timeout)
        output = (proc.stdout or "") + (proc.stderr or "")
        (out / f"{name}.log").write_text(output)
        if proc.returncode:
            raise RuntimeError(f"{name}: exit {proc.returncode}; see {out / (name + '.log')}")
        return output

    needed = {"": [], "_" + suffix: ["--local-" + part for part in
              (("fold", "kva") if a.lever == "FK" else ("fold",) if a.lever == "F" else ("kva",))]}
    if a.lever == "FK":
        needed["_f"] = ["--local-fold"]
    for tag, flags in needed.items():
        run([sys.executable, "src/host/taskgraph.py", "--kv-chunks", 16,
             *flags, "--emit", graph(tag), "--report"], "graph" + (tag or "_base"))
    if a.build:
        defs = ["-DFLEET_LOCAL_FOLD=1"] if "F" in a.lever else []
        defs += ["-DFLEET_LOCAL_KVA=1"] if "K" in a.lever else []
        for name, flags in (("nt", []), (suffix, defs),
                            (suffix + "_verify", defs + ["-DFLEET_VERIFY_LOCAL_ASSEMBLY=1"])):
            run(["hipcc", "--offload-arch=gfx942", "-O3", "-std=c++17",
                 "-Rpass-analysis=kernel-resource-usage",
                 "-DFLEET_NT_WEIGHTS=1", *flags, "-Isrc", "src/host/fleet_launch.hip",
                 "src/kernels/fleet_kernel.hip", "-o", binary(name)], "compile_" + name)

    def decode(arm, name, *extra):
        path = out / (name + ".json")
        if not a.dry_run:
            path.unlink(missing_ok=True)
        output = run([arm[0], "--graph", arm[1], "--tokens", 32, "--context", 1024,
                      "--tokens-per-launch", 32, "--repeat", 1, "--json", path, *extra], name)
        if a.dry_run:
            return {}
        if arm != baseline:
            fold = int("F" in a.lever) if arm != control else int(a.lever == "FK")
            kva = int("K" in a.lever) if arm != control else 0
            banner = f"local assembly: fold={fold} kva={kva} payload-verification={int(arm == proof)}"
            if banner not in output:
                raise RuntimeError(f"{name}: wrong feature/proof binary; expected {banner}")
        data = json.loads(path.read_text())
        checked_decode(data, output)
        return data

    print("Order: baseline/controls and current traces, loaded consumer proof, trace screen, full gates, pairs.")
    for label, arm in arms.items():
        decode(arm, "warm_" + label)
    for label, arm in arms.items():
        decode(arm, "trace_" + label, "--trace", out / (label + ".trace"))
    decode(proof, "payload_free")
    decode(proof, "payload_teacher", "--teacher-force")
    if not a.dry_run:
        flag = 4096 if a.lever == "F" else 8192
        keys = { (t["kind"], t["layer"]) for t in
                 (dict(zip(PACK_FIELDS, values)) for values in
                  struct.iter_unpack(PACK_FORMAT, candidate[1].read_bytes()))
                 if t["flags"] & flag and t["kind"] != TaskKind.REDUCE }
        tails = {label: reader_tail(arm[1], out / (label + ".trace"), keys)
                 for label, arm in arms.items()}
        print("Loaded reader-ready phase-tail sums (us): " + json.dumps(tails))
        if tails[a.lever + "1"] >= tails[a.lever + "0"]:
            raise RuntimeError("STOP: assembly/publication/wait/staging does not shorten reader-ready path")
    else:
        print("CHECK: candidate reader-ready tail sum < same-binary control; otherwise STOP.")
    for label, arm in arms.items():
        decode(arm, "free_" + label)
        decode(arm, "teacher_" + label, "--teacher-force")
        repeats = [decode(arm, f"det_{label}_{i}", "--dump-layers", out / f"det_{label}_{i}.bin")
                   for i in range(3)]
        if not a.dry_run:
            dumps = [(out / f"det_{label}_{i}.bin").read_bytes() for i in range(3)]
            if len(dumps[0]) != 27 * 2048 * 4 or any(dump != dumps[0] for dump in dumps[1:]) or \
                    any(run["produced"] != repeats[0]["produced"] for run in repeats[1:]):
                raise RuntimeError(f"STOP: {label} nondeterministic output or incomplete layer dump")
    verdicts = {}
    for ref_label in ("B", a.lever + "0"):
        pairs = []
        for i in range(a.pairs):
            order = (ref_label, a.lever + "1") if i % 2 == 0 else (a.lever + "1", ref_label)
            result = {label: decode(arms[label], f"pair_{ref_label}_{i}_{label}") for label in order}
            if not a.dry_run:
                pairs.append((result[ref_label]["median_ms"], result[a.lever + "1"]["median_ms"]))
        if not a.dry_run:
            refs, candidates = zip(*pairs)
            threshold = max(0.01, (max(refs) - min(refs)) / statistics.median(refs))
            gain = 1 - statistics.median(candidates) / statistics.median(refs)
            win = gain > threshold and all(c < b for b, c in pairs)
            verdicts[ref_label] = dict(pairs=pairs, gain=gain, threshold=threshold, win=win)
        else:
            print(f"CHECK: {a.lever}1 beats {ref_label} in every pair and median gain exceeds "
                  "max(1%, reference range / reference median); otherwise STOP.")
    if a.dry_run:
        print("DRY RUN: no GPU calls, files, payload/gate/performance verdicts produced.")
        return 0
    (out / "verdict.json").write_text(json.dumps(verdicts, indent=2) + "\n")
    print(json.dumps(verdicts, indent=2))
    if not all(row["win"] for row in verdicts.values()):
        raise RuntimeError("STOP: no repeatable above-noise win; leave flags OFF")
    print("PASS: payload, 32/32 tokens, 27/27 layers, determinism, reader-ready screen and pairs")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
