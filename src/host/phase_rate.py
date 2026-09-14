#!/usr/bin/env python3
"""Per-phase byte-rate table: where the milliseconds actually are.

docs/design.md §10 asks for achieved bandwidth *by phase*, and notes the
hardware counters cannot split inside one persistent kernel. They do not have
to: the static accounting already knows the bytes each task kind must move, and
the per-task trace already knows how long each task kind took. This script
divides one by the other and ranks the phases by the time still on the table.

Three numbers come out of it, and they answer different questions:

    worker GB/s    bytes / aggregate worker-time for that kind. Compare with
                   the same code in isolation (bench/microbench.hip (f)); a
                   phase far below the ceiling loses time inside the task
                   body, not in the schedule.

    phase TB/s     this kind's bytes for one layer / the time the layer spent
                   in that phase (needs --timeline). This is the rate the
                   phase actually delivers while every other phase waits.

    recover ms     aggregate worker-time that would disappear if this kind
                   reached the reference rate, divided by the 296 workers that
                   share it — i.e. an upper bound on the token time it can
                   give back. Upper bound, because a phase only pays out if it
                   is on the critical path.

The byte model is written out term by term below, each traced to the kernel's
row ranges, and reconciled against rocprofv3 FETCH_SIZE, so a wrong term shows
up as a visible gap instead of hiding inside a total.

    # from a raw trace, with the graph it was produced with
    python3 src/host/phase_rate.py --graph build/taskgraph_d8.bin \\
        --trace results/trace_v14.bin --timeline results/timeline_v14_L5.txt

    # from what is committed in results/ (no GPU, no trace binary)
    python3 src/host/phase_rate.py \\
        --summary results/trace_v14_nt_d16_coh_summary.txt \\
        --timeline results/timeline_v14_nt_d16_coh_L5.txt \\
        --measured-bytes results/rocprofv3_fetch_size.csv --tokens 4 \\
        --microbench results/microbench.json

Trace format (src/host/fleet_launch.hip, report_trace): 4 uint64 per
descriptor in graph order — wait-start, ready, done, and a packed word with
xcd in bits 0-7, staging ticks in 8-31 and prologue ticks in 32-55, at
100 MHz. A zero wait-start means the task never ran.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "host"))

import taskgraph as tg  # noqa: E402  (the descriptor layout lives in one place)

KIND = {k.value: k.name for k in tg.TaskKind}
TICK_US = 0.01            # s_memrealtime is a 100 MHz counter
BF16 = 2
WORKERS = 296             # 8 x 37, the workers that share the token's time
XCDS = 8
WORKERS_PER_XCD = 37


def _num(pattern: str, text: str) -> float:
    """First capture group as a float, or a loud failure."""
    m = re.search(pattern, text)
    if not m:
        raise SystemExit(f"unexpected line in input: {text!r}")
    return float(m.group(1))


# --------------------------------------------------------------- inputs

def load_graph(path: Path) -> list[dict]:
    """The descriptors a trace indexes into, field for field."""
    raw = path.read_bytes()
    if len(raw) % tg.DESCRIPTOR_BYTES:
        raise SystemExit(f"{path}: {len(raw)} bytes is not a multiple of "
                         f"{tg.DESCRIPTOR_BYTES}")
    out = []
    for i in range(len(raw) // tg.DESCRIPTOR_BYTES):
        vals = struct.unpack_from(tg.PACK_FORMAT, raw, i * tg.DESCRIPTOR_BYTES)
        out.append(dict(zip(tg.PACK_FIELDS, vals)))
    return out


def load_trace(path: Path, n_tasks: int) -> list[tuple[int, int, int, int]]:
    raw = path.read_bytes()
    if len(raw) != n_tasks * 32:
        raise SystemExit(f"{path}: {len(raw)} bytes, expected {n_tasks * 32} "
                         f"for {n_tasks} descriptors — graph and trace are from "
                         f"different runs")
    return [struct.unpack_from("<4Q", raw, i * 32) for i in range(n_tasks)]


SUMMARY_ROW = re.compile(
    r"^\s+([A-Z_]+)\s+(\d+)\s+([\d.]+) ms\s+([\d.]+) ms\s+(\S.*)$")


def parse_summary(path: Path) -> dict:
    """The launcher's own per-kind table: results/trace_*_summary.txt.

    Three columns of microseconds: avg busy, then the prologue, and — once the
    launcher split it out — the staging inside the prologue. Counted rather
    than matched, so an extra column is an extra number, not a parse failure."""
    kinds: dict[str, dict] = {}
    token_ms = None
    for line in path.read_text().splitlines():
        m = SUMMARY_ROW.match(line)
        if m:
            us = [float(x) for x in re.findall(r"([\d.]+) us", m.group(5))]
            kinds[m.group(1)] = {"tasks": int(m.group(2)),
                                 "busy_ms": float(m.group(3)),
                                 "wait_ms": float(m.group(4)),
                                 "avg_busy_us": us[0] if us else 0.0,
                                 "prologue_us": us[1] if len(us) > 1 else 0.0,
                                 "stage_us": us[2] if len(us) > 2 else 0.0}
        elif "from first wait to last signal" in line:
            token_ms = _num(r"([\d.]+) ms", line)
    if not kinds:
        raise SystemExit(f"{path}: no per-kind rows — is this a trace summary?")
    return {"kinds": kinds, "token_ms": token_ms}


TIMELINE_ROW = re.compile(r"^\s+([A-Z_]+)\s+(\S.*?)\s+(\d+)\s*$")


def parse_timeline(path: Path) -> dict:
    """One layer's per-kind first-ready / last-done: the critical path itself.

    Five numbers before the task count, or six once the staging sub-phase is
    split out of the prologue (the launcher grew that column), so the numbers
    are counted rather than matched position by position."""
    kinds: dict[str, dict] = {}
    span_us, layer = None, None
    for line in path.read_text().splitlines():
        m = TIMELINE_ROW.match(line)
        if m:
            nums = [float(x) for x in re.findall(r"([\d.]+)us", m.group(2))]
            if len(nums) < 5:
                continue
            kinds[m.group(1)] = {"first_ready_us": nums[0],
                                 "last_ready_us": nums[1],
                                 "last_done_us": nums[2],
                                 "avg_busy_us": nums[3],
                                 "prologue_us": nums[4],
                                 "stage_us": nums[5] if len(nums) > 5 else 0.0,
                                 "tasks": int(m.group(3))}
        elif line.startswith("layer "):
            layer = int(_num(r"layer (\d+)", line))
            span_us = _num(r"span ([\d.]+) us", line)
    if not kinds:
        raise SystemExit(f"{path}: no per-kind rows — is this a layer timeline?")
    return {"kinds": kinds, "span_us": span_us, "layer": layer}


def parse_fetch_size(path: Path, tokens: int) -> dict:
    """rocprofv3 --pmc FETCH_SIZE. Counter_Value is KB, summed over the whole
    dispatch, and one dispatch is every token because the token loop is inside
    the kernel — hence --tokens."""
    kernel_kb, total_kb, name = 0.0, 0.0, None
    with path.open() as f:
        cols = [c.strip('"') for c in f.readline().rstrip("\n").split(",")]
        # Counter_Name/Counter_Value (--pmc per-counter) or a FETCH_SIZE column
        ni = cols.index("Kernel_Name") if "Kernel_Name" in cols else cols.index("KernelName")
        vi = (cols.index("Counter_Value") if "Counter_Value" in cols
              else cols.index("FETCH_SIZE"))
        for line in f:
            parts = [p.strip('"') for p in line.rstrip("\n").split(",")]
            if len(parts) != len(cols):
                continue
            kb = float(parts[vi])
            total_kb += kb
            # rocprof appends .kd and [clone .kd] to the symbol
            this = parts[ni].replace(".kd", "").split(" [")[0].strip()
            if this == "fleet_decode_step":
                kernel_kb += kb
                name = this
    if name is None:
        raise SystemExit(f"{path}: no fleet_decode_step dispatch in this profile")
    return {"kernel_gb": kernel_kb / 1e6, "total_gb": total_kb / 1e6,
            "gb_per_token": kernel_kb / 1e6 / tokens, "tokens": tokens}


def parse_microbench(path: Path) -> dict:
    """The isolation ceiling, from the numbers the design already cites."""
    b = json.loads(path.read_text())
    best = float(b.get("gemv_best_tbs", 0.0))
    return {"best_tbs": best, "gbps_per_worker": best * 1e12 / WORKERS / 1e9}


# --------------------------------------------------------------- byte model
#
# Per layer, in MB. Each term is the kernel's own row range (src/kernels),
# not the design's idealised table: where the kernel reads a weight block once
# per chiplet, this charges it once per chiplet and says so.

def byte_model(cfg: dict, units: int, fold_tasks: int, kva_shared: bool,
               seq_len: int) -> dict[str, tuple[float, int]]:
    """kind -> (MB per layer, layers it runs on). Returns MB per token when
    multiplied out by the caller."""
    h = cfg["hidden_size"]
    heads, qk_nope, qk_rope = (cfg["num_attention_heads"], cfg["qk_nope_head_dim"],
                               cfg["qk_rope_head_dim"])
    v_head, lora = cfg["v_head_dim"], cfg["kv_lora_rank"]
    moe_i, dense_i = cfg["moe_intermediate_size"], cfg["intermediate_size"]
    n_routed, vocab = cfg["n_routed_experts"], cfg["vocab_size"]
    layers, first_k = cfg["num_hidden_layers"], cfg["first_k_dense_replace"]
    moe_layers = layers - first_k
    m = 1e-6
    q_rows = heads // XCDS * (qk_nope + qk_rope)          # 2 heads per chiplet
    kv_rows = lora + qk_rope                              # 576
    # stage_residual: x (fp32) + the 8 XCD partials (fp32), once per task that
    # folds them. Called by the q/kv_a prologue and by the router prologue.
    fold = (h * 4 + XCDS * h * 4) * fold_tasks * m
    per_layer = {
        "QKV_FUSED": (XCDS * q_rows * h
                      + (kv_rows * h if kva_shared else XCDS * kv_rows * h))
        * BF16 * m,
        # the shared MLA cache: 16 heads read the same 1.2 MB, so this is L2 /
        # Infinity-Cache traffic rather than HBM
        "ATTENTION": heads * seq_len * (lora + qk_rope) * BF16 * m,
        "MERGE_UV": heads * v_head * lora * BF16 * m,
        "O_PROJ": h * heads * v_head * BF16 * m,
        "NORM_ROUTER": XCDS * n_routed * h * BF16 * m + fold,
        "EXPERT_GATE_UP": units * 2 * moe_i * h * BF16 * m,
        "EXPERT_DOWN": units * h * moe_i * BF16 * m,
        "DENSE_GATE_UP": 2 * dense_i * h * BF16 * m,
        "DENSE_DOWN": h * dense_i * BF16 * m,
        "LM_HEAD": vocab * h * BF16 * m + fold,
        "EMBED": h * BF16 * m,
        "REDUCE": 0.0, "ARGMAX": 0.0,
    }
    runs = {"QKV_FUSED": layers, "ATTENTION": layers, "MERGE_UV": layers,
            "O_PROJ": layers, "NORM_ROUTER": layers,
            "EXPERT_GATE_UP": moe_layers, "EXPERT_DOWN": moe_layers,
            "DENSE_GATE_UP": first_k, "DENSE_DOWN": first_k,
            "LM_HEAD": 1, "EMBED": 1, "REDUCE": 0, "ARGMAX": 0}
    return {k: (v, runs[k]) for k, v in per_layer.items()}


# --------------------------------------------------------------- the table

def build_rows(kinds: dict[str, dict], model: dict[str, tuple[float, int]],
               ref_gbps: float) -> list[dict]:
    rows = []
    for kind, a in kinds.items():
        per_layer_mb, runs = model.get(kind, (0.0, 0))
        mb_token = per_layer_mb * runs
        busy_ms = a["busy_ms"]
        gbps = (mb_token / 1e3) / (busy_ms / 1e3) if busy_ms else 0.0
        recover = busy_ms * max(0.0, 1.0 - gbps / ref_gbps) if ref_gbps else 0.0
        rows.append({"kind": kind, "tasks": a["tasks"], "busy_ms": busy_ms,
                     "avg_busy_us": a.get("avg_busy_us", 0.0),
                     "prologue_us": a.get("prologue_us", 0.0),
                     "mb_layer": per_layer_mb, "mb": mb_token, "gbps": gbps,
                     "recover_worker_ms": recover,
                     "recover_ms": recover / WORKERS})
    rows.sort(key=lambda r: -r["recover_ms"])
    return rows


def print_rows(rows: list[dict], ref_gbps: float, ref_label: str,
               token_ms: float | None, timeline: dict | None,
               model_total_gb: float, measured: dict | None,
               attention_gb: float) -> None:
    print(f"\nreference worker rate {ref_gbps:6.2f} GB/s  ({ref_label})")
    if token_ms:
        print(f"token span {token_ms:.3f} ms")
    print(f"\n  {'kind':<16}{'tasks':>6}{'busy-sum':>10}{'avg-busy':>9}"
          f"{'prologue':>9}{'MB/layer':>10}{'MB/token':>10}{'GB/s':>7}"
          f"{'phase TB/s':>11}{'recover ms':>11}")
    tl = (timeline or {}).get("kinds", {})
    for r in rows:
        t = tl.get(r["kind"])
        phase = ""
        if t and t["last_done_us"] > t["first_ready_us"] and r["mb_layer"]:
            span_us = t["last_done_us"] - t["first_ready_us"]
            phase = f"{r['mb_layer']/1e3 / (span_us/1e6):7.2f}"
        rec = f"{r['recover_ms']:7.3f}" if r["mb_layer"] >= 1.0 else "      —"
        print(f"  {r['kind']:<16}{r['tasks']:>6}{r['busy_ms']:>8.1f} ms"
              f"{r['avg_busy_us']:>7.1f} us{r['prologue_us']:>7.1f} us"
              f"{r['mb_layer']:>10.1f}{r['mb']:>10.1f}{r['gbps']:>7.2f}"
              f"{phase:>11}{rec:>11}")

    top = [r for r in rows if r["mb_layer"] >= 1.0]
    print(f"\nrecoverable at the reference rate: "
          f"{sum(r['recover_ms'] for r in top):.3f} ms/token over "
          f"{len(top)} kinds")
    print("  (aggregate worker-time / 296 workers: an upper bound, since a")
    print("   phase only pays out if it is on the critical path)")

    print(f"\nbyte model per token: {model_total_gb:.2f} GB"
          f"   (attention cache reads {attention_gb:.2f} GB of that are L2"
          f"\n                      traffic, not HBM)")
    if measured:
        have = measured["gb_per_token"]
        print(f"  measured FETCH_SIZE {have:.2f} GB/token "
              f"({measured['kernel_gb']:.2f} GB over {measured['tokens']} tokens)"
              f"   model {model_total_gb - have:+.2f} GB "
              f"({100 * (model_total_gb / have - 1):+.1f}%)")

    if timeline:
        print(f"\nlayer {timeline['layer']} critical path "
              f"{timeline['span_us']:.1f} us")
        prev_end, prev_start = 0.0, 0.0
        for kind, v in sorted(tl.items(), key=lambda kv: kv[1]["first_ready_us"]):
            print(f"  {kind:<16}{v['first_ready_us']:>7.1f} ->{v['last_done_us']:>7.1f} us"
                  f"   busy {v['avg_busy_us']:>6.1f} us"
                  f"   start {v['first_ready_us'] - prev_start:>+6.1f} us"
                  f"   overlap {prev_end - v['first_ready_us']:>+6.1f} us"
                  f"   n {v['tasks']}")
            prev_end = max(prev_end, v["last_done_us"])
            prev_start = v["first_ready_us"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=Path, help="launcher per-kind table (txt)")
    ap.add_argument("--graph", type=Path, help="emitted descriptors (bin)")
    ap.add_argument("--trace", type=Path, help="per-descriptor trace (bin)")
    ap.add_argument("--timeline", type=Path, help="one layer's phase timeline (txt)")
    ap.add_argument("--measured-bytes", type=Path, help="rocprofv3 FETCH_SIZE csv")
    ap.add_argument("--microbench", type=Path, help="results/microbench.json")
    ap.add_argument("--tokens", type=int, default=1,
                    help="tokens in the profiled dispatch (fetch-size csv)")
    ap.add_argument("--config", type=Path,
                    default=ROOT / "reference" / "dsv2lite_config.json")
    ap.add_argument("--seq-len", type=int, default=1056)
    ap.add_argument("--units", type=int, default=8,
                    help="expert units per layer: routed slots + shared halves")
    ap.add_argument("--fold-tasks", type=int, default=WORKERS,
                    help="workers per layer whose prologue folds the residual; "
                         "0 drops the term")
    ap.add_argument("--kva-shared", action="store_true",
                    help="the analysed run had --kva-shared: kv_a read once")
    ap.add_argument("--ref-gbps", type=float, default=0.0,
                    help="reference worker rate; default: this run's best phase")
    ap.add_argument("--ceiling", action="store_true",
                    help="use the microbench isolation ceiling as the reference")
    a = ap.parse_args()

    cfg = json.loads(a.config.read_text())
    model = byte_model(cfg, a.units, a.fold_tasks, a.kva_shared, a.seq_len)

    timeline = parse_timeline(a.timeline) if a.timeline else None
    token_ms = None
    if a.summary:
        s = parse_summary(a.summary)
        kinds, token_ms = s["kinds"], s["token_ms"]
    elif a.graph and a.trace:
        tasks = load_graph(a.graph)
        tr = load_trace(a.trace, len(tasks))
        agg: dict[str, dict] = {}
        for t, (w, ready, done, packed) in zip(tasks, tr):
            if w == 0:
                continue
            e = agg.setdefault(KIND[t["kind"]],
                               {"tasks": 0, "busy_ticks": 0, "pro_ticks": 0,
                                "busy_ms": 0.0, "avg_busy_us": 0.0,
                                "prologue_us": 0.0})
            e["tasks"] += 1
            e["busy_ticks"] += done - ready
            e["pro_ticks"] += (packed >> 32) & 0xFFFFFF
        for e in agg.values():
            e["busy_ms"] = e["busy_ticks"] * TICK_US / 1e3
            e["avg_busy_us"] = e["busy_ticks"] * TICK_US / e["tasks"]
            e["prologue_us"] = e["pro_ticks"] * TICK_US / e["tasks"]
        kinds = agg
    else:
        raise SystemExit("need --summary, or --graph with --trace")

    measured = (parse_fetch_size(a.measured_bytes, a.tokens)
                if a.measured_bytes else None)
    micro = parse_microbench(a.microbench) if a.microbench else None

    if a.ref_gbps:
        ref, label = a.ref_gbps, "--ref-gbps"
    elif a.ceiling and micro:
        ref, label = micro["gbps_per_worker"], (
            f"microbench (c) {micro['best_tbs']:.2f} TB/s / {WORKERS} workers")
    else:
        probe = build_rows(kinds, model, 0.0)
        best = max(probe, key=lambda r: r["gbps"])
        ref, label = best["gbps"], f"this run's best phase, {best['kind']}"
    rows = build_rows(kinds, model, ref)
    print_rows(rows, ref, label, token_ms, timeline,
               sum(v * n for v, n in model.values()) / 1e3, measured,
               model["ATTENTION"][0] * model["ATTENTION"][1] / 1e3)


if __name__ == "__main__":
    main()
