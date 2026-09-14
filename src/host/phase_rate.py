#!/usr/bin/env python3
"""Per-phase byte-rate table: where the milliseconds actually are.

docs/design.md §10 asks for achieved bandwidth *by phase*, and notes the
hardware counters cannot split inside one persistent kernel. They do not have
to: the static accounting already knows the bytes each task kind must move, and
the per-task trace already knows how long each task kind took. This script
divides one by the other and ranks the phases by the time still on the table.

Three numbers come out of it, and they answer different questions:

    worker GB/s    bytes moved / aggregate worker-time for that kind. Compare
                   with the same code in isolation (bench/microbench.hip (f));
                   a phase far below the ceiling loses time inside the task
                   body, not in the schedule.

    phase TB/s     this kind's HBM bytes for one layer / the time the layer
                   spent in that phase (needs --timeline).

    recover ms     aggregate worker-time that would disappear if this kind
                   reached the reference rate, divided by the 296 workers that
                   share it — an upper bound on the token time it can give
                   back, because a phase only pays out if it is on the
                   critical path.

Bytes are counted on the side they are served from, because the two sides are
not the same thing and only one of them is what rocprof's FETCH_SIZE measures:

    HBM  a cold read: weights, the embedding row. FETCH_SIZE sees these.
    L2   a read of something another task just wrote: the folded partials, the
         shared MLA cache rows, the attention partials the merge consumes.
         These cost the worker time but never leave the chip, so they must not
         be charged against the FETCH_SIZE reconciliation.

The byte model is written out term by term below, each traced to the kernel's
row ranges and flag bits, and reconciled against rocprofv3 FETCH_SIZE on the
HBM side only, so a wrong term shows up as a visible gap instead of hiding
inside a total.

    # from a raw trace, with the graph it was produced with (flags drive the model)
    python3 src/host/phase_rate.py --graph build/taskgraph_d16.bin \
        --trace results/trace_v16.bin --timeline results/timeline_v16_L5.txt

    # from what is committed in results/ (no GPU, no trace binary)
    python3 src/host/phase_rate.py \
        --summary results/trace_v16_default_summary.txt \
        --timeline results/timeline_v16_default_L5.txt \
        --measured-bytes results/prof_fetch_v1.csv --tokens 4 \
        --microbench results/microbench.json --kva-shared

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
FLAGS = getattr(tg, "Flags", None)
TICK_US = 0.01            # s_memrealtime is a 100 MHz counter
BF16 = 2
WORKERS = 296             # 8 x 37, the workers that share the token's time
XCDS = 8
WORKERS_PER_XCD = 37
# The prologues that fold the previous phase's partials (kernel call sites of
# stage_residual): the next layer's q/kv_a, the router (always, for o_proj's
# K-split partials), and lm_head.
FOLD_SITES = "QKV_FUSED,NORM_ROUTER,LM_HEAD"


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


def flags_of(tasks: list[dict]) -> dict:
    """What the emitted graph says about the model's terms. The descriptor
    flags are the kernel's own switchboard, so the byte model is derived from
    the graph instead of assumed: kva_shared changes a weight read by 8x, and
    FOLD_PARTIALS decides whether the prologue still folds."""
    if FLAGS is None:
        return {"kva_shared": False, "fold_sites": set()}
    kv = sum(1 for t in tasks
             if t["kind"] == tg.TaskKind.QKV_FUSED.value
             and t["flags"] & int(FLAGS.KVA_SHARED))
    fold = {KIND[t["kind"]] for t in tasks if t["flags"] & int(FLAGS.FOLD_PARTIALS)}
    return {"kva_shared": kv > 0, "fold_sites": fold or {"NORM_ROUTER"}}


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

    Three columns of microseconds: average busy, then the prologue, and — once
    the launcher split it out — the staging inside the prologue. Counted
    rather than matched, so an extra column is an extra number rather than a
    parse failure."""
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
    """One layer's per-kind first-ready / last-done: the critical path itself."""
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
    the kernel — hence --tokens. It is an L2-fill count: traffic that another
    task already pulled into the L2 does not appear here."""
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
# Per layer, in MB, split by the side the bytes come from. Each term is the
# kernel's own row range or flag (src/kernels, src/host/taskgraph.py), not the
# design's idealised table: where the kernel reads a weight block once per
# chiplet, this charges it once per chiplet and says so.

def byte_model(cfg: dict, units: int, fold_tasks: int, kva_shared: bool,
               seq_len: int, fold_sites: set[str]) -> dict[str, dict]:
    """kind -> {hbm, l2} MB per layer, plus the number of layers it runs on."""
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
    # folds them. Which tasks those are is a flag in the graph, not a guess.
    fold = (h * 4 + XCDS * h * 4) * fold_tasks * m
    # the attention partials a merge task consumes: chunks x (m, l, acc[512])
    merge_l2 = heads * fold_tasks / (heads * XCDS) * (2 + lora) * 4 * m
    model = {
        "QKV_FUSED": {"hbm": (XCDS * q_rows * h
                              + (kv_rows * h if kva_shared else XCDS * kv_rows * h))
                      * BF16 * m,
                      "l2": fold if "QKV_FUSED" in fold_sites else 0.0,
                      "runs": layers},
        # the shared MLA cache row is written once per layer and read by all 16
        # heads, so this is L2 / Infinity-Cache traffic, never HBM
        "ATTENTION": {"hbm": 0.0,
                      "l2": heads * seq_len * (lora + qk_rope) * BF16 * m,
                      "runs": layers},
        "MERGE_UV": {"hbm": heads * v_head * lora * BF16 * m, "l2": merge_l2,
                     "runs": layers},
        "O_PROJ": {"hbm": h * heads * v_head * BF16 * m,
                   "l2": XCDS * h * 4 * WORKERS_PER_XCD * m, "runs": layers},
        "NORM_ROUTER": {"hbm": XCDS * n_routed * h * BF16 * m,
                        "l2": fold if "NORM_ROUTER" in fold_sites else 0.0,
                        "runs": layers},
        "EXPERT_GATE_UP": {"hbm": units * 2 * moe_i * h * BF16 * m,
                           "l2": XCDS * moe_i * 4 * m, "runs": moe_layers},
        "EXPERT_DOWN": {"hbm": units * h * moe_i * BF16 * m,
                        "l2": XCDS * moe_i * 4 * m, "runs": moe_layers},
        "DENSE_GATE_UP": {"hbm": 2 * dense_i * h * BF16 * m, "l2": 0.0,
                          "runs": first_k},
        "DENSE_DOWN": {"hbm": h * dense_i * BF16 * m, "l2": 0.0, "runs": first_k},
        "LM_HEAD": {"hbm": vocab * h * BF16 * m,
                    "l2": fold if "LM_HEAD" in fold_sites else 0.0, "runs": 1},
        "EMBED": {"hbm": h * BF16 * m, "l2": 0.0, "runs": 1},
        "REDUCE": {"hbm": 0.0, "l2": 0.0, "runs": 0},
        "ARGMAX": {"hbm": 0.0, "l2": 0.0, "runs": 0},
    }
    for v in model.values():
        v["mb_layer"] = v["hbm"] + v["l2"]
    return model


# --------------------------------------------------------------- the table

def build_rows(kinds: dict[str, dict], model: dict[str, dict],
               ref_gbps: float) -> list[dict]:
    rows = []
    for kind, a in kinds.items():
        e = model.get(kind, {"hbm": 0.0, "l2": 0.0, "runs": 0, "mb_layer": 0.0})
        runs = e["runs"]
        hbm, l2 = e["hbm"] * runs, e["l2"] * runs
        busy_ms = a["busy_ms"]
        moved = (hbm + l2) / 1e3                      # GB moved per token
        gbps = moved / (busy_ms / 1e3) if busy_ms else 0.0
        recover = busy_ms * max(0.0, 1.0 - gbps / ref_gbps) if ref_gbps else 0.0
        rows.append({"kind": kind, "tasks": a["tasks"], "busy_ms": busy_ms,
                     "avg_busy_us": a.get("avg_busy_us", 0.0),
                     "prologue_us": a.get("prologue_us", 0.0),
                     "mb_layer": e["hbm"], "mb_l2_layer": e["l2"],
                     "mb_layer_all": e["hbm"] + e["l2"],
                     "hbm": hbm, "l2": l2, "gbps": gbps,
                     "recover_worker_ms": recover, "recover_ms": recover / WORKERS})
    rows.sort(key=lambda r: -r["recover_ms"])
    return rows


def print_rows(rows: list[dict], ref_gbps: float, ref_label: str,
               token_ms: float | None, timeline: dict | None,
               model: dict[str, dict], measured: dict | None) -> None:
    hbm = sum(v["hbm"] * v["runs"] for v in model.values())
    l2 = sum(v["l2"] * v["runs"] for v in model.values())
    print(f"\nreference worker rate {ref_gbps:6.2f} GB/s  ({ref_label})")
    if token_ms:
        print(f"token span {token_ms:.3f} ms")
    print(f"\n  {'kind':<16}{'tasks':>6}{'busy-sum':>10}{'avg-busy':>9}"
          f"{'prologue':>9}{'HBM MB/l':>10}{'L2 MB/t':>9}{'GB/s':>7}"
          f"{'phase TB/s':>11}{'recover ms':>11}")
    tl = (timeline or {}).get("kinds", {})
    for r in rows:
        t = tl.get(r["kind"])
        phase = ""
        if t and t["last_done_us"] > t["first_ready_us"] and r["mb_layer"]:
            span_us = t["last_done_us"] - t["first_ready_us"]
            phase = f"{r['mb_layer']/1e3 / (span_us/1e6):7.2f}"
        rec = (f"{r['recover_ms']:7.3f}" if r["mb_layer_all"] >= 1.0
               else "      —")
        print(f"  {r['kind']:<16}{r['tasks']:>6}{r['busy_ms']:>8.1f} ms"
              f"{r['avg_busy_us']:>7.1f} us{r['prologue_us']:>7.1f} us"
              f"{r['mb_layer']:>10.1f}{r['l2']:>9.1f}{r['gbps']:>7.2f}"
              f"{phase:>11}{rec:>11}")

    top = [r for r in rows if r["mb_layer_all"] >= 1.0]
    print(f"\nrecoverable at the reference rate: "
          f"{sum(r['recover_ms'] for r in top):.3f} ms/token over "
          f"{len(top)} kinds")
    print("  (aggregate worker-time / 296 workers: an upper bound, since a")
    print("   phase only pays out if it is on the critical path)")

    print(f"\nbyte model per token: {hbm/1e3:.2f} GB from HBM "
          f"+ {l2/1e3:.2f} GB served from the L2")
    print("  (the L2 side costs the worker time but no HBM bandwidth, and")
    print("   FETCH_SIZE — an L2-fill count — does not see it)")
    if measured:
        have = measured["gb_per_token"]
        print(f"  measured FETCH_SIZE {have:.2f} GB/token "
              f"({measured['kernel_gb']:.2f} GB over {measured['tokens']} tokens)"
              f"   model {hbm/1e3 - have:+.2f} GB "
              f"({100 * (hbm/1e3 / have - 1):+.1f}%)")

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
    ap.add_argument("--fold-sites", default=FOLD_SITES,
                    help=f"task kinds whose prologue folds ({FOLD_SITES}); with "
                         "--graph the graph's FOLD_PARTIALS bit wins")
    ap.add_argument("--kva-shared", action="store_true",
                    help="the analysed run had the kv_a rows split over the "
                         "8 XCDs; with --graph the graph's flag wins")
    ap.add_argument("--ref-gbps", type=float, default=0.0,
                    help="reference worker rate; default: this run's best phase")
    ap.add_argument("--ceiling", action="store_true",
                    help="use the microbench isolation ceiling as the reference")
    a = ap.parse_args()

    cfg = json.loads(a.config.read_text())
    fold_sites = {s for s in a.fold_sites.split(",") if s}
    kva_shared = a.kva_shared
    timeline = parse_timeline(a.timeline) if a.timeline else None
    token_ms = None
    if a.summary:
        s = parse_summary(a.summary)
        kinds, token_ms = s["kinds"], s["token_ms"]
    elif a.graph and a.trace:
        tasks = load_graph(a.graph)
        f = flags_of(tasks)
        if f["fold_sites"]:
            fold_sites = f["fold_sites"]
        kva_shared = kva_shared or f["kva_shared"]
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

    print(f"model flags: kva_shared={kva_shared} fold={sorted(fold_sites)}")
    model = byte_model(cfg, a.units, a.fold_tasks, kva_shared, a.seq_len, fold_sites)
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
    print_rows(rows, ref, label, token_ms, timeline, model, measured)


if __name__ == "__main__":
    main()
