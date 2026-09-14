#!/usr/bin/env python3
"""The phase-rate table: inputs, arithmetic, and the byte model's terms.

`src/host/phase_rate.py` turns a trace into a ranked list of where the
milliseconds are, and its byte model is the one number in that list that is
not measured. So the things worth testing are:

  * the parsers accept exactly what the launcher and rocprofv3 emit
    (descriptor layout, trace packing, summary/timeline text, FETCH_SIZE csv),
  * a graph and a trace from different runs are rejected instead of
    silently producing a wrong table,
  * the rate and recover arithmetic is what the printed columns claim,
  * the byte model's two redundancy terms behave (8 chiplet reads of kv_a,
    and the fold), and the total stays reconciled with FETCH_SIZE.

    python3 tests/test_phase_rate.py
"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "host"))

import phase_rate as pr  # noqa: E402
import taskgraph as tg  # noqa: E402

def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {label:<48} [{'PASS' if ok else 'FAIL'}]  {detail}")
    return ok


def pack_descriptor(**over) -> bytes:
    """A descriptor, field for field, via the same layout the graph uses."""
    vals = {name: 0 for name in tg.PACK_FIELDS}
    vals.update(wait_event=-1, signal_event=-1, layer=-1, head=-1,
                kv_chunk=-1, expert_slot=-1, n_split=1)
    vals.update(over)
    return struct.pack(tg.PACK_FORMAT, *[vals[n] for n in tg.PACK_FIELDS])


def trace_word(wait: int, ready: int, done: int, xcd: int,
               stg_ticks: int = 0, pro_ticks: int = 0) -> bytes:
    return struct.pack("<4Q", wait, ready, done,
                       xcd | (stg_ticks << 8) | (pro_ticks << 32))


def keys(tmp: Path, write_graph: bool = True) -> tuple[Path, Path]:
    """3 QKV tasks, then 1 attention task that never ran, then 1 lm_head."""
    g = tmp / "graph.bin"
    t = tmp / "trace.bin"
    if write_graph:
        rows = [
            (dict(kind=tg.TaskKind.QKV_FUSED.value, layer=1, xcd=0, worker=0),
             (100, 200, 200 + 2000, 0, 0, 900)),      # 20.0 us busy, 9.0 us prologue
            (dict(kind=tg.TaskKind.QKV_FUSED.value, layer=1, xcd=0, worker=1),
             (100, 300, 300 + 1000, 0, 0, 500)),      # 10.0 us busy, 5.0 us prologue
            (dict(kind=tg.TaskKind.QKV_FUSED.value, layer=1, xcd=1, worker=0),
             (100, 400, 400 + 3000, 1, 0, 1200)),     # 30.0 us busy, 12.0 us prologue
            (dict(kind=tg.TaskKind.ATTENTION.value, layer=1, xcd=1, worker=1),
             (0, 0, 0, 1, 0, 0)),                     # never ran
            (dict(kind=tg.TaskKind.LM_HEAD.value, layer=-1, xcd=2, worker=0),
             (100, 500, 500 + 5000, 2, 0, 100)),      # 50.0 us busy
        ]
        g.write_bytes(b"".join(pack_descriptor(**d) for d, _ in rows))
        t.write_bytes(b"".join(trace_word(*w) for _, w in rows))
    return g, t


def main() -> int:
    results = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        g, t = keys(tmp)

        # ---- descriptor + trace round trip, and the aggregation
        tasks = pr.load_graph(g)
        results.append(check("descriptor round trip", len(tasks) == 5
                             and tasks[0]["kind"] == tg.TaskKind.QKV_FUSED.value
                             and tasks[0]["worker"] == 0,
                             f"{len(tasks)} descriptors, {len(tg.PACK_FIELDS)} fields"))
        tr = pr.load_trace(t, len(tasks))
        results.append(check("trace round trip", len(tr) == 5 and tr[2][3] & 7 == 1,
                             f"packed xcd {tr[2][3] & 7}"))
        rows = {}
        for desc, (w, ready, done, packed) in zip(tasks, tr):
            if w == 0:
                continue
            e = rows.setdefault(pr.KIND[desc["kind"]],
                                {"tasks": 0, "busy_ticks": 0, "pro_ticks": 0})
            e["tasks"] += 1
            e["busy_ticks"] += done - ready
            e["pro_ticks"] += (packed >> 32) & 0xFFFFFF
        busy_ms = rows["QKV_FUSED"]["busy_ticks"] * pr.TICK_US / 1e3
        results.append(check("busy-sum is ticks x 10 ns",
                             abs(busy_ms - 0.060) < 1e-9, f"{busy_ms:.4f} ms"))
        results.append(check("never-run task is skipped",
                             "ATTENTION" not in rows))
        pro_us = rows["QKV_FUSED"]["pro_ticks"] * pr.TICK_US / rows["QKV_FUSED"]["tasks"]
        results.append(check("prologue is the packed field",
                             abs(pro_us - 8.667) < 0.01, f"{pro_us:.3f} us avg"))

        # ---- a graph and a trace from different runs must not silently agree
        t2 = tmp / "short.bin"
        t2.write_bytes(t.read_bytes()[:64])
        try:
            pr.load_trace(t2, len(tasks))
            ok, detail = False, "accepted a mismatched trace"
        except SystemExit as e:
            ok, detail = "different runs" in str(e), str(e)[:44]
        results.append(check("mismatched trace rejected", ok, detail))

        # ---- the two text parsers, on the launcher's own formats
        summ = tmp / "s.txt"
        summ.write_text(
            "trace of token 1: 3.946 ms from first wait to last signal\n"
            "  kind             tasks   busy-sum   wait-sum   avg-busy avg-prologue\n"
            "  QKV_FUSED         7992   180.50 ms    38.15 ms     22.6 us        8.6 us\n"
            "  LM_HEAD            296    39.84 ms     1.05 ms    134.6 us        9.3 us\n")
        s = pr.parse_summary(summ)
        results.append(check("summary parsed",
                             s["kinds"]["QKV_FUSED"]["tasks"] == 7992
                             and abs(s["kinds"]["LM_HEAD"]["busy_ms"] - 39.84) < 1e-9
                             and abs(s["token_ms"] - 3.946) < 1e-9,
                             f"{len(s['kinds'])} kinds, span {s['token_ms']} ms"))
        tline = tmp / "l5.txt"
        tline.write_text(
            "layer 5: 1992 descriptors, span 139.3 us\n"
            "  kind            first-ready last-ready  last-done  avg-busy  prologue  n\n"
            "  QKV_FUSED             0.0us      0.7us     23.5us    22.6us     8.6us  296\n"
            "  EXPERT_DOWN         110.6us    113.5us    139.3us    23.3us     6.3us  296\n")
        tl = pr.parse_timeline(tline)
        results.append(check("timeline parsed",
                             tl["layer"] == 5 and abs(tl["span_us"] - 139.3) < 1e-9
                             and tl["kinds"]["EXPERT_DOWN"]["tasks"] == 296,
                             f"layer {tl['layer']}, span {tl['span_us']} us"))

        # ---- FETCH_SIZE: KB, summed over the dispatch, /--tokens
        csv = tmp / "f.csv"
        csv.write_text('"Kernel_Name","Grid_Size","Counter_Value"\n'
                       '"fleet_probe_xcc(unsigned int*)",77824,12.875\n'
                       '"fleet_decode_step",77824,22660092.0625\n'
                       '"__amd_rocclr_fillBufferAligned",512,3.625\n')
        f = pr.parse_fetch_size(csv, tokens=4)
        results.append(check("fetch-size: KB, fleet dispatch, per token",
                             abs(f["gb_per_token"] - 5.665) < 0.005
                             and abs(f["total_gb"] - 22.660108) < 0.001,
                             f"{f['gb_per_token']:.3f} GB/token of {f['total_gb']:.2f} GB"))

        # ---- the byte model's terms
        cfg = json.loads((ROOT / "reference" / "dsv2lite_config.json").read_text())
        layers, first_k = cfg["num_hidden_layers"], cfg["first_k_dense_replace"]
        moe_layers = layers - first_k
        base = pr.byte_model(cfg, units=8, fold_tasks=pr.WORKERS,
                             kva_shared=False, seq_len=1056)
        shared = pr.byte_model(cfg, units=8, fold_tasks=pr.WORKERS,
                               kva_shared=True, seq_len=1056)
        kv_rows = cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"]
        expect_delta = (pr.XCDS - 1) * kv_rows * cfg["hidden_size"] \
            * pr.BF16 / 1e6
        got = base["QKV_FUSED"][0] - shared["QKV_FUSED"][0]
        results.append(check("kv_a replication term is 7 extra chiplet reads",
                             abs(got - expect_delta) < 1e-6,
                             f"{got:.2f} MB/layer = {expect_delta:.2f}"))

        h, moe_i = cfg["hidden_size"], cfg["moe_intermediate_size"]
        expect_gu = 8 * 2 * moe_i * h * pr.BF16 / 1e6
        results.append(check("expert gate_up = 8 units x 2 x moe_inter x hidden",
                             abs(base["EXPERT_GATE_UP"][0] - expect_gu) < 1e-6,
                             f"{base['EXPERT_GATE_UP'][0]:.1f} MB/layer"))
        no_fold = pr.byte_model(cfg, 8, 0, False, 1056)
        fold_delta = base["NORM_ROUTER"][0] - no_fold["NORM_ROUTER"][0]
        results.append(check("fold term scales with the workers that fold",
                             abs(fold_delta - (h * 4 + pr.XCDS * h * 4)
                                 * pr.WORKERS / 1e6) < 1e-6,
                             f"{fold_delta:.1f} MB/layer over {pr.WORKERS} workers"))

        total_gb = sum(v * n for v, n in base.values()) / 1e3
        results.append(check("model stays reconciled with FETCH_SIZE 5.67 GB",
                             5.0 <= total_gb <= 7.0,
                             f"{total_gb:.2f} GB/token "
                             f"({100 * (total_gb / 5.665 - 1):+.1f}% vs measured)"))

        # ---- rate and recover arithmetic, on a hand-computed row
        rows = pr.build_rows({"LM_HEAD": {"tasks": 296, "busy_ms": 40.0,
                                          "avg_busy_us": 135.1, "prologue_us": 9.3}},
                             base, ref_gbps=11.0)
        r = rows[0]
        want_gbps = (base["LM_HEAD"][0] * 1 * 1e-3) / 0.040
        results.append(check("worker GB/s = MB/token / busy-sum",
                             abs(r["gbps"] - want_gbps) < 1e-9,
                             f"{r['gbps']:.2f} vs {want_gbps:.2f}"))
        results.append(check("recover ms = busy x (1-rate/ref) / workers",
                             abs(r["recover_ms"] - 0.0) < 1e-12,
                             f"{r['recover_ms']:.4f} when the phase beats the reference"))
        slow, same = ("LM_HEAD", {"tasks": 296, "busy_ms": 40.0,
                                  "avg_busy_us": 135.1, "prologue_us": 9.3}), None
        at_slow = pr.build_rows({slow[0]: slow[1]}, base, ref_gbps=16.0)[0]
        at_fast = pr.build_rows({slow[0]: slow[1]}, base, ref_gbps=4.0)[0]
        results.append(check("recover = busy x (1-rate/ref) / workers",
                             abs(at_slow["recover_ms"]
                                 - 40.0 * (1 - at_slow["gbps"] / 16.0) / pr.WORKERS) < 1e-9,
                             f"{at_slow['recover_ms']:.4f} ms/token at a 16 GB/s reference"))
        results.append(check("recover clamps to zero above the reference",
                             at_fast["recover_ms"] == 0.0
                             and at_slow["recover_ms"] > 0.0,
                             f"0.0 at 4 GB/s, {at_slow['recover_ms']:.4f} at 16"))

    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
