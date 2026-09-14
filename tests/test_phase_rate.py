#!/usr/bin/env python3
"""phase_rate: the parsers, the byte model's terms, and the arithmetic.

The tool's byte model is the one number in its output that is not measured, so
what is worth testing is: the parsers accept exactly what the launcher and
rocprof emit, a graph and a trace from different runs are rejected instead of
silently producing a wrong table, the model's terms behave, and the HBM side
stays reconciled with FETCH_SIZE while the L2 side stays out of it.

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


def make_inputs(tmp: Path) -> tuple[Path, Path]:
    """3 q/kv_a tasks (one carrying flags), 1 attention that never ran, 1 lm_head."""
    g, t = tmp / "graph.bin", tmp / "trace.bin"
    rows = [
        (dict(kind=tg.TaskKind.QKV_FUSED.value, layer=1, xcd=0, worker=0,
              flags=int(tg.Flags.KVA_SHARED) | int(tg.Flags.FOLD_PARTIALS)),
         (100, 200, 2200, 0, 0, 900)),               # 20.0 us busy, 9.0 us prologue
        (dict(kind=tg.TaskKind.QKV_FUSED.value, layer=1, xcd=0, worker=1),
         (100, 300, 1300, 0, 0, 500)),               # 10.0 us
        (dict(kind=tg.TaskKind.QKV_FUSED.value, layer=1, xcd=1, worker=0),
         (100, 400, 3400, 1, 0, 1200)),              # 30.0 us
        (dict(kind=tg.TaskKind.ATTENTION.value, layer=1, xcd=1, worker=1),
         (0, 0, 0, 1, 0, 0)),                        # never ran
        (dict(kind=tg.TaskKind.LM_HEAD.value, layer=-1, xcd=2, worker=0),
         (100, 500, 5500, 2, 0, 100)),               # 50.0 us
    ]
    g.write_bytes(b"".join(pack_descriptor(**d) for d, _ in rows))
    t.write_bytes(b"".join(trace_word(*w) for _, w in rows))
    return g, t


def main() -> int:
    results = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        g, t = make_inputs(tmp)

        # ---- descriptor and trace round trip
        tasks = pr.load_graph(g)
        results.append(check("descriptor round trip", len(tasks) == 5
                             and tasks[0]["kind"] == tg.TaskKind.QKV_FUSED.value,
                             f"{len(tasks)} descriptors, {len(tg.PACK_FIELDS)} fields"))
        tr = pr.load_trace(t, len(tasks))
        results.append(check("trace round trip", len(tr) == 5 and tr[2][3] & 7 == 1,
                             f"packed xcd {tr[2][3] & 7}"))
        f = pr.flags_of(tasks)
        results.append(check("the graph's flags drive the model",
                             f["kva_shared"] and f["fold_sites"] == {"QKV_FUSED"},
                             f"kva_shared={f['kva_shared']} "
                             f"fold={sorted(f['fold_sites'])}"))

        agg: dict[str, dict] = {}
        for desc, (w, ready, done, packed) in zip(tasks, tr):
            if w == 0:
                continue
            e = agg.setdefault(pr.KIND[desc["kind"]], {"n": 0, "busy": 0, "pro": 0})
            e["n"] += 1
            e["busy"] += done - ready
            e["pro"] += (packed >> 32) & 0xFFFFFF
        busy_ms = agg["QKV_FUSED"]["busy"] * pr.TICK_US / 1e3
        results.append(check("busy-sum is ticks x 10 ns",
                             abs(busy_ms - 0.060) < 1e-9, f"{busy_ms:.4f} ms"))
        results.append(check("never-run task is skipped", "ATTENTION" not in agg))
        pro_us = agg["QKV_FUSED"]["pro"] * pr.TICK_US / agg["QKV_FUSED"]["n"]
        results.append(check("prologue is the packed high field",
                             abs(pro_us - 8.667) < 0.01, f"{pro_us:.3f} us avg"))

        t2 = tmp / "short.bin"
        t2.write_bytes(t.read_bytes()[:64])
        try:
            pr.load_trace(t2, len(tasks))
            ok, detail = False, "accepted a mismatched trace"
        except SystemExit as e:
            ok, detail = "different runs" in str(e), str(e)[:40]
        results.append(check("mismatched trace rejected", ok, detail))

        # ---- the launcher's two text formats and rocprof's two csv schemas
        summ = tmp / "s.txt"
        summ.write_text(
            "trace of token 1: 3.669 ms from first wait to last signal\n"
            "  kind             tasks   busy-sum   wait-sum   avg-busy avg-prologue  avg-stage\n"
            "  QKV_FUSED         7992   135.71 ms    32.89 ms     17.0 us        7.5 us      4.9 us\n"
            "  LM_HEAD            296    39.84 ms     1.05 ms    134.6 us        9.3 us      0.0 us\n")
        s = pr.parse_summary(summ)
        results.append(check("summary parsed, extra column and all",
                             s["kinds"]["QKV_FUSED"]["tasks"] == 7992
                             and abs(s["kinds"]["QKV_FUSED"]["busy_ms"] - 135.71) < 1e-9
                             and abs(s["kinds"]["LM_HEAD"]["prologue_us"] - 9.3) < 1e-9
                             and abs(s["token_ms"] - 3.669) < 1e-9,
                             f"{len(s['kinds'])} kinds, span {s['token_ms']} ms"))
        tline = tmp / "l5.txt"
        tline.write_text(
            "layer 5: 1992 descriptors, span 129.3 us\n"
            "  kind            first-ready last-ready  last-done  avg-busy  prologue of it:stage  n\n"
            "  QKV_FUSED             0.0us      1.6us     19.3us    16.9us     7.6us       4.8us  296\n"
            "  EXPERT_DOWN         106.6us    108.1us    129.3us    18.7us     1.6us       0.0us  296\n")
        tl = pr.parse_timeline(tline)
        results.append(check("timeline parsed",
                             tl["layer"] == 5 and abs(tl["span_us"] - 129.3) < 1e-9
                             and tl["kinds"]["QKV_FUSED"]["tasks"] == 296
                             and abs(tl["kinds"]["QKV_FUSED"]["last_done_us"] - 19.3) < 1e-9,
                             f"layer {tl['layer']}, span {tl['span_us']} us"))
        csv = tmp / "f.csv"
        csv.write_text('"Kernel_Name","Counter_Value"\n'
                       '"fleet_probe_xcc(unsigned int*)",12.875\n'
                       '"fleet_decode_step",22660092.0625\n'
                       '"__amd_rocclr_fillBufferAligned",3.625\n')
        fa = pr.parse_fetch_size(csv, tokens=4)
        results.append(check("fetch-size: Counter_Value, per token",
                             abs(fa["gb_per_token"] - 5.665) < 0.005,
                             f"{fa['gb_per_token']:.3f} GB/token"))
        csv2 = tmp / "f2.csv"
        csv2.write_text('Index,KernelName,grd,FETCH_SIZE\n'
                        '0,"__amd_rocclr_fillBufferAligned.kd",77824,32.0\n'
                        '70,"fleet_decode_step.kd",77824,20809785.75\n')
        fb = pr.parse_fetch_size(csv2, tokens=4)
        results.append(check("fetch-size: FETCH_SIZE column, .kd suffix",
                             abs(fb["gb_per_token"] - 5.202) < 0.005,
                             f"{fb['gb_per_token']:.3f} GB/token"))

        # ---- the byte model's terms
        cfg = json.loads((ROOT / "reference" / "dsv2lite_config.json").read_text())
        sites = {"QKV_FUSED", "NORM_ROUTER", "LM_HEAD"}
        rep = pr.byte_model(cfg, kva_shared=False, fold_sites=sites)
        sh = pr.byte_model(cfg, kva_shared=True, fold_sites=sites)
        kv_rows = cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"]
        expect = (pr.XCDS - 1) * kv_rows * cfg["hidden_size"] * pr.BF16 / 1e6
        got = rep["QKV_FUSED"]["hbm"] - sh["QKV_FUSED"]["hbm"]
        results.append(check("kv_a replication is 7 extra chiplet reads",
                             abs(got - expect) < 1e-6, f"{got:.2f} MB/layer"))
        h, moe_i = cfg["hidden_size"], cfg["moe_intermediate_size"]
        units = cfg["num_experts_per_tok"] + 2
        results.append(check("expert gate_up = units x 2 x moe_inter x hidden",
                             abs(rep["EXPERT_GATE_UP"]["hbm"]
                                 - units * 2 * moe_i * h * pr.BF16 / 1e6) < 1e-6,
                             f"{rep['EXPERT_GATE_UP']['hbm']:.1f} MB/layer, "
                             f"{units} units"))
        heads, qk_nope, lora = (cfg["num_attention_heads"], cfg["qk_nope_head_dim"],
                                cfg["kv_lora_rank"])
        want_wuk = (heads * (pr.KV_CHUNKS - 1) * qk_nope * lora * pr.BF16
                    + heads * pr.SEQ_LEN * (lora + cfg["qk_rope_head_dim"]) * pr.BF16) / 1e6
        results.append(check("W_UK is charged to the head's chunk tasks",
                             abs(rep["ATTENTION"]["l2"] - want_wuk) < 1e-6
                             and abs(rep["ATTENTION"]["hbm"]
                                     - heads * qk_nope * lora * pr.BF16 / 1e6) < 1e-6,
                             f"{rep['ATTENTION']['l2']:.1f} MB/layer from L2, "
                             f"{rep['ATTENTION']['hbm']:.1f} from HBM"))
        want_fold = (h * 4 + pr.XCDS * h * 4) * pr.WORKERS / 1e6
        results.append(check("fold term = x + 8 partials, once per worker",
                             abs(rep["NORM_ROUTER"]["l2"] - want_fold) < 1e-6,
                             f"{rep['NORM_ROUTER']['l2']:.1f} MB/layer"))
        no_lm = pr.byte_model(cfg, True, {"QKV_FUSED", "NORM_ROUTER"})
        results.append(check("a fold site can be dropped per kind",
                             rep["LM_HEAD"]["l2"] > 0 and no_lm["LM_HEAD"]["l2"] == 0,
                             f"lm_head l2 {rep['LM_HEAD']['l2']:.1f} -> "
                             f"{no_lm['LM_HEAD']['l2']:.1f} MB/layer"))

        # Only the HBM side is what an L2-fill counter can see, and it is what
        # the design's 4.94 GB/token accounting describes.
        prof = ROOT / "results" / "prof_fetch_v1.csv"
        measured = (pr.parse_fetch_size(prof, tokens=4)["gb_per_token"]
                    if prof.exists() else 5.202)
        hbm = sum(v["hbm"] * v["runs"] for v in sh.values()) / 1e3
        l2 = sum(v["l2"] * v["runs"] for v in sh.values()) / 1e3
        results.append(check("HBM side reconciles with the measured counter",
                             abs(hbm / measured - 1) <= 0.10,
                             f"{hbm:.2f} GB HBM vs {measured:.2f} GB measured "
                             f"({100 * (hbm / measured - 1):+.1f}%)"))
        results.append(check("the L2 side stays out of that reconciliation",
                             l2 > 1.0,
                             f"{l2:.2f} GB/token served from L2, invisible to "
                             f"FETCH_SIZE"))
        # two profiles in results/ are two different binaries, so the footprint
        # of the dispatch has to travel with the number
        if prof.exists():
            real = pr.parse_fetch_size(prof, tokens=4)
            results.append(check("the profile carries the dispatch footprint",
                                 "scr" in real["sig"] and "lds" in real["sig"],
                                 f"{real['file']}: {real['sig']}"))
        results.append(check("a profile without those columns has no footprint",
                             pr.parse_fetch_size(csv, tokens=4)["sig"] == ""))

        # ---- the rate and recover arithmetic
        one = {"LM_HEAD": {"tasks": 296, "busy_ms": 40.0, "avg_busy_us": 135.1,
                           "prologue_us": 9.1}}
        r = pr.build_rows(one, rep, ref_gbps=11.0)[0]
        want = (rep["LM_HEAD"]["hbm"] + rep["LM_HEAD"]["l2"]) * 1e-3 / 0.040
        results.append(check("worker GB/s = MB/token / busy-sum",
                             abs(r["gbps"] - want) < 1e-9,
                             f"{r['gbps']:.2f} vs {want:.2f}"))
        slow = pr.build_rows(one, rep, ref_gbps=16.0)[0]
        fast = pr.build_rows(one, rep, ref_gbps=4.0)[0]
        results.append(check("recover = busy x (1-rate/ref) / workers",
                             abs(slow["recover_ms"]
                                 - 40.0 * (1 - slow["gbps"] / 16.0) / pr.WORKERS) < 1e-9,
                             f"{slow['recover_ms']:.4f} ms/token at a 16 GB/s reference"))
        results.append(check("recover clamps to zero above the reference",
                             fast["recover_ms"] == 0.0 and slow["recover_ms"] > 0.0,
                             f"0 at 4 GB/s, {slow['recover_ms']:.4f} at 16"))

    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
