#!/usr/bin/env python3
"""Fleet task-graph builder for batch-1 decode (docs/design.md §3, §4).

Builds the immutable, ahead-of-time task queues the persistent kernel runs:
descriptors are generated once on the host, validated as a DAG, assigned to
per-XCD worker queues, and emitted as a flat binary the kernel reads without
synchronisation.

Per MoE layer the graph is

    QKV ──global──▶ AT ──xcd-local──▶ MG ──global──▶ OP ──global──▶ N2
     │                                                               │
     └───────────────────────────── global ──────────────────────────┘
    N2 ──global──▶ GU ──xcd-local──▶ DN ──global──▶ RD ──global──▶ next layer

which is 6 global events and 2 XCD-local events per layer; four fusions (§3)
removed the other four global events an unfused graph would need.

    python3 src/host/taskgraph.py --report
    python3 src/host/taskgraph.py --emit build/taskgraph.bin
"""
from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

XCDS = 8              # MI300X chiplets
CUS_PER_XCD = 38      # 1 scheduler + 37 workers
WORKERS_PER_XCD = CUS_PER_XCD - 1
DESCRIPTOR_BYTES = 64


class TaskKind(IntEnum):
    """Mirrors the task levels in §3. Values are baked into the descriptor."""
    QKV_FUSED = 0      # q_proj ‖ kv_a, Chiplet-task x8, RMSNorm in prologue
    ATTENTION = 1      # MLA, CU-task, head-affine, kv-post + q-absorb in prologue
    MERGE_UV = 2       # merge split-KV partials + W_UV, CU-task per head
    O_PROJ = 3         # o_proj + residual, Chiplet-task x8
    NORM_ROUTER = 4    # RMSNorm + router top-k, CU-task x1
    EXPERT_GATE_UP = 5  # gate_up + SiLU⊙up, Chiplet-task x8
    EXPERT_DOWN = 6    # down x routing weight, Chiplet-task x8
    REDUCE = 7         # reduce partials + residual, CU-task x1
    DENSE_GATE_UP = 8  # layer 0 only
    DENSE_DOWN = 9     # layer 0 only
    EMBED = 10
    LM_HEAD = 11       # Chiplet-task x8, final norm in prologue
    ARGMAX = 12


# Implementation status per task kind, reported because the task spec asks for
# "Fleet-native operations, remaining fallbacks" as a deliverable metric.
#
#   fleet-verified   : runs in the persistent kernel, checked against the
#                      reference on device
#   fleet-unverified : implemented in the kernel, never executed on hardware
#   fallback         : still runs as a torch op between launches
#
# Nothing is "fleet-verified" until it has run on the MI300X. Marking code
# verified because it was written would defeat the purpose of the column.
IMPLEMENTATION: dict[str, str] = {
    "EMBED": "fleet-unverified",
    "QKV_FUSED": "fleet-unverified",
    "ATTENTION": "fleet-unverified",
    "MERGE_UV": "fleet-unverified",
    "O_PROJ": "fleet-unverified",
    "NORM_ROUTER": "fleet-unverified",
    "EXPERT_GATE_UP": "fleet-unverified",
    "EXPERT_DOWN": "fleet-unverified",
    "REDUCE": "fleet-unverified",
    "DENSE_GATE_UP": "fleet-unverified",
    "DENSE_DOWN": "fleet-unverified",
    "LM_HEAD": "fleet-unverified",
    "ARGMAX": "fleet-unverified",
}


class Scope(IntEnum):
    """Synchronisation scope of the event a task signals (§4)."""
    NONE = 0
    XCD_LOCAL = 1      # L2-resident counter, no fence
    GLOBAL = 2         # cross-XCD: counter + mirroring by the per-XCD scheduler


@dataclass
class Task:
    kind: TaskKind
    layer: int
    xcd: int
    wait_event: int | None     # event id this task blocks on
    signal_event: int | None   # event id it contributes to on completion
    signal_scope: Scope = Scope.NONE
    head: int = -1             # attention/merge only
    kv_chunk: int = -1         # split-KV only
    expert_slot: int = -1      # 0..top_k-1 routed, or shared half
    n_split: int = 1           # how many tasks share this signal_event
    worker: int = -1           # filled by assign_workers()
    index: int = -1

    def pack(self) -> bytes:
        """64-byte descriptor; layout is fixed so the kernel can index it."""
        return struct.pack(
            "<HHhh hhhh hhhh 40x",
            int(self.kind), self.layer, self.xcd, self.worker,
            -1 if self.wait_event is None else self.wait_event,
            -1 if self.signal_event is None else self.signal_event,
            int(self.signal_scope), self.n_split,
            self.head, self.kv_chunk, self.expert_slot, self.index,
        )


@dataclass
class Graph:
    tasks: list[Task] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)

    def new_event(self, scope: Scope, producers: int, label: str) -> int:
        eid = len(self.events)
        self.events.append({"id": eid, "scope": int(scope),
                            "producers": producers, "label": label})
        return eid

    def add(self, t: Task) -> Task:
        t.index = len(self.tasks)
        self.tasks.append(t)
        return t


def build_moe_layer(g: Graph, layer: int, prev_event: int | None,
                    cfg: dict) -> int:
    """Emit one MoE layer; returns the event the next layer must wait on."""
    heads, top_k = cfg["heads"], cfg["top_k"]
    kv_chunks = cfg["kv_chunks"]

    # 1. fused q_proj ‖ kv_a GEMV. RMSNorm is recomputed per worker in the
    #    prologue (4 KB vector) instead of costing a global event.
    e_qkv = g.new_event(Scope.GLOBAL, XCDS, f"L{layer}.qkv")
    for xcd in range(XCDS):
        g.add(Task(TaskKind.QKV_FUSED, layer, xcd, prev_event, e_qkv,
                   Scope.GLOBAL, n_split=XCDS))

    # 2. attention. XCD k owns heads 2k, 2k+1 — 3072 q columns / 8 = exactly
    #    two heads — so the merge below stays XCD-local.
    heads_per_xcd = heads // XCDS
    e_attn = [g.new_event(Scope.XCD_LOCAL, kv_chunks, f"L{layer}.attn.h{h}")
              for h in range(heads)]
    for h in range(heads):
        xcd = h // heads_per_xcd
        for chunk in range(kv_chunks):
            g.add(Task(TaskKind.ATTENTION, layer, xcd, e_qkv, e_attn[h],
                       Scope.XCD_LOCAL, head=h, kv_chunk=chunk,
                       n_split=kv_chunks))

    # 3. merge flash-decoding partials, then absorb W_UV. Same XCD as the head.
    e_merge = g.new_event(Scope.GLOBAL, heads, f"L{layer}.merge")
    for h in range(heads):
        g.add(Task(TaskKind.MERGE_UV, layer, h // heads_per_xcd, e_attn[h],
                   e_merge, Scope.GLOBAL, head=h, n_split=heads))

    # 4. o_proj + residual — needs all 16 heads, so this one is global.
    e_oproj = g.new_event(Scope.GLOBAL, XCDS, f"L{layer}.o_proj")
    for xcd in range(XCDS):
        g.add(Task(TaskKind.O_PROJ, layer, xcd, e_merge, e_oproj,
                   Scope.GLOBAL, n_split=XCDS))

    # 5. post-attention norm + router. Single task: needs the whole vector.
    #    Rotate the single-task XCD by layer: these signal global events, so the
    #    placement is free, and pinning them all to XCD 0 would queue 55 extra
    #    tasks there (a 25% imbalance across the run).
    e_router = g.new_event(Scope.GLOBAL, 1, f"L{layer}.router")
    g.add(Task(TaskKind.NORM_ROUTER, layer, layer % XCDS, e_oproj, e_router,
               Scope.GLOBAL))

    # 6/7. experts: 8 balanced units = top_k routed + shared split in two.
    #      Each XCD streams exactly one unit; gate_up → down is XCD-local.
    e_gate_up = [g.new_event(Scope.XCD_LOCAL, 1, f"L{layer}.gate_up.x{x}")
                 for x in range(XCDS)]
    e_down = g.new_event(Scope.GLOBAL, XCDS, f"L{layer}.down")
    for xcd in range(XCDS):
        slot = xcd if xcd < top_k else -(xcd - top_k + 1)  # <0 marks shared half
        g.add(Task(TaskKind.EXPERT_GATE_UP, layer, xcd, e_router,
                   e_gate_up[xcd], Scope.XCD_LOCAL, expert_slot=slot))
        g.add(Task(TaskKind.EXPERT_DOWN, layer, xcd, e_gate_up[xcd], e_down,
                   Scope.GLOBAL, expert_slot=slot, n_split=XCDS))

    # 8. reduce the 8 partials + residual.
    e_out = g.new_event(Scope.GLOBAL, 1, f"L{layer}.out")
    g.add(Task(TaskKind.REDUCE, layer, (layer + 4) % XCDS, e_down, e_out,
               Scope.GLOBAL))
    return e_out


def build_dense_layer(g: Graph, layer: int, prev_event: int | None,
                      cfg: dict) -> int:
    """Layer 0: same attention path, dense MLP instead of experts.

    The dense gate_up → down boundary needs the full h[10944], which does not
    fit an XCD-local buffer, so it costs one extra global event versus MoE.
    """
    heads, kv_chunks = cfg["heads"], cfg["kv_chunks"]
    heads_per_xcd = heads // XCDS

    e_qkv = g.new_event(Scope.GLOBAL, XCDS, f"L{layer}.qkv")
    for xcd in range(XCDS):
        g.add(Task(TaskKind.QKV_FUSED, layer, xcd, prev_event, e_qkv,
                   Scope.GLOBAL, n_split=XCDS))

    e_attn = [g.new_event(Scope.XCD_LOCAL, kv_chunks, f"L{layer}.attn.h{h}")
              for h in range(heads)]
    for h in range(heads):
        for chunk in range(kv_chunks):
            g.add(Task(TaskKind.ATTENTION, layer, h // heads_per_xcd, e_qkv,
                       e_attn[h], Scope.XCD_LOCAL, head=h, kv_chunk=chunk,
                       n_split=kv_chunks))

    e_merge = g.new_event(Scope.GLOBAL, heads, f"L{layer}.merge")
    for h in range(heads):
        g.add(Task(TaskKind.MERGE_UV, layer, h // heads_per_xcd, e_attn[h],
                   e_merge, Scope.GLOBAL, head=h, n_split=heads))

    e_oproj = g.new_event(Scope.GLOBAL, XCDS, f"L{layer}.o_proj")
    for xcd in range(XCDS):
        g.add(Task(TaskKind.O_PROJ, layer, xcd, e_merge, e_oproj,
                   Scope.GLOBAL, n_split=XCDS))

    e_norm = g.new_event(Scope.GLOBAL, 1, f"L{layer}.norm")
    g.add(Task(TaskKind.NORM_ROUTER, layer, 0, e_oproj, e_norm, Scope.GLOBAL))

    e_gate_up = g.new_event(Scope.GLOBAL, XCDS, f"L{layer}.dense_gate_up")
    for xcd in range(XCDS):
        g.add(Task(TaskKind.DENSE_GATE_UP, layer, xcd, e_norm, e_gate_up,
                   Scope.GLOBAL, n_split=XCDS))

    e_out = g.new_event(Scope.GLOBAL, XCDS, f"L{layer}.dense_down")
    for xcd in range(XCDS):
        g.add(Task(TaskKind.DENSE_DOWN, layer, xcd, e_gate_up, e_out,
                   Scope.GLOBAL, n_split=XCDS))
    return e_out


def build_graph(cfg: dict) -> Graph:
    g = Graph()
    e = g.new_event(Scope.GLOBAL, 1, "embed")
    g.add(Task(TaskKind.EMBED, -1, 0, None, e, Scope.GLOBAL))

    for layer in range(cfg["layers"]):
        if layer < cfg["first_k_dense"]:
            e = build_dense_layer(g, layer, e, cfg)
        else:
            e = build_moe_layer(g, layer, e, cfg)

    e_lm = g.new_event(Scope.GLOBAL, XCDS, "lm_head")
    for xcd in range(XCDS):
        g.add(Task(TaskKind.LM_HEAD, -1, xcd, e, e_lm, Scope.GLOBAL,
                   n_split=XCDS))
    e_arg = g.new_event(Scope.GLOBAL, 1, "argmax")
    g.add(Task(TaskKind.ARGMAX, -1, 0, e_lm, e_arg, Scope.GLOBAL))
    return g


def validate(g: Graph) -> list[str]:
    """DAG checks that would otherwise show up as a deadlocked kernel."""
    errors: list[str] = []
    signalled: dict[int, int] = {}
    for t in g.tasks:
        if t.signal_event is not None:
            signalled[t.signal_event] = signalled.get(t.signal_event, 0) + 1

    for ev in g.events:
        got, want = signalled.get(ev["id"], 0), ev["producers"]
        if got != want:
            errors.append(f"event {ev['id']} ({ev['label']}): {got} producers "
                          f"signal it, descriptor says {want} — kernel would hang")

    # A task may only wait on an event every one of whose producers appears
    # earlier in the queue; otherwise AOT ordering can deadlock.
    last_producer: dict[int, int] = {}
    for t in g.tasks:
        if t.signal_event is not None:
            last_producer[t.signal_event] = max(
                last_producer.get(t.signal_event, -1), t.index)
    for t in g.tasks:
        if t.wait_event is not None and last_producer.get(t.wait_event, 1 << 30) > t.index:
            errors.append(f"task {t.index} ({t.kind.name}) waits on event "
                          f"{t.wait_event} produced later — cycle")

    # XCD-local events must not cross chiplets.
    xcds_of_event: dict[int, set[int]] = {}
    for t in g.tasks:
        if t.signal_event is not None:
            xcds_of_event.setdefault(t.signal_event, set()).add(t.xcd)
    for t in g.tasks:
        if t.wait_event is None:
            continue
        ev = g.events[t.wait_event]
        if ev["scope"] == int(Scope.XCD_LOCAL):
            producers = xcds_of_event.get(t.wait_event, set())
            if producers - {t.xcd}:
                errors.append(f"task {t.index} on XCD {t.xcd} waits on XCD-local "
                              f"event {t.wait_event} signalled from {sorted(producers)}")
    return errors


def assign_workers(g: Graph) -> None:
    """Round-robin within each XCD: AOT queues, one worker polls one flag."""
    cursor = [0] * XCDS
    for t in g.tasks:
        t.worker = cursor[t.xcd] % WORKERS_PER_XCD
        cursor[t.xcd] += 1


def report(g: Graph, cfg: dict) -> None:
    from collections import Counter
    kinds = Counter(t.kind.name for t in g.tasks)
    globals_ = sum(1 for e in g.events if e["scope"] == int(Scope.GLOBAL))
    locals_ = sum(1 for e in g.events if e["scope"] == int(Scope.XCD_LOCAL))

    moe_tasks = [t for t in g.tasks if t.layer == cfg["first_k_dense"]]
    moe_events = {t.signal_event for t in moe_tasks} | {
        t.wait_event for t in moe_tasks if t.wait_event is not None}
    moe_global = sum(1 for e in moe_events
                     if e is not None and g.events[e]["scope"] == int(Scope.GLOBAL))

    print(f"tasks: {len(g.tasks)}   events: {len(g.events)} "
          f"({globals_} global, {locals_} XCD-local)")
    print(f"\nper MoE layer: {len(moe_tasks)} tasks, "
          f"{moe_global} global events touched "
          f"(kv_chunks={cfg['kv_chunks']})")
    print("\ntask kinds:")
    for k, n in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<16} {n:6d}")

    per_xcd = [sum(1 for t in g.tasks if t.xcd == x) for x in range(XCDS)]
    print(f"\nper-XCD task counts: {per_xcd}")
    print(f"  imbalance: {max(per_xcd) - min(per_xcd)} tasks "
          f"({(max(per_xcd)/min(per_xcd) - 1)*100:.1f}%)")
    print(f"\ndescriptor memory: {len(g.tasks) * DESCRIPTOR_BYTES / 1024:.1f} KB "
          f"({len(g.tasks)} x {DESCRIPTOR_BYTES} B)")

    # Fleet-native vs fallback, weighted by how many tasks each kind accounts
    # for — a kind that is 1 task per token matters less than one that is 432.
    by_status: dict[str, int] = {}
    for kind, n in kinds.items():
        by_status[IMPLEMENTATION.get(kind, "fallback")] = \
            by_status.get(IMPLEMENTATION.get(kind, "fallback"), 0) + n
    total = sum(by_status.values())
    print("\nFleet-native vs fallback (by task count):")
    for status in ("fleet-verified", "fleet-unverified", "fallback"):
        n = by_status.get(status, 0)
        print(f"  {status:<18} {n:6d}  {100*n/total:5.1f}%")
    fb = sorted(k for k in kinds if IMPLEMENTATION.get(k, "fallback") == "fallback")
    if fb:
        print(f"  still on torch: {', '.join(fb)}")

    errors = validate(g)
    print(f"\nvalidation: {'PASS' if not errors else f'{len(errors)} ERRORS'}")
    for e in errors[:10]:
        print(f"  ! {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).resolve().parents[2] / "reference" / "dsv2lite_config.json")
    ap.add_argument("--kv-chunks", type=int, default=1,
                    help="split-KV factor: 1 = one task per head (D2), 4 = D4")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--emit", type=Path, help="write packed descriptors here")
    a = ap.parse_args()

    c = json.loads(a.config.read_text())
    cfg = {
        "layers": c["num_hidden_layers"],
        "heads": c["num_attention_heads"],
        "top_k": c["num_experts_per_tok"],
        "first_k_dense": c["first_k_dense_replace"],
        "kv_chunks": a.kv_chunks,
    }

    g = build_graph(cfg)
    assign_workers(g)

    if a.emit:
        a.emit.parent.mkdir(parents=True, exist_ok=True)
        a.emit.write_bytes(b"".join(t.pack() for t in g.tasks))
        print(f"wrote {len(g.tasks)} descriptors to {a.emit}")
    if a.report or not a.emit:
        report(g, cfg)
    raise SystemExit(1 if validate(g) else 0)


if __name__ == "__main__":
    main()
