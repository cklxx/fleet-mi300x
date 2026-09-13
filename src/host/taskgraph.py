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

Two granularities are reported. A *logical task* is a node of the graph above
(66 per MoE layer). A *descriptor* is what a worker dequeues: a CU-task is one
descriptor, but a Chiplet-task fans out to one descriptor per worker on each
XCD it spans (37 per XCD), each carrying its worker id, so the kernel's row
split is a pure function of (xcd, worker) and the signalled event simply has
8 x 37 producers. Nothing on the device ever has to broadcast a task.

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
# Must match struct TaskDescriptor in src/runtime/fleet_runtime.h field for
# field; tests/test_descriptor_layout.py checks that it does.
PACK_FORMAT = "<Hhhh hhhh hhhh h 2x i 32x"
# Field order of PACK_FORMAT; the layout test compares it with the C struct.
PACK_FIELDS = ["kind", "layer", "xcd", "worker", "wait_event", "signal_event",
               "signal_scope", "n_split", "head", "kv_chunk", "expert_slot",
               "wait_scope", "wait_count", "index"]


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
    """Synchronisation scope of an event (§4)."""
    NONE = 0
    XCD_LOCAL = 1      # counter on the producing XCD, no scheduler hop
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
    n_split: int = 1           # attention/merge: number of KV chunks
    worker: int = -1           # pinned for Chiplet-tasks, else assign_workers()
    index: int = -1
    wait_scope: Scope = Scope.NONE   # filled by finalize() from the event table
    wait_count: int = 0              # producers of wait_event, per epoch
    logical: int = -1                # graph node this descriptor belongs to

    def pack(self) -> bytes:
        """64-byte descriptor; layout is fixed so the kernel can index it."""
        return struct.pack(
            PACK_FORMAT,
            int(self.kind), self.layer, self.xcd, self.worker,
            -1 if self.wait_event is None else self.wait_event,
            -1 if self.signal_event is None else self.signal_event,
            int(self.signal_scope), self.n_split,
            self.head, self.kv_chunk, self.expert_slot,
            int(self.wait_scope), self.wait_count, self.index,
        )


@dataclass
class Graph:
    tasks: list[Task] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    n_logical: int = 0

    def new_event(self, scope: Scope, producers: int, label: str) -> int:
        eid = len(self.events)
        self.events.append({"id": eid, "scope": int(scope),
                            "producers": producers, "label": label})
        return eid

    def add(self, t: Task) -> Task:
        t.index = len(self.tasks)
        self.tasks.append(t)
        return t

    def cu_task(self, t: Task) -> Task:
        """A single-workgroup task."""
        t.logical = self.n_logical
        self.n_logical += 1
        return self.add(t)

    def chiplet_task(self, proto: Task, xcds) -> None:
        """Chiplet-tasks: one logical task per XCD in `xcds` (the design's
        "Chiplet-task x8"), each a descriptor per worker of that XCD.

        The event they signal must have been created with
        len(xcds) * WORKERS_PER_XCD producers; validate() checks that.
        """
        for xcd in xcds:
            lid = self.n_logical
            self.n_logical += 1
            for w in range(WORKERS_PER_XCD):
                t = Task(**{**proto.__dict__, "xcd": xcd, "worker": w,
                            "logical": lid})
                self.add(t)


def chiplet_producers(n_xcds: int = XCDS) -> int:
    return n_xcds * WORKERS_PER_XCD


def build_attention(g: Graph, layer: int, e_qkv: int, cfg: dict) -> int:
    """Attention + merge for one layer; returns the merge event (global)."""
    heads, kv_chunks = cfg["heads"], cfg["kv_chunks"]
    heads_per_xcd = heads // XCDS

    # XCD k owns heads 2k, 2k+1 — 3072 q columns / 8 = exactly two heads — so
    # the merge below stays XCD-local.
    e_attn = [g.new_event(Scope.XCD_LOCAL, kv_chunks, f"L{layer}.attn.h{h}")
              for h in range(heads)]
    for h in range(heads):
        for chunk in range(kv_chunks):
            g.cu_task(Task(TaskKind.ATTENTION, layer, h // heads_per_xcd, e_qkv,
                           e_attn[h], Scope.XCD_LOCAL, head=h, kv_chunk=chunk,
                           n_split=kv_chunks))

    e_merge = g.new_event(Scope.GLOBAL, heads, f"L{layer}.merge")
    for h in range(heads):
        g.cu_task(Task(TaskKind.MERGE_UV, layer, h // heads_per_xcd, e_attn[h],
                       e_merge, Scope.GLOBAL, head=h, n_split=kv_chunks))
    return e_merge


def build_moe_layer(g: Graph, layer: int, prev_event: int | None,
                    cfg: dict) -> int:
    """Emit one MoE layer; returns the event the next layer must wait on."""
    top_k = cfg["top_k"]

    # 1. fused q_proj ‖ kv_a GEMV. RMSNorm is recomputed per worker in the
    #    prologue (8 KB vector) instead of costing a global event.
    e_qkv = g.new_event(Scope.GLOBAL, chiplet_producers(), f"L{layer}.qkv")
    g.chiplet_task(Task(TaskKind.QKV_FUSED, layer, -1, prev_event, e_qkv,
                        Scope.GLOBAL), range(XCDS))

    # 2/3. attention, merge + W_UV.
    e_merge = build_attention(g, layer, e_qkv, cfg)

    # 4. o_proj + residual — needs all 16 heads, so this one is global.
    e_oproj = g.new_event(Scope.GLOBAL, chiplet_producers(), f"L{layer}.o_proj")
    g.chiplet_task(Task(TaskKind.O_PROJ, layer, -1, e_merge, e_oproj,
                        Scope.GLOBAL), range(XCDS))

    # 5. post-attention norm + router. Single task: needs the whole vector.
    #    Rotate the single-task XCD by layer: these signal global events, so the
    #    placement is free, and pinning them all to XCD 0 would queue every
    #    CU-task of the run there.
    e_router = g.new_event(Scope.GLOBAL, 1, f"L{layer}.router")
    g.cu_task(Task(TaskKind.NORM_ROUTER, layer, layer % XCDS, e_oproj, e_router,
                   Scope.GLOBAL))

    # 6/7. experts: 8 balanced units = top_k routed + shared split in two.
    #      Each XCD streams exactly one unit; gate_up → down is XCD-local.
    e_gate_up = [g.new_event(Scope.XCD_LOCAL, WORKERS_PER_XCD,
                             f"L{layer}.gate_up.x{x}") for x in range(XCDS)]
    e_down = g.new_event(Scope.GLOBAL, chiplet_producers(), f"L{layer}.down")
    for xcd in range(XCDS):
        slot = xcd if xcd < top_k else -(xcd - top_k + 1)  # <0 marks shared half
        g.chiplet_task(Task(TaskKind.EXPERT_GATE_UP, layer, xcd, e_router,
                            e_gate_up[xcd], Scope.XCD_LOCAL, expert_slot=slot),
                       [xcd])
    for xcd in range(XCDS):
        slot = xcd if xcd < top_k else -(xcd - top_k + 1)
        g.chiplet_task(Task(TaskKind.EXPERT_DOWN, layer, xcd, e_gate_up[xcd],
                            e_down, Scope.GLOBAL, expert_slot=slot), [xcd])

    # 8. reduce the 8 partials + residual.
    e_out = g.new_event(Scope.GLOBAL, 1, f"L{layer}.out")
    g.cu_task(Task(TaskKind.REDUCE, layer, (layer + 4) % XCDS, e_down, e_out,
                   Scope.GLOBAL))
    return e_out


def build_dense_layer(g: Graph, layer: int, prev_event: int | None,
                      cfg: dict) -> int:
    """Layer 0: same attention path, dense MLP instead of experts.

    The dense gate_up → down boundary needs the full h[10944], which does not
    fit an XCD-local buffer, so it costs one extra global event versus MoE.
    """
    e_qkv = g.new_event(Scope.GLOBAL, chiplet_producers(), f"L{layer}.qkv")
    g.chiplet_task(Task(TaskKind.QKV_FUSED, layer, -1, prev_event, e_qkv,
                        Scope.GLOBAL), range(XCDS))

    e_merge = build_attention(g, layer, e_qkv, cfg)

    e_oproj = g.new_event(Scope.GLOBAL, chiplet_producers(), f"L{layer}.o_proj")
    g.chiplet_task(Task(TaskKind.O_PROJ, layer, -1, e_merge, e_oproj,
                        Scope.GLOBAL), range(XCDS))

    e_norm = g.new_event(Scope.GLOBAL, 1, f"L{layer}.norm")
    g.cu_task(Task(TaskKind.NORM_ROUTER, layer, layer % XCDS, e_oproj, e_norm,
                   Scope.GLOBAL))

    e_gate_up = g.new_event(Scope.GLOBAL, chiplet_producers(),
                            f"L{layer}.dense_gate_up")
    g.chiplet_task(Task(TaskKind.DENSE_GATE_UP, layer, -1, e_norm, e_gate_up,
                        Scope.GLOBAL), range(XCDS))

    e_out = g.new_event(Scope.GLOBAL, chiplet_producers(), f"L{layer}.dense_down")
    g.chiplet_task(Task(TaskKind.DENSE_DOWN, layer, -1, e_gate_up, e_out,
                        Scope.GLOBAL), range(XCDS))
    return e_out


def build_graph(cfg: dict) -> Graph:
    g = Graph()
    e = g.new_event(Scope.GLOBAL, 1, "embed")
    g.cu_task(Task(TaskKind.EMBED, -1, 0, None, e, Scope.GLOBAL))

    for layer in range(cfg["layers"]):
        if layer < cfg["first_k_dense"]:
            e = build_dense_layer(g, layer, e, cfg)
        else:
            e = build_moe_layer(g, layer, e, cfg)

    e_lm = g.new_event(Scope.GLOBAL, chiplet_producers(), "lm_head")
    g.chiplet_task(Task(TaskKind.LM_HEAD, -1, -1, e, e_lm, Scope.GLOBAL),
                   range(XCDS))
    e_arg = g.new_event(Scope.GLOBAL, 1, "argmax")
    g.cu_task(Task(TaskKind.ARGMAX, -1, 0, e_lm, e_arg, Scope.GLOBAL))
    return g


def finalize(g: Graph) -> None:
    """Copy each waited event's scope and producer count onto the waiter.

    The kernel computes its wait target as epoch x wait_count and picks the
    counter array from wait_scope; both are properties of the *waited* event.
    """
    for t in g.tasks:
        if t.wait_event is None:
            continue
        ev = g.events[t.wait_event]
        t.wait_scope = Scope(ev["scope"])
        t.wait_count = ev["producers"]


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
                          f"signal it, event table says {want} — kernel would hang")

    for t in g.tasks:
        if t.wait_event is not None:
            ev = g.events[t.wait_event]
            if t.wait_count != ev["producers"] or int(t.wait_scope) != ev["scope"]:
                errors.append(f"task {t.index}: wait-side fields "
                              f"({t.wait_scope.name}, {t.wait_count}) disagree "
                              f"with event {ev['id']} — finalize() not run?")
        if t.worker < 0 or t.worker >= WORKERS_PER_XCD or t.xcd < 0 or t.xcd >= XCDS:
            errors.append(f"task {t.index} has no valid (xcd, worker): "
                          f"({t.xcd}, {t.worker})")

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
    """Round-robin the CU-tasks within each XCD; Chiplet-task descriptors
    already carry their worker."""
    cursor = [0] * XCDS
    for t in g.tasks:
        if t.worker >= 0:
            continue
        t.worker = cursor[t.xcd] % WORKERS_PER_XCD
        cursor[t.xcd] += 1


def build(cfg: dict) -> Graph:
    g = build_graph(cfg)
    assign_workers(g)
    finalize(g)
    return g


def emit(g: Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(t.pack() for t in g.tasks))
    # Event labels alongside, so the launcher can name an event in a timeout
    # message instead of printing a number.
    side = path.with_suffix(path.suffix + ".events.txt")
    side.write_text("".join(f"{e['id']} {e['scope']} {e['producers']} {e['label']}\n"
                            for e in g.events))
    print(f"wrote {len(g.tasks)} descriptors to {path} "
          f"({len(g.tasks) * DESCRIPTOR_BYTES / 1024:.0f} KB) and {side.name}")


def report(g: Graph, cfg: dict) -> None:
    from collections import Counter
    logical_kinds = Counter()
    seen = set()
    for t in g.tasks:
        if t.logical not in seen:
            seen.add(t.logical)
            logical_kinds[t.kind.name] += 1
    desc_kinds = Counter(t.kind.name for t in g.tasks)
    globals_ = sum(1 for e in g.events if e["scope"] == int(Scope.GLOBAL))
    locals_ = sum(1 for e in g.events if e["scope"] == int(Scope.XCD_LOCAL))

    moe = [t for t in g.tasks if t.layer == cfg["first_k_dense"]]
    moe_logical = len({t.logical for t in moe})
    moe_events = {t.signal_event for t in moe}
    moe_global = sum(1 for e in moe_events if g.events[e]["scope"] == int(Scope.GLOBAL))
    moe_local = len(moe_events) - moe_global

    print(f"logical tasks: {g.n_logical}   descriptors: {len(g.tasks)}   "
          f"events: {len(g.events)} ({globals_} global, {locals_} XCD-local)")
    print(f"\nper MoE layer: {moe_logical} logical tasks, {len(moe)} descriptors, "
          f"{moe_global} global + {moe_local} XCD-local events produced "
          f"(kv_chunks={cfg['kv_chunks']})")
    print("\ntask kinds (logical / descriptors):")
    for k, n in sorted(logical_kinds.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<16} {n:6d} / {desc_kinds[k]:6d}")

    per_xcd = [sum(1 for t in g.tasks if t.xcd == x) for x in range(XCDS)]
    print(f"\nper-XCD descriptor counts: {per_xcd}")
    print(f"  imbalance: {max(per_xcd) - min(per_xcd)} descriptors "
          f"({(max(per_xcd)/min(per_xcd) - 1)*100:.1f}%)")
    per_worker = Counter((t.xcd, t.worker) for t in g.tasks)
    print(f"  per-worker queue length: min {min(per_worker.values())}, "
          f"max {max(per_worker.values())}")
    print(f"\ndescriptor memory: {len(g.tasks) * DESCRIPTOR_BYTES / 1024:.1f} KB "
          f"({len(g.tasks)} x {DESCRIPTOR_BYTES} B)")

    # Fleet-native vs fallback, weighted by logical task count.
    by_status: dict[str, int] = {}
    for kind, n in logical_kinds.items():
        st = IMPLEMENTATION.get(kind, "fallback")
        by_status[st] = by_status.get(st, 0) + n
    total = sum(by_status.values())
    print("\nFleet-native vs fallback (by logical task count):")
    for status in ("fleet-verified", "fleet-unverified", "fallback"):
        n = by_status.get(status, 0)
        print(f"  {status:<18} {n:6d}  {100*n/total:5.1f}%")
    fb = sorted(k for k in logical_kinds if IMPLEMENTATION.get(k, "fallback") == "fallback")
    if fb:
        print(f"  still on torch: {', '.join(fb)}")

    errors = validate(g)
    print(f"\nvalidation: {'PASS' if not errors else f'{len(errors)} ERRORS'}")
    for e in errors[:10]:
        print(f"  ! {e}")


def load_cfg(config: Path, kv_chunks: int) -> dict:
    c = json.loads(config.read_text())
    return {
        "layers": c["num_hidden_layers"],
        "heads": c["num_attention_heads"],
        "top_k": c["num_experts_per_tok"],
        "first_k_dense": c["first_k_dense_replace"],
        "kv_chunks": kv_chunks,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).resolve().parents[2] / "reference" / "dsv2lite_config.json")
    ap.add_argument("--kv-chunks", type=int, default=1,
                    help="split-KV factor: 1 = one task per head (D2), 4 = D4")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--emit", type=Path, help="write packed descriptors here")
    a = ap.parse_args()

    cfg = load_cfg(a.config, a.kv_chunks)
    g = build(cfg)

    if a.emit:
        emit(g, a.emit)
    if a.report or not a.emit:
        report(g, cfg)
    raise SystemExit(1 if validate(g) else 0)


if __name__ == "__main__":
    main()
