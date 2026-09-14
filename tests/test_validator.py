#!/usr/bin/env python3
"""taskgraph.validate() is the only thing between a mis-wired graph and a
hung GPU, and nothing used to call it except main().

Every check below sabotages a *built* graph the way a real edit would break
it, and requires the validator to say so. A validator that stopped checking
would be invisible otherwise: the graph still emits, the kernel still
launches, and the failure is a GPU that never finishes.

    python3 tests/test_validator.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "host"))

from taskgraph import (Flags, Scope, TaskKind, XCDS, WORKERS_PER_XCD,  # noqa: E402
                       build, load_cfg, validate)

CONFIG = ROOT / "reference" / "dsv2lite_config.json"

VARIANTS = {
    "default": dict(kv_chunks=16),
    "8 chunks": dict(kv_chunks=8),
    "kv_a replicated": dict(kv_chunks=16, kva_shared=False),
    "top-k published": dict(kv_chunks=16, topk_published=True),
    "prefetch": dict(kv_chunks=16, prefetch=True),
    "split workers": dict(kv_chunks=16, split_workers=18, k_chunk=512),
    "o_proj row split": dict(kv_chunks=16, oproj_row_split=True),
}


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {label:<52} [{'PASS' if ok else 'FAIL'}]{('  ' + detail) if detail else ''}")
    return ok


def first(tasks, pred):
    for t in tasks:
        if pred(t):
            return t
    raise AssertionError("no task matched")


def xcd_local_merge(b) -> None:
    """Release a row-split o_proj from an XCD-local merge. The graph stays
    *locally* consistent -- each XCD waits on a counter only it signals --
    so nothing else notices, while every o_proj reads 7/8 of o before the
    other XCDs have written it. Silent wrong answers, not a hang."""
    for e in {t.wait_event for t in b.tasks
              if (t.flags & Flags.OPROJ_ROW_SPLIT) and t.kind == TaskKind.O_PROJ}:
        b.events[e]["scope"] = int(Scope.XCD_LOCAL)


def sabotage(label: str, g, mutate, want: str) -> bool:
    """Apply `mutate` to a copy of the graph; the validator must complain,
    and the complaint must mention `want`."""
    bad = copy.deepcopy(g)
    mutate(bad)
    errs = validate(bad)
    hit = any(want in e for e in errs)
    return check(label, hit,
                 (errs[0][:64] if errs else "validator found nothing") if not hit else "")


def main() -> int:
    results = []

    graphs = {}
    for name, kw in VARIANTS.items():
        g = build(load_cfg(CONFIG, **kw))
        graphs[name] = g
        errs = validate(g)
        results.append(check(f"clean: {name}", not errs, errs[0][:64] if errs else ""))

    g = graphs["default"]

    results.append(sabotage(
        "row-split o_proj on an XCD-local merge is caught",
        graphs["o_proj row split"], xcd_local_merge, "row-split o_proj"))

    results.append(sabotage(
        "event producer count off by one is caught", g,
        lambda b: b.events.__setitem__(
            40, {**b.events[40], "producers": b.events[40]["producers"] + 1}),
        "kernel would hang"))

    results.append(sabotage(
        "wait-side count disagreeing with the table is caught", g,
        lambda b: setattr(first(b.tasks, lambda t: t.wait_event is not None),
                          "wait_count", 999),
        "disagree with event"))

    results.append(sabotage(
        "an XCD-local event produced from two XCDs is caught", g,
        lambda b: setattr(first(b.tasks, lambda t: t.kind == TaskKind.EXPERT_GATE_UP
                                and t.xcd == 0), "xcd", 1),
        "XCD-local"))

    results.append(sabotage(
        "waiting on an event produced later is caught", g,
        lambda b: setattr(b.tasks[0], "wait_event", b.tasks[-1].signal_event),
        "produced later"))

    results.append(sabotage(
        "a task with no valid (xcd, worker) is caught", g,
        lambda b: setattr(b.tasks[5], "worker", WORKERS_PER_XCD + 3),
        "no valid"))

    results.append(sabotage(
        "ROUTING_CACHED without its gate_up in front is caught", g,
        lambda b: setattr(first(b.tasks, lambda t: t.flags & Flags.ROUTING_CACHED),
                          "layer", 26),
        "ROUTING_CACHED"))

    results.append(sabotage(
        "CHUNK_WAIT not waiting on its chunk 0 event is caught", g,
        lambda b: setattr(first(b.tasks, lambda t: t.flags & Flags.CHUNK_WAIT),
                          "wait_event", 3),
        "CHUNK_WAIT"))

    results.append(sabotage(
        "TOPK_PUBLISH without a full-XCD counter is caught", graphs["top-k published"],
        lambda b: setattr(first(b.tasks, lambda t: t.flags & Flags.TOPK_PUBLISH),
                          "n_split", 5),
        "TOPK_PUBLISH"))

    results.append(sabotage(
        "one descriptor disagreeing on its share is caught", g,
        lambda b: setattr(first(b.tasks, lambda t: t.signal_xcd_count > 1),
                          "signal_xcd_count", 1),
        "disagree"))

    def shrink_one_whole_share(b):
        # every descriptor of one (event, xcd) group, so the descriptors stay
        # consistent with each other and only their sum is wrong
        t0 = first(b.tasks, lambda t: t.signal_xcd_count > 1)
        for t in b.tasks:
            if t.signal_event == t0.signal_event and t.xcd == t0.xcd:
                t.signal_xcd_count = t0.signal_xcd_count - 1

    results.append(sabotage(
        "per-XCD shares that do not sum to the producers are caught", g,
        shrink_one_whole_share, "do not sum to"))

    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
