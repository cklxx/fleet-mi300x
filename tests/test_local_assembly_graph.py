#!/usr/bin/env python3
"""Opt-in F/K wiring, lifetime and monotonic event coverage (no GPU required).

This models dependencies and queue execution, not hardware cache visibility.
The frozen default digests were generated from the pre-change v0.18 graph;
new flags must not silently change the shipped descriptors or event sidecar.
"""
from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "host"))
from taskgraph import Flags, TaskKind, WORKERS_PER_XCD, XCDS, build, load_cfg, validate
from test_queue_simulation import decode, simulate

CONFIG = ROOT / "reference" / "dsv2lite_config.json"
BASELINE = {
    8: ("f5192d756bdfc50c4b93dce11acef9fcae9de89a5e81bae8625a22e440ce07d8",
        "22f1394b3290b2401373e2c0d82dcf37029ac674f2a71257d8ecf17984caa9ab"),
    16: ("a2fecf169e2b18fbe5da46af5f68f1b296b89075fd98a51fc9adc181ab0798fa",
         "11deba57db5dc719a0ba999fe092354ad0ad733a213b5c42ed0c4182c7e0814f"),
}


def check(label, ok, detail=""):
    print(f"  {label:<60} [{'PASS' if ok else 'FAIL'}] {detail}")
    return ok


def serialized(g):
    return (b"".join(t.pack() for t in g.tasks),
            "".join(f"{e['id']} {e['scope']} {e['producers']} {e['label']}\n"
                    for e in g.events).encode())


def predecessors(g):
    """Task/event DAG, including FIFO queues and waits inside task bodies."""
    n = len(g.tasks)
    pred = [[] for _ in range(n + len(g.events))]
    previous = {}
    for t in g.tasks:
        queue = (t.xcd, t.worker)
        if queue in previous:
            pred[t.index].append(previous[queue])
        previous[queue] = t.index
        waits = [] if t.wait_event is None else [t.wait_event]
        if t.flags & Flags.PUB_WAIT:
            waits.append(t.local_event)
        if t.flags & Flags.CHUNK_WAIT:
            waits += [t.local_event + c for c in range(t.kv_chunk)]
        pred[t.index] += [n + e for e in waits]
        if t.signal_event is not None:
            pred[n + t.signal_event].append(t.index)
        if t.flags & Flags.CHUNK_SIGNAL:
            for c in range(t.kv_chunk):
                pred[n + t.local_event + c].append(t.index)
    return pred


def dominates_all(pred, after, before):
    remaining, visited, stack = set(before), set(), list(pred[after])
    while remaining and stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        remaining.discard(node)
        stack.extend(pred[node])
    return not remaining


def lifetime(g):
    """Every scratch overwrite follows *all* old readers, even on slow XCDs.

    The next token is released through argmax -> embed by the runtime. Check
    the first/last uses against those endpoints as well as in-token reuse.
    """
    pred = predecessors(g)
    consumers = {}
    for t in g.tasks:
        if t.kind != TaskKind.REDUCE and t.flags & (Flags.LOCAL_FOLD | Flags.LOCAL_KVA):
            consumers.setdefault(t.wait_event, set()).add(t.index)
    last = {}
    for t in g.tasks:
        if t.kind != TaskKind.REDUCE:
            continue
        key = (t.head == 2, t.xcd)  # folds reuse one buffer; kv_a uses another
        if key in last and not dominates_all(pred, t.index, last[key]):
            return False
        if key not in last and not dominates_all(pred, t.index, {0}):
            return False
        last[key] = consumers[t.signal_event]
    return all(dominates_all(pred, len(g.tasks) - 1, readers) for readers in last.values())


def main():
    results = []
    for chunks in (8, 16):
        implicit = build(load_cfg(CONFIG, chunks))
        explicit = build(load_cfg(CONFIG, chunks, local_fold=False, local_kva=False))
        blobs = serialized(implicit)
        digests = tuple(hashlib.sha256(b).hexdigest() for b in blobs)
        results.append(check(f"{chunks} chunks: v0.18 descriptor + event SHA256", digests == BASELINE[chunks]))
        results.append(check(f"{chunks} chunks: omitted flags == explicit False bytes", blobs == serialized(explicit)))

    baseline = build(load_cfg(CONFIG, 16))
    originals = baseline.tasks
    graphs = {}
    for fold, kva in ((False, False), (True, False), (False, True), (True, True)):
        label = f"F={int(fold)} K={int(kva)}"
        g = build(load_cfg(CONFIG, 16, local_fold=fold, local_kva=kva))
        graphs[(fold, kva)] = g
        errs = validate(g)
        results.append(check(label + ": graph valid", not errs, str(errs[:1]) if errs else ""))
        old = [t for t in g.tasks if t.kind != TaskKind.REDUCE]
        results.append(check(label + ": all original workers and task order preserved",
                             [(t.kind, t.layer, t.xcd, t.worker, t.head, t.kv_chunk) for t in old]
                             == [(t.kind, t.layer, t.xcd, t.worker, t.head, t.kv_chunk) for t in originals]))
        results.append(check(label + ": original global/local event tables preserved",
                             g.events[:len(baseline.events)] == baseline.events))
        pubs = [t for t in g.tasks if t.kind == TaskKind.REDUCE]
        results.append(check(label + ": exactly 53x8 folds / 27x8 gathers",
                             len(pubs) == (53 * XCDS if fold else 0) + (27 * XCDS if kva else 0)))
        if fold or kva:
            publication = {t.signal_event: t for t in pubs}
            exact = True
            for a, b in zip(originals, old):
                want_fold = fold and (a.kind == TaskKind.NORM_ROUTER or (
                    a.kind in (TaskKind.QKV_FUSED, TaskKind.LM_HEAD) and a.flags & Flags.FOLD_PARTIALS))
                want_kva = kva and a.kind == TaskKind.ATTENTION
                want_flags = a.flags | (Flags.LOCAL_FOLD if want_fold else Flags.NONE) \
                    | (Flags.LOCAL_KVA if want_kva else Flags.NONE)
                exact &= b.flags == want_flags
                if want_fold or want_kva:
                    pub = publication.get(b.wait_event)
                    exact &= pub is not None and pub.wait_event == a.wait_event
                else:
                    exact &= a.wait_event == b.wait_event
            results.append(check(label + ": every consumer retains its exact source barrier", exact))
            results.append(check(label + ": scratch reuse waits for every reader / epoch", lifetime(g)))
        if kva:
            before = [t.pack() for t in originals if t.kind == TaskKind.Q_ABSORB]
            after = []
            for a, b in zip(originals, old):
                if a.kind == TaskKind.Q_ABSORB:
                    copy_b = copy.copy(b)
                    copy_b.index = a.index
                    after.append(copy_b.pack())
            results.append(check(label + ": q_c publishers byte-identical except index", before == after))
            results.append(check(label + ": attention retains the same PUB_WAIT payload event",
                                 all(a.local_event == b.local_event and b.flags & Flags.PUB_WAIT
                                     for a, b in zip(originals, old) if a.kind == TaskKind.ATTENTION)))
            gather_ok = True
            for pub in (t for t in pubs if t.head == 2):
                tasks = [t for t in old if t.layer == pub.layer and t.xcd == pub.xcd]
                active = {t.worker for t in tasks if t.kind in (TaskKind.Q_ABSORB, TaskKind.ATTENTION)}
                gather_ok &= len(active) == 36 and {pub.worker} == set(range(WORKERS_PER_XCD)) - active
                gather_ok &= all(t.index < pub.index and t.wait_event == pub.wait_event
                                 for t in tasks if t.kind == TaskKind.Q_ABSORB)
            results.append(check(label + ": gather uses sole idle worker after q_c emission", gather_ok))
        tasks = decode(serialized(g)[0])
        for order in ("forward", "reverse", "random"):
            err = simulate(tasks, epochs=3, order=order, seed=917, events=g.events)
            results.append(check(label + f": 3 epochs / {order} / exact counters", err is None, err or ""))

    for label, kw in (
        ("F row-split", dict(local_fold=True, oproj_row_split=True)),
        ("F replicated kv_a", dict(local_fold=True, kva_shared=False)),
        ("K 8 chunks", dict(local_kva=True, kv_chunks=8)),
        ("K per-task q_c", dict(local_kva=True, qc_published=False)),
        ("K replicated kv_a", dict(local_kva=True, kva_shared=False)),
    ):
        try:
            build(load_cfg(CONFIG, **{"kv_chunks": 16, **kw}))
            rejected = False
        except ValueError:
            rejected = True
        results.append(check(label + ": unsupported combination rejected", rejected))
    cfg = load_cfg(CONFIG, 16, local_kva=True)
    cfg["heads"] = 8
    try:
        build(cfg)
        rejected = False
    except ValueError:
        rejected = True
    results.append(check("K non-16 heads: unsupported combination rejected", rejected))

    combined = graphs[(True, True)]
    for label, mutate, expected in (
        ("assembly loses global acquire", lambda g, t: setattr(t, "wait_event", t.signal_event), "GLOBAL payload"),
        ("assembly flags disagree with buffer", lambda g, t: setattr(t, "flags", Flags.LOCAL_KVA), "invalid local assembly"),
        ("gather steals q_c worker", lambda g, t: setattr(t, "worker", next(
            q.worker for q in g.tasks if q.layer == t.layer and q.xcd == t.xcd and q.kind == TaskKind.Q_ABSORB)), "steals"),
    ):
        bad = copy.deepcopy(combined)
        pub = next(t for t in bad.tasks if t.kind == TaskKind.REDUCE
                   and (t.head == 2 if "steals" in label else t.head == 0))
        mutate(bad, pub)
        errs = validate(bad)
        results.append(check(label + ": sabotage caught", any(expected in e for e in errs)))

    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
