#!/usr/bin/env python3
"""Execute the emitted queues under the kernel's event protocol, on the CPU.

`taskgraph.py --report` proves the graph is a DAG. This test goes one step
further and runs the *protocol* the device runs: 296 worker queues walked in
order, monotonic counters that are never reset, wait targets of
epoch x wait_count read from the same descriptor fields the kernel reads, and
XCD-local counters that only the producing XCD can see. It runs several
epochs (tokens) back to back and in adversarial worker orders, and fails on
the first wait that can never be satisfied — which on the GPU would be a hung
launch.

It is a model of the runtime, not the runtime, so it says nothing about
memory visibility; it does catch every mistake in the descriptor wiring, and
the wiring is where the last two deadlocks came from.

    python3 tests/test_queue_simulation.py
"""
from __future__ import annotations

import random
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "host"))

from taskgraph import (PACK_FIELDS, PACK_FORMAT, WAVES, WORKERS_PER_XCD, XCDS,  # noqa: E402
                       Flags, Scope, TaskKind, build, load_cfg)

CONFIG = ROOT / "reference" / "dsv2lite_config.json"


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {label:<40} [{'PASS' if ok else 'FAIL'}]{('  ' + detail) if detail else ''}")
    return ok


def decode(blob: bytes) -> list[dict]:
    """Unpack with the same format the kernel's struct mirrors."""
    n = len(blob) // 64
    return [dict(zip(PACK_FIELDS, struct.unpack_from(PACK_FORMAT, blob, 64 * i)))
            for i in range(n)]


def simulate(tasks: list[dict], epochs: int, order: str, seed: int = 0,
             events: list[dict] | None = None) -> str | None:
    """Run the protocol; return None on success or a message on deadlock.

    With `events` (the builder's table) every counter is also checked at the
    end of each epoch against epoch x producers: an over-signalled event
    releases its waiters early, which is silent on the device."""
    queues: dict[tuple[int, int], list[dict]] = {}
    for t in tasks:
        queues.setdefault((t["xcd"], t["worker"]), []).append(t)
    workers = sorted(queues)
    rng = random.Random(seed)
    n_events = 1 + max(max(t["signal_event"], t["local_event"] + max(t["kv_chunk"], 0))
                       for t in tasks)
    done_event = tasks[-1]["signal_event"]
    glob = [0] * n_events
    local = [[0] * n_events for _ in range(XCDS)]
    # Workers never read the global counters: they read their XCD's mirror,
    # which the scheduler refreshes asynchronously. Modelled as a copy taken
    # once per pass, so within a pass a worker sees stale values — liveness
    # must not depend on the mirror being fresh.
    mirror = [[0] * n_events for _ in range(XCDS)]

    for epoch in range(1, epochs + 1):
        head = {w: 0 for w in workers}
        ran = 0
        while any(head[w] < len(queues[w]) for w in workers):
            if order == "reverse":
                sched = list(reversed(workers))
            elif order == "random":
                sched = workers[:]
                rng.shuffle(sched)
            else:
                sched = workers
            for x in range(XCDS):                      # the schedulers' pass
                mirror[x] = glob[:]
            progressed = False
            for w in sched:
                q = queues[w]
                # A worker runs its queue in order; it may run several tasks
                # per pass if their waits are already satisfied.
                while head[w] < len(q):
                    t = q[head[w]]
                    # token boundary (in-kernel loop): embed of token t waits
                    # for the previous token's final event
                    if t["kind"] == TaskKind.EMBED and epoch > 1:
                        if mirror[t["xcd"]][done_event] < epoch - 1:
                            break
                    if t["wait_event"] >= 0:
                        target = epoch * t["wait_count"]
                        arr = (local[t["xcd"]] if t["wait_scope"] == Scope.XCD_LOCAL
                               else mirror[t["xcd"]])
                        if arr[t["wait_event"]] < target:
                            break
                        # CHUNK_WAIT: the body also waits chunks 1.. (same
                        # scope and count); modelled as all-before-start
                        if t["flags"] & Flags.CHUNK_WAIT and any(
                                arr[t["local_event"] + c] < target
                                for c in range(1, t["kv_chunk"])):
                            break
                    if t["flags"] & Flags.CHUNK_SIGNAL:
                        for c in range(t["kv_chunk"]):     # every wave, in-body
                            local[t["xcd"]][t["local_event"] + c] += WAVES
                    do_signal = True
                    if t["flags"] & Flags.SIGNAL_LAST:
                        local[t["xcd"]][t["local_event"]] += 1
                        do_signal = local[t["xcd"]][t["local_event"]] == epoch * t["n_split"]
                    if t["signal_event"] >= 0 and do_signal:
                        e = t["signal_event"]
                        local[t["xcd"]][e] += 1          # every signal arrives locally
                        if t["signal_scope"] == Scope.GLOBAL:
                            # the last arrival of this XCD's share adds the share
                            if local[t["xcd"]][e] == epoch * t["signal_xcd_count"]:
                                glob[e] += t["signal_xcd_count"]
                    head[w] += 1
                    ran += 1
                    progressed = True
            if not progressed:
                stuck = [(w, queues[w][head[w]]) for w in workers if head[w] < len(queues[w])]
                w, t = stuck[0]
                arr = local[t["xcd"]] if t["wait_scope"] == Scope.XCD_LOCAL else mirror[t["xcd"]]
                return (f"epoch {epoch}: {len(stuck)} workers stuck; e.g. worker {w} "
                        f"task {t['index']} waits on event {t['wait_event']} "
                        f"(scope {t['wait_scope']}) for {epoch * t['wait_count']}, "
                        f"counter is {arr[t['wait_event']]}")
        if ran != len(tasks):
            return f"epoch {epoch}: ran {ran} of {len(tasks)} descriptors"
        if glob[done_event] != epoch:
            return f"epoch {epoch}: final event at {glob[done_event]}, expected {epoch}"
        for ev in events or []:
            e, want = ev["id"], epoch * ev["producers"]
            got = glob[e] if ev["scope"] == Scope.GLOBAL else max(local[x][e] for x in range(XCDS))
            if ev["scope"] == Scope.GLOBAL:
                arrivals = sum(local[x][e] for x in range(XCDS))
                if arrivals != want:
                    return f"epoch {epoch}: event {e} local arrivals {arrivals} != {want}"
            if got != want:
                return (f"epoch {epoch}: event {e} ({ev['label']}) count {got}, expected {want} "
                        f"— {'over' if got > want else 'under'}-signalled")
    return None


def main() -> int:
    results = []
    variants = [dict(kv_chunks=1), dict(kv_chunks=8), dict(kv_chunks=8, prefetch=True),
                dict(kv_chunks=16, kva_shared=True),
                dict(kv_chunks=16, split_workers=18, k_chunk=512)]
    for v in variants:
        kv_chunks, prefetch = v["kv_chunks"], v.get("prefetch", False)
        g = build(load_cfg(CONFIG, **v))
        tasks = decode(b"".join(t.pack() for t in g.tasks))
        events = g.events
        print(f"{v}: {len(tasks)} descriptors, "
              f"{len({(t['xcd'], t['worker']) for t in tasks})} worker queues")

        results.append(check("every worker queue is populated",
                             len({(t["xcd"], t["worker"]) for t in tasks})
                             == XCDS * WORKERS_PER_XCD))
        for order in ("forward", "reverse", "random"):
            err = simulate(tasks, epochs=3, order=order, events=events)
            results.append(check(f"3 epochs, {order} worker order", err is None,
                                 err or "no deadlock, every counter exact"))

        # The final event must reach exactly `epochs`: that is what stops the
        # schedulers, and a task signalling the wrong event would show here.
        # (simulate() already verified every descriptor ran once per epoch.)

        # Sabotage: the bug the descriptor used to have — waiting with the
        # signal-side count — must be caught, or this test proves nothing.
        bad = [dict(t) for t in tasks]
        for t in bad:
            if t["wait_event"] >= 0 and t["kind"] == 3:   # O_PROJ waits on merge(16)
                t["wait_count"] = 296                      # ...with its own count
        err = simulate(bad, epochs=1, order="forward")
        results.append(check("wrong wait_count is detected as a hang",
                             err is not None, (err or "")[:70]))
        # Sabotage 2: if every KV chunk signalled the merge event (no
        # last-arrival rule) the global count would overshoot the producer
        # count — the simulator sees that as the final event being off, or
        # as a wait target reached early. Only meaningful with > 1 chunk.
        if kv_chunks > 1 and any(t["flags"] & Flags.SIGNAL_LAST for t in tasks):
            bad = [dict(t) for t in tasks]
            for t in bad:
                if t["flags"] & Flags.SIGNAL_LAST:
                    t["flags"] = int(t["flags"]) & ~int(Flags.SIGNAL_LAST)
            err = simulate(bad, epochs=2, order="forward", events=events)
            results.append(check("missing last-arrival rule is detected",
                                 err is not None, (err or "not detected")[:70]))

    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
