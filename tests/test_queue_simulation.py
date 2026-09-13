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

from taskgraph import (PACK_FIELDS, PACK_FORMAT, WORKERS_PER_XCD, XCDS,  # noqa: E402
                       Scope, build, load_cfg)

CONFIG = ROOT / "reference" / "dsv2lite_config.json"


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {label:<40} [{'PASS' if ok else 'FAIL'}]{('  ' + detail) if detail else ''}")
    return ok


def decode(blob: bytes) -> list[dict]:
    """Unpack with the same format the kernel's struct mirrors."""
    n = len(blob) // 64
    return [dict(zip(PACK_FIELDS, struct.unpack_from(PACK_FORMAT, blob, 64 * i)))
            for i in range(n)]


def simulate(tasks: list[dict], epochs: int, order: str, seed: int = 0) -> str | None:
    """Run the protocol; return None on success or a message on deadlock."""
    queues: dict[tuple[int, int], list[dict]] = {}
    for t in tasks:
        queues.setdefault((t["xcd"], t["worker"]), []).append(t)
    workers = sorted(queues)
    rng = random.Random(seed)
    n_events = 1 + max(t["signal_event"] for t in tasks)
    glob = [0] * n_events
    local = [[0] * n_events for _ in range(XCDS)]

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
            progressed = False
            for w in sched:
                q = queues[w]
                # A worker runs its queue in order; it may run several tasks
                # per pass if their waits are already satisfied.
                while head[w] < len(q):
                    t = q[head[w]]
                    if t["wait_event"] >= 0:
                        target = epoch * t["wait_count"]
                        arr = local[t["xcd"]] if t["wait_scope"] == Scope.XCD_LOCAL else glob
                        if arr[t["wait_event"]] < target:
                            break
                    if t["signal_event"] >= 0:
                        if t["signal_scope"] == Scope.XCD_LOCAL:
                            local[t["xcd"]][t["signal_event"]] += 1
                        else:
                            glob[t["signal_event"]] += 1
                    head[w] += 1
                    ran += 1
                    progressed = True
            if not progressed:
                stuck = [(w, queues[w][head[w]]) for w in workers if head[w] < len(queues[w])]
                w, t = stuck[0]
                arr = local[t["xcd"]] if t["wait_scope"] == Scope.XCD_LOCAL else glob
                return (f"epoch {epoch}: {len(stuck)} workers stuck; e.g. worker {w} "
                        f"task {t['index']} waits on event {t['wait_event']} "
                        f"(scope {t['wait_scope']}) for {epoch * t['wait_count']}, "
                        f"counter is {arr[t['wait_event']]}")
        if ran != len(tasks):
            return f"epoch {epoch}: ran {ran} of {len(tasks)} descriptors"
    return None


def main() -> int:
    results = []
    for kv_chunks in (1, 4):
        g = build(load_cfg(CONFIG, kv_chunks))
        tasks = decode(b"".join(t.pack() for t in g.tasks))
        print(f"kv_chunks={kv_chunks}: {len(tasks)} descriptors, "
              f"{len({(t['xcd'], t['worker']) for t in tasks})} worker queues")

        results.append(check("every worker queue is populated",
                             len({(t["xcd"], t["worker"]) for t in tasks})
                             == XCDS * WORKERS_PER_XCD))
        for order in ("forward", "reverse", "random"):
            err = simulate(tasks, epochs=3, order=order)
            results.append(check(f"3 epochs, {order} worker order", err is None,
                                 err or "no deadlock, counters monotonic"))

        # Sabotage: the bug the descriptor used to have — waiting with the
        # signal-side count — must be caught, or this test proves nothing.
        bad = [dict(t) for t in tasks]
        for t in bad:
            if t["wait_event"] >= 0 and t["kind"] == 3:   # O_PROJ waits on merge(16)
                t["wait_count"] = 296                      # ...with its own count
        err = simulate(bad, epochs=1, order="forward")
        results.append(check("wrong wait_count is detected as a hang",
                             err is not None, (err or "")[:70]))

    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
