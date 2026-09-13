#!/usr/bin/env python3
"""The host packer and the device struct must agree, byte for byte.

`taskgraph.py` writes descriptors with struct.pack; `fleet_runtime.h` reads them
as a C struct. Nothing checks that at build time, and a mismatch does not crash
— the kernel simply reads a field from the wrong offset and waits on an event
that never fires. That failure looks like a hung GPU, hours after the mistake.

So this test parses the header and compares it against the packer directly.

    python3 tests/test_descriptor_layout.py
"""
from __future__ import annotations

import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "host"))

from taskgraph import DESCRIPTOR_BYTES, Task, TaskKind, Scope  # noqa: E402

C_SCALARS = {"uint16_t": ("H", 2), "int16_t": ("h", 2), "uint8_t": ("B", 1)}

# Field order as emitted by Task.pack(); kept here so a reordering there has to
# be mirrored deliberately rather than silently.
PY_FIELDS = ["kind", "layer", "xcd", "worker", "wait_event", "signal_event",
             "signal_scope", "n_split", "head", "kv_chunk", "expert_slot", "index"]


def parse_header(path: Path) -> tuple[list[str], int]:
    body = re.search(r"struct TaskDescriptor \{(.*?)\};",
                     path.read_text(), re.S).group(1)
    fields, size = [], 0
    for line in body.splitlines():
        m = re.match(r"\s*(uint16_t|int16_t|uint8_t)\s+(\w+)(?:\[(\d+)\])?;", line)
        if not m:
            continue
        ty, name, arr = m.groups()
        size += C_SCALARS[ty][1] * (int(arr) if arr else 1)
        if not name.startswith("_"):
            fields.append(name)
    return fields, size


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {label:<34} [{'PASS' if ok else 'FAIL'}]{('  ' + detail) if detail else ''}")
    return ok


def main() -> int:
    hdr = ROOT / "src" / "runtime" / "fleet_runtime.h"
    c_fields, c_size = parse_header(hdr)

    sample = Task(kind=TaskKind.ATTENTION, layer=7, xcd=3, wait_event=11,
                  signal_event=12, signal_scope=Scope.XCD_LOCAL, head=5,
                  kv_chunk=2, expert_slot=-1, n_split=4, worker=9)
    sample.index = 123
    packed = sample.pack()

    print(f"descriptor layout: {hdr.relative_to(ROOT)} vs taskgraph.Task.pack()")
    results = [
        check("python pack size == 64", len(packed) == DESCRIPTOR_BYTES == 64,
              f"{len(packed)} B"),
        check("c struct size == 64", c_size == 64, f"{c_size} B"),
        check("field names and order match", c_fields == PY_FIELDS,
              "" if c_fields == PY_FIELDS else f"{c_fields} != {PY_FIELDS}"),
    ]

    # Round-trip the sample through the C field order and confirm every value
    # lands where the kernel will look for it.
    vals = struct.unpack_from("<HHhhhhhhhhhh", packed, 0)
    got = dict(zip(PY_FIELDS, vals))
    expect = {"kind": int(TaskKind.ATTENTION), "layer": 7, "xcd": 3, "worker": 9,
              "wait_event": 11, "signal_event": 12,
              "signal_scope": int(Scope.XCD_LOCAL), "n_split": 4, "head": 5,
              "kv_chunk": 2, "expert_slot": -1, "index": 123}
    bad = {k: (got[k], expect[k]) for k in expect if got[k] != expect[k]}
    results.append(check("field values round-trip", not bad,
                         "" if not bad else str(bad)))

    # -1 sentinels must survive as signed, not become 65535.
    empty = Task(kind=TaskKind.EMBED, layer=0, xcd=0, wait_event=None,
                 signal_event=None)
    empty.index = 0
    w, s = struct.unpack_from("<hh", empty.pack(), 8)
    results.append(check("None -> -1 sentinel, signed", w == -1 and s == -1,
                         f"wait={w} signal={s}"))

    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
