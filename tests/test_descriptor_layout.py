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

from taskgraph import (DESCRIPTOR_BYTES, PACK_FIELDS, PACK_FORMAT,  # noqa: E402
                       Task, TaskKind, Scope)

C_SCALARS = {"uint16_t": ("H", 2), "int16_t": ("h", 2), "int32_t": ("i", 4),
             "uint8_t": ("B", 1)}

# Field order as emitted by Task.pack(); kept here so a reordering there has to
# be mirrored deliberately rather than silently.
PY_FIELDS = ["kind", "layer", "xcd", "worker", "wait_event", "signal_event",
             "signal_scope", "n_split", "head", "kv_chunk", "expert_slot",
             "wait_scope", "wait_count", "local_event", "flags",
             "signal_xcd_count", "index"]


def parse_header(path: Path) -> tuple[list[tuple[str, str, int]], int]:
    """Fields as (name, struct code, byte offset), plus total size, walking
    the struct the way the C compiler lays it out (natural alignment)."""
    body = re.search(r"struct TaskDescriptor \{(.*?)\};",
                     path.read_text(), re.S).group(1)
    fields, size = [], 0
    for line in body.splitlines():
        m = re.match(r"\s*(uint16_t|int16_t|int32_t|uint8_t)\s+(\w+)(?:\[(\d+)\])?;", line)
        if not m:
            continue
        ty, name, arr = m.groups()
        code, width = C_SCALARS[ty]
        size += (-size) % width                      # alignment padding
        if not name.startswith("_"):
            fields.append((name, code, size))
        size += width * (int(arr) if arr else 1)
    return fields, size


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {label:<34} [{'PASS' if ok else 'FAIL'}]{('  ' + detail) if detail else ''}")
    return ok


def main() -> int:
    hdr = ROOT / "src" / "runtime" / "fleet_runtime.h"
    c_fields, c_size = parse_header(hdr)
    c_names = [n for n, _, _ in c_fields]

    # Offsets on the Python side, from the pack format itself.
    py_offsets = {}
    off = 0
    it = iter(PACK_FIELDS)
    for count, code in re.findall(r"(\d*)([HhiBx])", PACK_FORMAT.replace("<", "")):
        n = int(count) if count else 1
        width = struct.calcsize(code)
        if code == "x":
            off += n
            continue
        for _ in range(n):
            py_offsets[next(it)] = (code, off)
            off += width
    c_offsets = {n: (code, o) for n, code, o in c_fields}

    from taskgraph import Flags
    sample = Task(kind=TaskKind.ATTENTION, layer=7, xcd=3, wait_event=11,
                  signal_event=12, signal_scope=Scope.XCD_LOCAL, head=5,
                  kv_chunk=2, expert_slot=-1, n_split=4, worker=9,
                  wait_scope=Scope.GLOBAL, wait_count=296, local_event=13,
                  flags=Flags.SIGNAL_LAST, signal_xcd_count=2)
    sample.index = 40000            # > int16: the d2 graph has 33k descriptors
    packed = sample.pack()

    print(f"descriptor layout: {hdr.relative_to(ROOT)} vs taskgraph.Task.pack()")
    results = [
        check("python pack size == 64", len(packed) == DESCRIPTOR_BYTES == 64,
              f"{len(packed)} B"),
        check("c struct size == 64", c_size == 64, f"{c_size} B"),
        check("field names and order match", c_names == PY_FIELDS == PACK_FIELDS,
              "" if c_names == PY_FIELDS else f"{c_names} != {PY_FIELDS}"),
        check("field types and offsets match", c_offsets == py_offsets,
              "" if c_offsets == py_offsets else
              str({k: (c_offsets.get(k), py_offsets.get(k)) for k in set(c_offsets) | set(py_offsets)
                   if c_offsets.get(k) != py_offsets.get(k)})),
    ]

    # Round-trip the sample through the C offsets and confirm every value
    # lands where the kernel will look for it.
    got = {n: struct.unpack_from("<" + code, packed, o)[0] for n, code, o in c_fields}
    expect = {"kind": int(TaskKind.ATTENTION), "layer": 7, "xcd": 3, "worker": 9,
              "wait_event": 11, "signal_event": 12,
              "signal_scope": int(Scope.XCD_LOCAL), "n_split": 4, "head": 5,
              "kv_chunk": 2, "expert_slot": -1, "index": 40000,
              "wait_scope": int(Scope.GLOBAL), "wait_count": 296,
              "local_event": 13, "flags": int(Flags.SIGNAL_LAST), "signal_xcd_count": 2}
    bad = {k: (got.get(k), expect[k]) for k in expect if got.get(k) != expect[k]}
    results.append(check("field values round-trip", not bad,
                         "" if not bad else str(bad)))

    # -1 sentinels must survive as signed, not become 65535 — including the
    # layer of embed / lm_head / argmax, which is -1 too.
    empty = Task(kind=TaskKind.EMBED, layer=-1, xcd=0, wait_event=None,
                 signal_event=None)
    empty.index = 0
    lay, = struct.unpack_from("<h", empty.pack(), 2)
    w, s = struct.unpack_from("<hh", empty.pack(), 8)
    le, = struct.unpack_from("<h", empty.pack(), c_offsets["local_event"][1])
    results.append(check("None/-1 sentinels stay signed",
                         w == -1 and s == -1 and lay == -1 and le == -1,
                         f"layer={lay} wait={w} signal={s} local={le}"))

    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
