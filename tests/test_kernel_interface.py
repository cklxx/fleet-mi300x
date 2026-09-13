#!/usr/bin/env python3
"""Static cross-check between the kernel and the runtime header.

No ROCm toolchain exists on the development machine, and this machine's clang
cannot resolve the C++ standard library at all, so `fleet_kernel.hip` will be
parsed for the first time on the MI300X — where the VM bills by wall-clock.

These checks are what can still be done without a compiler: that every runtime
function the kernel calls is declared, that every task kind has a handler, and
that every descriptor field the kernel reads exists in the struct. They catch
renames and typos, which is the bulk of what a first compile would catch.

    python3 tests/test_kernel_interface.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "src" / "kernels" / "fleet_kernel.hip"
RUNTIME = ROOT / "src" / "runtime" / "fleet_runtime.h"
# Task bodies included by the kernel; their symbols must also resolve.
BODIES = [ROOT / "src" / "kernels" / n for n in ("gemv.h", "attention.h", "expert.h")]


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {label:<38} [{'PASS' if ok else 'FAIL'}]{('  ' + detail) if detail else ''}")
    return ok


def main() -> int:
    k = KERNEL.read_text()
    r = RUNTIME.read_text()

    print(f"interface: {KERNEL.relative_to(ROOT)} vs {RUNTIME.relative_to(ROOT)}")
    results = []

    # 1. runtime symbols the kernel calls must be declared in the header
    called = set(re.findall(
        r"\b(fetch_task|signal_event|wait_event|xcd_barrier|run_scheduler|"
        r"xcd_id|is_scheduler|worker_id|poll|backoff)\s*\(", k))
    declared = set(re.findall(r"__device__[^\n]*?\b(\w+)\s*\(", r))
    missing = sorted(called - declared)
    results.append(check("runtime symbols declared", not missing,
                         f"{len(called)} called" if not missing else str(missing)))

    # 1b. task-body symbols from gemv.h / attention.h / expert.h. These are the
    #     newest seams and the ones a compiler would normally catch first.
    body_src = "\n".join(p.read_text() for p in BODIES)
    body_decl = set(re.findall(r"__device__[^\n]*?\b(\w+)\s*\(", body_src))
    body_calls = set(re.findall(
        r"\b(gemv_chiplet|gemv_chiplet_epi|dot_row_bf16|block_reduce_ordered|"
        r"kv_post|q_absorb|attention_chunk|merge_and_uv|apply_rope_interleaved|"
        r"resolve_unit|expert_gate_up|expert_down|dense_gate_up)\s*\(", k))
    body_missing = sorted(body_calls - body_decl)
    results.append(check("task-body symbols declared", not body_missing,
                         f"{len(body_calls)} called"
                         if not body_missing else str(body_missing)))

    # 1c. each body header must be included, or the calls resolve to nothing
    included = set(re.findall(r'#include\s+"([^"]+)"', k))
    want = {p.name for p in BODIES}
    not_included = sorted(want - {Path(i).name for i in included})
    results.append(check("task-body headers included", not not_included,
                         f"{len(want)} headers"
                         if not not_included else str(not_included)))

    # 1d. argument-count sanity for run_task: definition vs call site. A
    #     signature change that misses the call site is exactly the kind of
    #     error no static grep would otherwise notice.
    defn = re.search(r"void run_task\(([^)]*)\)", k)
    callsite = re.search(r"run_task\(([^;]*?)\);", k)
    if defn and callsite:
        n_params = len([p for p in defn.group(1).split(",") if p.strip()])
        n_args = len([a for a in callsite.group(1).split(",") if a.strip()])
        results.append(check("run_task call matches signature",
                            n_params == n_args, f"{n_params} params, {n_args} args"))
    else:
        results.append(check("run_task call matches signature", False, "not found"))

    # 2. every task kind must have an explicit case, so adding a kind to the
    #    host graph cannot silently become a no-op on the device
    defined = set(re.findall(r"(TASK_\w+)\s*=", r))
    handled = set(re.findall(r"case\s+(TASK_\w+)\s*:", k))
    unhandled = sorted(defined - handled)
    results.append(check("all task kinds have a case", not unhandled,
                         f"{len(defined)} kinds" if not unhandled else str(unhandled)))

    # 3. descriptor fields read by the kernel must exist in the struct
    body = re.search(r"struct TaskDescriptor \{(.*?)\};", r, re.S).group(1)
    fields = set(re.findall(r"\b(?:uint16_t|int16_t|uint8_t)\s+(\w+)", body))
    used = set(re.findall(r"\bt(?:->|\.)(\w+)", k))
    bad = sorted(used - fields)
    results.append(check("descriptor fields exist", not bad,
                         f"{len(used)} read" if not bad else str(bad)))

    # 4. the kernel must be launched cooperatively: grid=304 residency is what
    #    makes the scheduler/worker wait safe (design.md §4)
    results.append(check("grid constants consistent",
                         "kGrid = kXCDs * kCUsPerXCD" in r and "kXCDs = 8" in r))

    # 5. determinism rule: no float atomics anywhere (design.md §4)
    float_atomics = re.findall(r"atomicAdd\s*\(\s*[^,]*float", k)
    results.append(check("no float atomics (determinism)", not float_atomics,
                         "" if not float_atomics else str(float_atomics[:3])))

    # 6. Shared-struct fields. ModelDims/Weights/Activations/KVCache moved into
    #    fleet_types.h so the launcher could pass them by value; that split the
    #    definitions from every use of them across three files, with no compiler
    #    here to notice a typo. Check each `d.x` / `act.y` resolves.
    types = (ROOT / "src" / "runtime" / "fleet_types.h").read_text()
    users = "\n".join(p.read_text() for p in (
        KERNEL, ROOT / "src" / "host" / "fleet_launch.hip",
        ROOT / "src" / "kernels" / "expert.h"))
    all_missing: dict[str, list[str]] = {}
    total_used = 0
    for sname, var in (("ModelDims", "d"), ("Activations", "act"),
                       ("Weights", "w"), ("KVCache", "kv")):
        m = re.search(r"struct " + sname + r" \{(.*?)\n\};", types, re.S)
        if not m:
            all_missing[sname] = ["struct not found"]
            continue
        fields = set()
        for line in m.group(1).splitlines():
            fields.update(re.findall(r"(\w+)\s*(?:;|,)", re.sub(r"//.*", "", line)))
        used = set(re.findall(rf"\b{var}\.(\w+)", users))
        total_used += len(used)
        if used - fields:
            all_missing[sname] = sorted(used - fields)
    results.append(check("shared struct fields resolve", not all_missing,
                         f"{total_used} field uses"
                         if not all_missing else str(all_missing)))

    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
