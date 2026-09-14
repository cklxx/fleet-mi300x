#!/usr/bin/env python3
"""Static cross-check between the kernel, the runtime header and the launcher.

No ROCm toolchain exists on the development machine, and this machine's clang
cannot resolve the C++ standard library at all, so `fleet_kernel.hip` will be
parsed for the first time on the MI300X — where the VM bills by wall-clock.

These checks are what can still be done without a compiler: that every runtime
function the kernel calls is declared, that every task kind has a handler, that
every descriptor field the kernel reads exists in the struct, and — because
each of these was once missing — that the launcher really launches, really
sets the epoch, and that the kernel waits with the wait-side descriptor
fields rather than the signal-side ones.

    python3 tests/test_kernel_interface.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "src" / "kernels" / "fleet_kernel.hip"
RUNTIME = ROOT / "src" / "runtime" / "fleet_runtime.h"
LAUNCH = ROOT / "src" / "host" / "fleet_launch.hip"
# Task bodies included by the kernel; their symbols must also resolve.
BODIES = [ROOT / "src" / "kernels" / n for n in ("gemv.h", "attention.h", "expert.h")]


def strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {label:<38} [{'PASS' if ok else 'FAIL'}]{('  ' + detail) if detail else ''}")
    return ok


def main() -> int:
    k = KERNEL.read_text()
    r = RUNTIME.read_text()
    kc = strip_comments(k)
    lc = strip_comments(LAUNCH.read_text())

    print(f"interface: {KERNEL.relative_to(ROOT)} vs {RUNTIME.relative_to(ROOT)}")
    results = []

    # 1. runtime symbols the kernel calls must be declared in the header
    called = set(re.findall(
        r"\b(fetch_task|signal_event|wait_event|run_scheduler|claim_role|"
        r"xcd_id|poll|backoff|aborted|raise_abort|fence_release|fence_acquire)\s*\(", kc))
    declared = set(re.findall(r"__device__[^\n]*?\b(\w+)\s*\(", r))
    missing = sorted(called - declared)
    results.append(check("runtime symbols declared", not missing,
                         f"{len(called)} called" if not missing else str(missing)))

    # 1b. task-body symbols from gemv.h / attention.h / expert.h. These are the
    #     newest seams and the ones a compiler would normally catch first.
    body_src = "\n".join(p.read_text() for p in BODIES)
    body_decl = set(re.findall(r"__device__[^\n]*?\b(\w+)\s*\(", body_src))
    body_calls = set(re.findall(
        r"\b(gemv_rows|gemv_gate_up_rows|wave_dot|wave_sum|block_reduce_ordered|"
        r"stage_vector|bf16_round|silu|kv_post|q_absorb|attention_chunk|"
        r"merge_and_uv|apply_rope_interleaved|resolve_unit|expert_gate_up|"
        r"expert_down|dense_gate_up)\s*\(", kc))
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
    defn = re.search(r"(?:void|bool) run_task\(([^)]*)\)", kc)
    callsite = re.search(r"=\s*run_task\(([^;]*?)\);", kc)
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
    handled = set(re.findall(r"case\s+(TASK_\w+)\s*:", kc))
    unhandled = sorted(defined - handled)
    results.append(check("all task kinds have a case", not unhandled,
                         f"{len(defined)} kinds" if not unhandled else str(unhandled)))

    # 3. descriptor fields read by the kernel must exist in the struct
    body = re.search(r"struct TaskDescriptor \{(.*?)\};", r, re.S).group(1)
    fields = set(re.findall(r"\b(?:uint16_t|int16_t|int32_t|uint8_t)\s+(\w+)", body))
    used = set(re.findall(r"\bt(?:->|\.)(\w+)", kc))
    bad = sorted(used - fields)
    results.append(check("descriptor fields exist", not bad,
                         f"{len(used)} read" if not bad else str(bad)))

    # 3b. the wait must use the wait-side pair. Passing signal_scope/n_split
    #     to wait_event made every wait target wrong once already.
    wait_call = re.search(r"wait_event\(\s*rt,\s*tc\.epoch,\s*t->wait_event,[^;]*;", kc, re.S)
    ok_wait = bool(wait_call) and "wait_scope" in wait_call.group(0) \
        and "wait_count" in wait_call.group(0) \
        and "signal_scope" not in wait_call.group(0) and "n_split" not in wait_call.group(0)
    results.append(check("wait_event uses wait-side fields", ok_wait))

    # 4. the kernel must be launched cooperatively: grid=304 residency is what
    #    makes the scheduler/worker wait safe (design.md §4). The call must be
    #    real code, not a comment.
    results.append(check("grid constants consistent",
                         "kGrid = kXCDs * kBlocksPerXCD" in r and "kXCDs = 8" in r))
    results.append(check("cooperative launch is real code",
                         "hipLaunchCooperativeKernel(" in kc))
    # 4a. constants the host graph and the kernel must agree on
    tg = (ROOT / "src" / "host" / "taskgraph.py").read_text()
    ty = (ROOT / "src" / "runtime" / "fleet_types.h").read_text()
    gm = (ROOT / "src" / "kernels" / "gemv.h").read_text()
    pairs = [("EXPERT_K_CHUNK", tg, "kExpertKChunk", ty), ("WAVES", tg, "kWaves", gm)]
    mismatched = []
    for py_name, py_src, c_name, c_src in pairs:
        py = re.search(rf"^{py_name}\s*=\s*(\d+)", py_src, re.M)
        c = re.search(rf"{c_name}\s*=\s*(?:256\s*/\s*kWaveLanes|(\d+))", c_src)
        c_val = None
        if c:
            c_val = int(c.group(1)) if c.group(1) else 256 // 64
        if not py or c_val is None or int(py.group(1)) != c_val:
            mismatched.append(f"{py_name}={py and py.group(1)} vs {c_name}={c_val}")
    results.append(check("graph/kernel constants agree", not mismatched, ", ".join(mismatched)))

    # 4c. the memory model. Deleting either fence leaves every other gate
    #     green and yields a kernel that reads stale activations on the
    #     machine; the microbench measures them, nothing else checks that
    #     they are still called.
    def body(src, name):
        i = src.find(name + "(")
        while i > 0 and src[i - 1] != chr(10):
            i -= 1
        j = src.find(chr(10) + "}", i)
        return src[i:j] if i >= 0 and j > i else ""

    sig, wait = body(r, "signal_event"), body(r, "wait_event")
    results.append(check("signal_event releases before publishing",
                         "fence_release()" in sig and "__hip_atomic_fetch_add" in sig))
    results.append(check("wait_event acquires on both scopes",
                         "fence_acquire_local()" in wait and "fence_acquire()" in wait))
    # inside the function, not anywhere in the file: the comment above it
    # names the instruction too, and a check the comment can satisfy is not
    # a check on the code
    loc = body(r, "void fence_acquire_local")
    results.append(check("the XCD-local acquire is the L1-only one",
                         "buffer_inv sc0" in loc and "buffer_inv sc1" not in loc))
    # measured: dropping the producer-side writeback made 10 of 18 launches
    # produce wrong tokens even though the fence-free bench passes (STATUS.md)
    results.append(check("the producer writeback is unconditional",
                         "return true;" in body(r, "event_release_fenced")))

    # 4b. the epoch is what makes any wait target non-zero; the launcher must
    #     set it from the token index before each launch.
    results.append(check("launcher sets rt.epoch0 and n_tokens",
                         re.search(r"rt\.epoch0\s*=", lc) is not None
                         and re.search(r"rt\.n_tokens\s*=", lc) is not None))
    results.append(check("launcher checks the abort code",
                         "kAbortWaitTimeout" in lc and "kAbortXcdDistribution" in lc))

    # 5. determinism rule: no float atomics anywhere (design.md §4)
    all_dev = kc + strip_comments(body_src) + strip_comments(r)
    float_atomics = re.findall(r"atomicAdd\s*\(\s*[^,]*float", all_dev)
    results.append(check("no float atomics (determinism)", not float_atomics,
                         "" if not float_atomics else str(float_atomics[:3])))

    # 6. Shared-struct fields. ModelDims/Weights/Activations/KVCache live in
    #    fleet_types.h so the launcher can pass them by value; that splits the
    #    definitions from every use of them across files, with no compiler
    #    here to notice a typo. Check each `d.x` / `act.y` resolves.
    types = (ROOT / "src" / "runtime" / "fleet_types.h").read_text()
    users = "\n".join(strip_comments(p.read_text()) for p in (
        KERNEL, LAUNCH, ROOT / "src" / "kernels" / "expert.h",
        ROOT / "src" / "kernels" / "attention.h"))
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
            fields.update(re.findall(r"(\w+)\s*(?:\[\d+\])?\s*(?:;|,)", re.sub(r"//.*", "", line)))
        used = set(re.findall(rf"\b{var}\.(\w+)", users))
        total_used += len(used)
        if used - fields:
            all_missing[sname] = sorted(used - fields)
    results.append(check("shared struct fields resolve", not all_missing,
                         f"{total_used} field uses"
                         if not all_missing else str(all_missing)))

    # 7. every RuntimeState field the kernel or header reads is set by the
    #    launcher (a missing assignment is a zero, which for `epoch` means
    #    "every wait passes immediately").
    m = re.search(r"struct RuntimeState \{(.*?)\n\};", r, re.S)
    rt_fields = set(re.findall(r"(\w+)\s*;", re.sub(r"//.*", "", m.group(1))))
    rt_used = set(re.findall(r"\brt\.(\w+)", kc + strip_comments(r)))
    rt_set = set(re.findall(r"\brt\.(\w+)\s*=", lc))
    unset = sorted((rt_used & rt_fields) - rt_set)
    results.append(check("launcher sets every RuntimeState field read",
                         not unset, f"{len(rt_used & rt_fields)} fields"
                         if not unset else str(unset)))

    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
