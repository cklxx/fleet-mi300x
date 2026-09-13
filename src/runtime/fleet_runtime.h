// Fleet runtime: AOT task queues, hierarchical events, XCD discovery.
//
// docs/design.md §4 and §8. Task code sees only three operations —
// fetch_task, signal_event, xcd_barrier — so the tasks stay runtime-agnostic
// and can later be moved onto the upstream ROCm/fleet-chiplet-megakernel
// without being rewritten.
//
// The protocol, in one paragraph: descriptors are built on the host, never
// mutated, and pre-assigned to per-XCD worker queues (ahead-of-time launch, so
// a worker waits on its own next event rather than being dispatched by a
// scheduler after the fact — one synchronisation hop instead of two). Each XCD
// dedicates one CU to a scheduler whose only job is to mirror global event
// counters into that XCD's L2, so 296 workers poll L2 instead of the fabric.
#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace fleet {

constexpr int kXCDs = 8;
constexpr int kCUsPerXCD = 38;
constexpr int kWorkersPerXCD = kCUsPerXCD - 1;   // one CU is the scheduler
constexpr int kGrid = kXCDs * kCUsPerXCD;        // 304 workgroups, 1 per CU
constexpr int kBlock = 256;                      // 4 waves, 1 wave/SIMD
constexpr int kMaxEvents = 4096;

// Task kinds — must match TaskKind in src/host/taskgraph.py.
enum TaskKind : uint16_t {
    TASK_QKV_FUSED = 0,
    TASK_ATTENTION = 1,
    TASK_MERGE_UV = 2,
    TASK_O_PROJ = 3,
    TASK_NORM_ROUTER = 4,
    TASK_EXPERT_GATE_UP = 5,
    TASK_EXPERT_DOWN = 6,
    TASK_REDUCE = 7,
    TASK_DENSE_GATE_UP = 8,
    TASK_DENSE_DOWN = 9,
    TASK_EMBED = 10,
    TASK_LM_HEAD = 11,
    TASK_ARGMAX = 12,
};

enum EventScope : int16_t {
    SCOPE_NONE = 0,
    SCOPE_XCD_LOCAL = 1,
    SCOPE_GLOBAL = 2,
};

// 64 bytes, laid out to match taskgraph.py's struct.pack("<HHhh hhhh hhhh 40x").
// Any drift here is silent and catastrophic, so static_assert guards the size
// and tests/test_descriptor_layout.py re-checks the field offsets from Python.
struct TaskDescriptor {
    uint16_t kind;
    uint16_t layer;
    int16_t  xcd;
    int16_t  worker;
    int16_t  wait_event;      // -1 = no dependency
    int16_t  signal_event;    // -1 = signals nothing
    int16_t  signal_scope;    // EventScope
    int16_t  n_split;         // producers sharing signal_event
    int16_t  head;            // attention / merge, else -1
    int16_t  kv_chunk;        // split-KV index, else -1
    int16_t  expert_slot;     // >=0 routed slot k, <0 shared half, else -1
    int16_t  index;           // position in the global task list
    uint8_t  _pad[40];
};
static_assert(sizeof(TaskDescriptor) == 64, "descriptor must stay 64 B");

// Device-side state, allocated once by the host.
struct RuntimeState {
    const TaskDescriptor* __restrict__ tasks;   // all descriptors, immutable
    const int32_t* __restrict__ queue_offset;   // [kXCDs * kWorkersPerXCD + 1]
    const int32_t* __restrict__ queue_index;    // task indices, grouped per worker

    uint32_t* __restrict__ global_events;       // [kMaxEvents], device scope
    uint32_t* __restrict__ xcd_flags;           // [kXCDs][kMaxEvents], in L2
    uint32_t* __restrict__ xcd_counters;        // [kXCDs][kMaxEvents], worker->worker

    uint32_t epoch;        // monotonic per token; counters are never reset (§12)
    int32_t  n_events;
    int32_t  use_uncached_counters;  // event scheme (ii) from §4, set by D1 (a)
};

// ---------------------------------------------------------------- primitives

// gfx942 exposes the chiplet id in HW_REG_XCC_ID. A workgroup uses it to find
// its own queue and flags; MI300X dispatches workgroups round-robin over XCDs,
// so this is the only reliable way to know where you are.
__device__ __forceinline__ int xcd_id() {
#if defined(__gfx942__)
    unsigned id;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(id));
    return (int)(id & 0xf);
#else
    return (int)(blockIdx.x % kXCDs);
#endif
}

__device__ __forceinline__ bool is_scheduler() {
    // One CU per XCD is the scheduler: 2.6% of CUs, irrelevant while
    // bandwidth-bound, and it removes 288 of 296 fabric pollers.
    return (blockIdx.x / kXCDs) == 0;
}

__device__ __forceinline__ int worker_id() {
    return (blockIdx.x / kXCDs) - 1;  // scheduler is -1
}

__device__ __forceinline__ uint32_t poll(const volatile uint32_t* p) {
    return __builtin_nontemporal_load((const uint32_t*)p);
}

__device__ __forceinline__ void backoff() {
    // Keep polling off the fabric between reads; s_sleep 1 is ~64 cycles.
    __builtin_amdgcn_s_sleep(1);
}

// ---------------------------------------------------------------- the three ops

// fetch_task: with AOT queues a worker already knows its whole task list, so
// this is an index lookup, not a dispatch. Returns nullptr when drained.
__device__ __forceinline__ const TaskDescriptor* fetch_task(
        const RuntimeState& rt, int xcd, int worker, int slot) {
    const int q = xcd * kWorkersPerXCD + worker;
    const int32_t begin = rt.queue_offset[q];
    const int32_t end = rt.queue_offset[q + 1];
    if (begin + slot >= end) return nullptr;
    return &rt.tasks[rt.queue_index[begin + slot]];
}

// signal_event: the last producer of an event makes it visible.
//
// Scheme (i)  buffer_wbl2 on the last worker, then a device-scope atomic.
// Scheme (ii) write-through stores plus counters in uncached memory, so the
//             atomic resolves at the Infinity Cache with no L2 flush.
// D1 microbenchmark (a) decides which; the flag lives in RuntimeState so the
// kernel does not need rebuilding to switch.
__device__ __forceinline__ void signal_event(
        const RuntimeState& rt, int event, EventScope scope, int xcd) {
    if (event < 0) return;
    if (scope == SCOPE_XCD_LOCAL) {
        __threadfence_block();
        atomicAdd(&rt.xcd_counters[xcd * kMaxEvents + event], 1u);
        return;
    }
    if (!rt.use_uncached_counters) {
        __builtin_amdgcn_buffer_wbl2();   // scheme (i): publish dirty L2 lines
    }
    __threadfence();                      // both schemes need store ordering
    atomicAdd(&rt.global_events[event], 1u);
}

// Wait for an event to reach `producers` for this epoch. Workers wait on the
// XCD-local mirror; only the scheduler touches the global counter.
__device__ __forceinline__ void wait_event(
        const RuntimeState& rt, int event, EventScope scope, int xcd,
        int producers) {
    if (event < 0) return;
    const uint32_t target = rt.epoch * (uint32_t)producers;
    const volatile uint32_t* p =
        (scope == SCOPE_XCD_LOCAL)
            ? &rt.xcd_counters[xcd * kMaxEvents + event]
            : &rt.xcd_flags[xcd * kMaxEvents + event];
    while (poll(p) < target) backoff();
}

// xcd_barrier: workers inside one Chiplet-task synchronising through L2, with
// no device fence — used for gate_up -> down and split-KV partial completion.
__device__ __forceinline__ void xcd_barrier(
        const RuntimeState& rt, int xcd, int counter, int participants) {
    __threadfence_block();
    if (threadIdx.x == 0) {
        atomicAdd(&rt.xcd_counters[xcd * kMaxEvents + counter], 1u);
        const uint32_t target = rt.epoch * (uint32_t)participants;
        while (poll(&rt.xcd_counters[xcd * kMaxEvents + counter]) < target) backoff();
    }
    __syncthreads();
}

// ---------------------------------------------------------------- scheduler
//
// The per-XCD scheduler's entire job: mirror global counters into this XCD's
// L2 so its 37 workers poll L2 instead of the fabric. Global-counter polling
// drops from 296 pollers to 8.
__device__ __forceinline__ void run_scheduler(const RuntimeState& rt, int xcd,
                                              const uint32_t* done_flag) {
    __builtin_amdgcn_s_setprio(3);   // scheduler waves win arbitration
    const int lane = threadIdx.x;
    while (!poll(done_flag)) {
        for (int e = lane; e < rt.n_events; e += blockDim.x) {
            const uint32_t g = poll(&rt.global_events[e]);
            uint32_t* mirror = &rt.xcd_flags[xcd * kMaxEvents + e];
            if (*mirror != g) *mirror = g;   // plain store: stays in this L2
        }
        backoff();
    }
    __builtin_amdgcn_s_setprio(0);
}

}  // namespace fleet
