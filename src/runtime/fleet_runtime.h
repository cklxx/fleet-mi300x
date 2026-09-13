// Fleet runtime: AOT task queues, hierarchical events, XCD discovery.
//
// docs/design.md §4 and §8. Task code sees only three operations —
// fetch_task, signal_event, wait_event — so the tasks stay runtime-agnostic
// and can later be moved onto the upstream ROCm/fleet-chiplet-megakernel
// without being rewritten.
//
// The protocol, in one paragraph: descriptors are built on the host, never
// mutated, and pre-assigned to per-XCD worker queues (ahead-of-time launch, so
// a worker waits on its own next event rather than being dispatched by a
// scheduler after the fact — one synchronisation hop instead of two). Each XCD
// dedicates one CU to a scheduler whose only job is to mirror global event
// counters into that XCD's flag array, so 37 workers poll a local array
// instead of 296 workers polling the global counters.
//
// Memory model. MI300X L2 is per XCD and not coherent across XCDs for
// ordinary device memory, so every cross-workgroup handshake here is written
// against the LLVM AMDGPU memory model for gfx942 rather than against cache
// folklore: producers release with an agent-scope fence (buffer_wbl2), the
// counter is an agent-scope atomic, consumers acquire with an agent-scope
// fence (buffer_inv), and polling uses agent-scope atomic loads so a stale
// L1/L2 line can never satisfy the wait. XCD-local events use the same fences
// today; the fence-free L2-resident variant the paper describes is a measured
// optimisation for D1 (b), not something to assume before it is measured.
#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace fleet {

constexpr int kXCDs = 8;
constexpr int kCUsPerXCD = 38;
constexpr int kWorkersPerXCD = kCUsPerXCD - 1;   // one CU is the scheduler
constexpr int kGrid = kXCDs * kCUsPerXCD;        // 304 workgroups, 1 per CU
constexpr int kBlock = 256;                      // 4 waves, 1 wave/SIMD
constexpr int kWaves = kBlock / 64;
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

// 64 bytes, laid out to match taskgraph.py's struct.pack("<Hhhh hhhh hhhh h 2x i 32x").
// Any drift here is silent and catastrophic, so static_assert guards the size
// and tests/test_descriptor_layout.py re-checks the field offsets from Python.
//
// A wait needs the *waited* event's scope and producer count, which are
// properties of that event, not of this task's own signal — so both sides are
// stored explicitly. Passing the signal-side pair to wait_event was the bug
// that made every wait target wrong.
struct TaskDescriptor {
    uint16_t kind;
    int16_t  layer;           // -1 for embed / lm_head / argmax
    int16_t  xcd;
    int16_t  worker;
    int16_t  wait_event;      // -1 = no dependency
    int16_t  signal_event;    // -1 = signals nothing
    int16_t  signal_scope;    // EventScope of signal_event
    int16_t  n_split;         // task-kind specific: kv_chunks for attention/merge
    int16_t  head;            // attention / merge, else -1
    int16_t  kv_chunk;        // split-KV index, else -1
    int16_t  expert_slot;     // >=0 routed slot k, <0 shared half, else -1
    int16_t  wait_scope;      // EventScope of wait_event
    int16_t  wait_count;      // producers of wait_event (per epoch)
    int16_t  _pad0;
    int32_t  index;           // position in the global task list (> 32767)
    uint8_t  _pad[32];
};
static_assert(sizeof(TaskDescriptor) == 64, "descriptor must stay 64 B");

// Device-side state, allocated once by the host.
struct RuntimeState {
    const TaskDescriptor* __restrict__ tasks;   // all descriptors, immutable
    const int32_t* __restrict__ queue_offset;   // [kXCDs * kWorkersPerXCD + 1]
    const int32_t* __restrict__ queue_index;    // task indices, grouped per worker

    uint32_t* __restrict__ global_events;       // [kMaxEvents], device scope
    uint32_t* __restrict__ xcd_flags;           // [kXCDs][kMaxEvents], mirror
    uint32_t* __restrict__ xcd_counters;        // [kXCDs][kMaxEvents], local events

    uint32_t* __restrict__ xcd_arrivals;        // [kXCDs] role tickets, reset per launch
    uint32_t* __restrict__ grid_arrivals;       // [1] grid barrier, reset per launch
    uint32_t* __restrict__ abort;               // [1] 0 = fine, else a reason code

    // The ids of the GLOBAL-scope events, compact. XCD-local events never
    // touch global_events, so the scheduler must not poll their slots: with
    // 165 global among 805 events that would be 80% wasted fabric reads.
    const int32_t* __restrict__ global_event_ids;
    int32_t  n_global_events;

    uint32_t epoch;        // 1-based token index; counters are never reset (§12)
    int32_t  n_events;
    int32_t  done_event;   // the graph's final event: schedulers stop at epoch
    int32_t  use_uncached_counters;  // event scheme (ii) from §4, set by D1 (a)
    int32_t  smoke;        // 1 = run the protocol only, skip every task body
    int32_t  dump_layers;  // 1 = copy every layer's output x to act.layer_dump
    uint32_t spin_limit;   // polls before a wait declares a deadlock

    // Optional per-descriptor trace (design.md §10): [n_descriptors][3] of
    // s_memrealtime ticks — wait done, task done — plus the XCD id. nullptr
    // disables it. Written by thread 0 of the worker, so it costs one store.
    uint64_t* __restrict__ trace;
};

// Abort reason codes, read back by the host after the launch.
constexpr uint32_t kAbortNone = 0;
constexpr uint32_t kAbortXcdDistribution = 0x10000000u;  // | xcd << 8 | count
constexpr uint32_t kAbortWaitTimeout = 0x20000000u;      // | event id
constexpr uint32_t kAbortBarrierTimeout = 0x30000000u;

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

// Agent-scope atomic load: never satisfied by a stale L1/L2 line.
__device__ __forceinline__ uint32_t poll(uint32_t* p) {
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ void backoff() {
    // Keep polling off the fabric between reads; s_sleep 1 is ~64 cycles.
    __builtin_amdgcn_s_sleep(1);
}

__device__ __forceinline__ void fence_release() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");   // buffer_wbl2 + waitcnt
}

__device__ __forceinline__ void fence_acquire() {
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");   // buffer_inv
}

__device__ __forceinline__ bool aborted(const RuntimeState& rt) {
    return poll(rt.abort) != kAbortNone;
}

__device__ __forceinline__ void raise_abort(const RuntimeState& rt, uint32_t code) {
    // First reason wins; later ones are consequences of it.
    uint32_t expected = kAbortNone;
    __hip_atomic_compare_exchange_strong(rt.abort, &expected, code,
                                         __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                         __HIP_MEMORY_SCOPE_AGENT);
}

// ---------------------------------------------------------------- roles
//
// Which workgroup is the scheduler and which are workers is decided by an
// XCD-local ticket, not by blockIdx: the hardware places blocks on XCDs
// round-robin, but nothing guarantees blockIdx 0..7 land one per XCD, and an
// XCD with no scheduler would hang every worker on it. The ticket makes the
// role follow the physical placement. Returns -1 for the scheduler, else the
// worker id; the caller must then pass claim_role's grid barrier, which
// converts a wrong distribution (an XCD with != 38 blocks) into an abort.
__device__ __forceinline__ int claim_role(const RuntimeState& rt, int xcd,
                                          int* slot_smem) {
    if (threadIdx.x == 0) {
        *slot_smem = (int)atomicAdd(&rt.xcd_arrivals[xcd], 1u);
        atomicAdd(rt.grid_arrivals, 1u);
    }
    __syncthreads();
    const int slot = *slot_smem;

    // Grid barrier: cooperative launch guarantees residency, so this cannot
    // deadlock unless fewer than kGrid blocks were launched at all.
    if (threadIdx.x == 0) {
        uint32_t spins = 0;
        while (poll(rt.grid_arrivals) < (uint32_t)kGrid) {
            backoff();
            if (++spins > rt.spin_limit) { raise_abort(rt, kAbortBarrierTimeout); break; }
        }
        const uint32_t here = poll(&rt.xcd_arrivals[xcd]);
        if (here != (uint32_t)kCUsPerXCD) {
            raise_abort(rt, kAbortXcdDistribution | ((uint32_t)xcd << 8) | (here & 0xff));
        }
    }
    __syncthreads();
    return slot - 1;
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

// signal_event: publish this task's stores, then count. Called by one thread
// after a __syncthreads(), so every wave's stores precede the fence.
//
// Scheme (i)  (default) the release fence writes back this XCD's dirty L2
//             lines, then the counter atomic resolves device-wide.
// Scheme (ii) counters in uncached memory (hipDeviceMallocUncached): the
//             atomic resolves at the Infinity Cache. The fence is the same;
//             D1 microbenchmark (a) measures whether the allocation matters.
__device__ __forceinline__ void signal_event(
        const RuntimeState& rt, int event, EventScope scope, int xcd) {
    if (event < 0) return;
    fence_release();
    uint32_t* counter = (scope == SCOPE_XCD_LOCAL)
        ? &rt.xcd_counters[xcd * kMaxEvents + event]
        : &rt.global_events[event];
    __hip_atomic_fetch_add(counter, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// wait_event: block until the event has `producers` signals for this epoch.
// Global events are read from this XCD's mirror (written by its scheduler);
// XCD-local events are read from the local counter directly. Called by one
// thread; the caller's __syncthreads() releases the other waves, and the
// acquire fence here is what makes the producers' data visible to them.
//
// Returns false if the wait was abandoned because of an abort (this worker's
// own timeout or someone else's), so the worker can drain and exit instead of
// hanging the launch.
__device__ __forceinline__ bool wait_event(
        const RuntimeState& rt, int event, EventScope scope, int xcd,
        int producers) {
    if (event < 0) return true;
    const uint32_t target = rt.epoch * (uint32_t)producers;
    uint32_t* p = (scope == SCOPE_XCD_LOCAL)
        ? &rt.xcd_counters[xcd * kMaxEvents + event]
        : &rt.xcd_flags[xcd * kMaxEvents + event];
    uint32_t spins = 0;
    while (poll(p) < target) {
        backoff();
        if ((++spins & 1023u) == 0) {
            if (aborted(rt)) return false;
            if (spins > rt.spin_limit) {
                raise_abort(rt, kAbortWaitTimeout | (uint32_t)event);
                return false;
            }
        }
    }
    fence_acquire();
    return true;
}

// ---------------------------------------------------------------- scheduler
//
// The per-XCD scheduler's entire job: mirror the global counters (only the
// global-scope events, from the compact id list) into this XCD's flag array
// so its 37 workers poll that instead of the global counters. Global-counter
// polling drops from 296 pollers to 8. It stops once the graph's final event
// reaches this epoch — every task precedes that event transitively, so
// nothing can still be waiting.
__device__ __forceinline__ void run_scheduler(const RuntimeState& rt, int xcd) {
    asm volatile("s_setprio 3");     // scheduler waves win arbitration
    const int lane = threadIdx.x;
    uint32_t* mirror = rt.xcd_flags + xcd * kMaxEvents;
    const uint32_t done_target = rt.epoch;
    __shared__ int stop;
    if (lane == 0) stop = 0;
    __syncthreads();
    while (!stop) {
        // Relay with a formal happens-before chain: the producer's release
        // fence -> its counter increment -> our relaxed read -> our acquire
        // fence -> our release fence -> our mirror store -> the worker's
        // relaxed read of the mirror -> the worker's acquire fence. Without
        // the two fences here the worker would acquire on an object the
        // producer never released on, and the payload's visibility would rest
        // on cache behaviour rather than on the memory model.
        bool changed = false;
        for (int k = lane; k < rt.n_global_events; k += blockDim.x) {
            const int e = rt.global_event_ids[k];
            const uint32_t g = poll(&rt.global_events[e]);
            if (poll(&mirror[e]) != g) {
                if (!changed) { fence_acquire(); fence_release(); changed = true; }
                __hip_atomic_store(&mirror[e], g, __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_AGENT);
            }
        }
        backoff();
        if (lane == 0 && (poll(&rt.global_events[rt.done_event]) >= done_target ||
                          aborted(rt))) {
            stop = 1;
        }
        __syncthreads();
    }
    asm volatile("s_setprio 0");
}

}  // namespace fleet
