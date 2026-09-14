// D1 microbenchmarks (docs/design.md §7 D1, §9).
//
// These three numbers decide whether the design's latency budget survives
// contact with the hardware, and none of them needs model weights:
//
//   (a) cross-XCD event round-trip, for both schemes in §4, using exactly the
//       protocol src/runtime/fleet_runtime.h runs: agent-scope release fence
//       (buffer_wbl2 on gfx942), agent-scope atomic, agent-scope atomic-load
//       polling, agent-scope acquire fence (buffer_inv). The schemes differ in
//       where the counter lives:
//         (i)  ordinary device memory
//         (ii) hipDeviceMallocUncached memory, so the atomic resolves at the
//              Infinity Cache instead of walking the XCD L2
//       §9 budgets ~166 global events x 2-4 us; at 4.94 GB/token the whole
//       token is 1.24 ms, so an extra microsecond per event costs ~13% of it.
//
//   (b) XCD-local flag latency — producer and consumer on the same chiplet,
//       same protocol. The difference between (b) and (a) is what the
//       scheduler's mirroring (§4) can save per event at most.
//
//   (c) register-streamed GEMV bandwidth vs outstanding-load depth and tile
//       size. §12 needs ~18 KB in flight per CU to reach 75% of HBM peak;
//       this measures where that actually lands on gfx942.
//
//   (d) payload visibility — the one semantic the design rests on and (a)
//       does not check: a producer workgroup's four waves write 32 KB, the
//       workgroup barriers, *thread 0 alone* releases and signals (exactly
//       what fleet_kernel.hip does), and a consumer on another XCD acquires
//       (thread 0 alone) and verifies every word. Any stale word is a
//       failure of the protocol, not a latency question, so this runs
//       before (a).
//
//   (e) (a) again under load: the other 302 workgroups stream HBM the way
//       296 workers do in the real kernel while two workgroups ping-pong.
//       This is the number §9 should carry, not the idle one.
//
// Build:  hipcc --offload-arch=gfx942 -O3 -std=c++17 bench/microbench.hip -o build/microbench
// Run:    ./build/microbench --json results/microbench.json

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t _e = (expr);                                                    \
    if (_e != hipSuccess) {                                                    \
      std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #expr,      \
                   hipGetErrorString(_e));                                     \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

static constexpr int kXCDs = 8;
static constexpr int kCUsPerXCD = 38;
static constexpr int kGrid = kXCDs * kCUsPerXCD;  // 304 workgroups, 1 per CU
static constexpr int kBlock = 256;

// ---------------------------------------------------------------- helpers

// gfx942 exposes the chiplet id in HW_REG_XCC_ID; a workgroup uses it to find
// its own queue and flags (§4).
__device__ __forceinline__ unsigned xcc_id() {
#if defined(__gfx942__)
  unsigned id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(id));
  return id & 0xf;
#else
  return 0;
#endif
}

__device__ __forceinline__ unsigned long long clk() {
  return __builtin_amdgcn_s_memrealtime();
}

// Agent-scope atomic load: never satisfied by a stale L1/L2 line. This is the
// poll the runtime uses (fleet_runtime.h poll()).
__device__ __forceinline__ unsigned poll(unsigned* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void fence_release() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");   // buffer_wbl2 on gfx942
}
__device__ __forceinline__ void fence_acquire() {
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");   // buffer_inv sc1 on gfx942: L1 + L2
}
// The runtime's acquire for a consumer on the producer's own XCD: L1 only.
__device__ __forceinline__ void fence_acquire_local() {
#if defined(__gfx942__) || defined(__gfx940__) || defined(__gfx941__)
  asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\tbuffer_inv sc0\n\ts_waitcnt vmcnt(0)" ::: "memory");
#else
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
#endif
}
__device__ __forceinline__ void signal(unsigned* p) {
  fence_release();
  __hip_atomic_fetch_add(p, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void wait_for(unsigned* p, unsigned target) {
  while (poll(p) < target) { __builtin_amdgcn_s_sleep(1); }
  fence_acquire();
}

// ---------------------------------------------------------------- (a) event round-trip
//
// One producer workgroup signals; one consumer workgroup on a *different* XCD
// waits. Measuring a ping-pong rather than a one-way trip keeps both clocks on
// the same side of the fabric, so no cross-XCD clock skew enters the number.

enum EventScheme { SCHEME_CACHED = 0, SCHEME_UNCACHED = 1 };

// The kernel is the same for both schemes; the host allocates `ping`/`pong`
// differently. Every hop pays the full protocol (release, atomic, poll,
// acquire), so the number is what one global event costs the runtime.
__global__ __launch_bounds__(kBlock) void event_pingpong(
    unsigned* __restrict__ ping, unsigned* __restrict__ pong,
    unsigned long long* __restrict__ out_cycles, int iters) {
  const int wg = blockIdx.x;
  const bool is_producer = (wg == 0);
  const bool is_consumer = (wg == 1);
  if (!is_producer && !is_consumer) return;
  // Record both XCC_IDs so the host can confirm the pair really spans XCDs.
  if (threadIdx.x == 0) out_cycles[iters + (is_producer ? 0 : 1)] = xcc_id();

  __shared__ unsigned long long t0;

  for (int i = 1; i <= iters; ++i) {
    if (is_producer) {
      if (threadIdx.x == 0) t0 = clk();
      __syncthreads();
      if (threadIdx.x == 0) {
        signal(ping);
        wait_for(pong, (unsigned)i);      // the echo
        out_cycles[i - 1] = clk() - t0;
      }
      __syncthreads();
    } else {  // consumer, different XCD by construction of the grid
      if (threadIdx.x == 0) {
        wait_for(ping, (unsigned)i);
        signal(pong);
      }
      __syncthreads();
    }
  }
}

// ---------------------------------------------------------------- (d) payload visibility

constexpr int kPayloadWords = 8192;   // 32 KB, 32 words per thread

__device__ __forceinline__ unsigned mix(unsigned i, unsigned idx) {
  unsigned h = i * 0x9E3779B1u ^ idx * 0x85EBCA77u;
  h ^= h >> 15; h *= 0xC2B2AE3Du; h ^= h >> 13;
  return h;
}

// Producer = block 0, consumer = block 1 (different XCDs under round-robin
// placement; both record XCC_ID so the host can confirm). Per iteration the
// producer rewrites the whole payload with an iteration-dependent pattern, so
// a consumer that sees the previous iteration's data is caught.
// FENCED = false is the uncached-activation protocol (fleet_runtime.h,
// uncached_acts): the payload is in MTYPE-UC memory, the producer only
// drains its stores (s_waitcnt) before the relaxed atomic, the consumer only
// polls — no L2 writeback, no L2 invalidate anywhere.
template <bool FENCED>
__global__ __launch_bounds__(kBlock) void payload_visibility(
    unsigned* __restrict__ payload, unsigned* __restrict__ flag_a,
    unsigned* __restrict__ flag_b, unsigned* __restrict__ errors,
    unsigned long long* __restrict__ ids, int iters) {
  const int wg = blockIdx.x;
  const bool is_producer = (wg == 0);
  const bool is_consumer = (wg == 1);
  if (!is_producer && !is_consumer) return;
  if (threadIdx.x == 0) ids[is_producer ? 0 : 1] = xcc_id();

  for (int i = 1; i <= iters; ++i) {
    if (is_producer) {
      for (int k = threadIdx.x; k < kPayloadWords; k += blockDim.x) {
        payload[k] = mix((unsigned)i, (unsigned)k);   // all four waves store
      }
      __syncthreads();                                // the kernel's barrier
      if (threadIdx.x == 0) {
        if (FENCED) {
          signal(flag_a);                             // release + atomic, thread 0 only
          wait_for(flag_b, (unsigned)i);
        } else {
          __builtin_amdgcn_s_waitcnt(0);
          __hip_atomic_fetch_add(flag_a, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
          while (poll(flag_b) < (unsigned)i) { __builtin_amdgcn_s_sleep(1); }
        }
      }
      __syncthreads();
    } else {
      if (threadIdx.x == 0) {                         // poll (+ acquire), thread 0 only
        if (FENCED) wait_for(flag_a, (unsigned)i);
        else        while (poll(flag_a) < (unsigned)i) { __builtin_amdgcn_s_sleep(1); }
      }
      __syncthreads();
      unsigned bad = 0;
      for (int k = threadIdx.x; k < kPayloadWords; k += blockDim.x) {
        bad += payload[k] != mix((unsigned)i, (unsigned)k);
      }
      if (bad) atomicAdd(errors, bad);
      __syncthreads();
      if (threadIdx.x == 0) {
        if (FENCED) signal(flag_b);
        else __hip_atomic_fetch_add(flag_b, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
  }
}

// ---------------------------------------------------------------- (d') multi-producer, last-arriver flush
//
// The runtime's global-event scheme with many producers: every producer
// workgroup of one XCD writes its own 32 KB slice, drains its stores
// (s_waitcnt), and bumps an XCD-local arrival counter; only the arrival that
// completes the group does the L2 writeback and publishes. A consumer on
// another XCD (acquire, then verify all slices) checks that the single
// flush covered every producer's drained stores; a consumer on the *same*
// XCD checks that its L2 invalidate does not lose lines that were never
// written back. Roles come from XCD-local tickets, like the kernel's.
constexpr int kProducers = 37;

__global__ __launch_bounds__(kBlock) void payload_lastflush(
    unsigned* __restrict__ payload, unsigned* __restrict__ tickets,
    unsigned* __restrict__ arrivals, unsigned* __restrict__ published,
    unsigned* __restrict__ acks, unsigned* __restrict__ errors, int iters) {
  const unsigned xcd = xcc_id();
  __shared__ unsigned ticket_s;
  if (threadIdx.x == 0) ticket_s = atomicAdd(&tickets[xcd], 1u);
  __syncthreads();
  const unsigned ticket = ticket_s;
  // XCD 0: tickets 0..36 produce, ticket 37 consumes locally; XCD 1: ticket 0
  // consumes across the fabric. Everyone else leaves.
  const bool producer = xcd == 0 && ticket < kProducers;
  const bool consumer_local = xcd == 0 && ticket == kProducers;
  const bool consumer_cross = xcd == 1 && ticket == 0;
  if (!producer && !consumer_local && !consumer_cross) return;
  unsigned* mine = payload + (size_t)ticket * kPayloadWords;

  for (int i = 1; i <= iters; ++i) {
    if (producer) {
      for (int k = threadIdx.x; k < kPayloadWords; k += blockDim.x) mine[k] = mix((unsigned)i + 7u * ticket, (unsigned)k);
      __syncthreads();
      if (threadIdx.x == 0) {
        __builtin_amdgcn_s_waitcnt(0);                                         // drain to L2
        const unsigned old = __hip_atomic_fetch_add(arrivals, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        if (old + 1 == (unsigned)i * kProducers) {                             // last arrival
          fence_release();                                                     // the one wbl2
          __hip_atomic_store(published, (unsigned)i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        wait_for(acks, 2u * (unsigned)i);                                      // both consumers done
      }
      __syncthreads();
    } else {
      if (threadIdx.x == 0) {
        while (poll(published) < (unsigned)i) { __builtin_amdgcn_s_sleep(1); }
        if (consumer_local) fence_acquire_local();        // same L2: L1 invalidate only
        else                fence_acquire();              // other XCD: L1 + L2
      }
      __syncthreads();
      unsigned bad = 0;
      for (unsigned p = 0; p < kProducers; ++p) {
        const unsigned* slice = payload + (size_t)p * kPayloadWords;
        for (int k = threadIdx.x; k < kPayloadWords; k += blockDim.x) {
          bad += slice[k] != mix((unsigned)i + 7u * p, (unsigned)k);
        }
      }
      if (bad) atomicAdd(&errors[consumer_local ? 0 : 1], bad);
      __syncthreads();
      if (threadIdx.x == 0) signal(acks);
    }
  }
}

static void run_payload_lastflush(int iters, unsigned* bad_local, unsigned* bad_cross) {
  unsigned *payload, *misc, *errors;
  HIP_CHECK(hipMalloc(&payload, (size_t)(kProducers + 1) * kPayloadWords * sizeof(unsigned)));
  HIP_CHECK(hipMalloc(&misc, 4096));      // tickets[16] | arrivals | published | acks, on separate lines
  HIP_CHECK(hipMalloc(&errors, 2 * sizeof(unsigned)));
  HIP_CHECK(hipMemset(payload, 0, (size_t)(kProducers + 1) * kPayloadWords * sizeof(unsigned)));
  HIP_CHECK(hipMemset(misc, 0, 4096));
  HIP_CHECK(hipMemset(errors, 0, 2 * sizeof(unsigned)));
  hipLaunchKernelGGL(payload_lastflush, dim3(kGrid), dim3(kBlock), 0, 0,
                     payload, misc, misc + 64, misc + 128, misc + 192, errors, iters);
  HIP_CHECK(hipDeviceSynchronize());
  unsigned e[2];
  HIP_CHECK(hipMemcpy(e, errors, sizeof(e), hipMemcpyDeviceToHost));
  *bad_local = e[0]; *bad_cross = e[1];
  HIP_CHECK(hipFree(payload)); HIP_CHECK(hipFree(misc)); HIP_CHECK(hipFree(errors));
}

// ---------------------------------------------------------------- (e) event cost under load
//
// Blocks 0/1 ping-pong exactly as in (a); every other block streams a large
// buffer into registers until the producer raises `stop`. That is the
// bandwidth environment a real event lives in.
// Launched cooperatively: the streaming blocks spin until the ping-pong pair
// is done, so all kGrid blocks must be resident or the pair may never run.
// out_cycles[iters], [iters+1] receive the pair's XCC_IDs; trips[wg] counts
// each streaming block's 4-load trips so the host can report the achieved
// load bandwidth, i.e. prove the load was real.
__global__ __launch_bounds__(kBlock) void event_pingpong_loaded(
    unsigned* __restrict__ ping, unsigned* __restrict__ pong,
    unsigned long long* __restrict__ out_cycles, int iters,
    const float4* __restrict__ w, long long n_float4, float* __restrict__ sink,
    unsigned* __restrict__ stop, unsigned long long* __restrict__ trips) {
  const int wg = blockIdx.x;
  if (wg >= 2) {
    const long long stride = (long long)(gridDim.x - 2) * blockDim.x;
    long long i = (long long)(wg - 2) * blockDim.x + threadIdx.x;
    float acc = 0.f;
    unsigned long long n = 0;
    __shared__ int go;
    if (threadIdx.x == 0) go = 1;
    __syncthreads();
    while (go) {
      // 32 trips (128 loads, 2 KB per lane) between stop polls: the poll
      // is one agent-scope read by one thread, negligible next to the stream.
      for (int k = 0; k < 32; ++k) {
        float4 buf[4];
#pragma unroll
        for (int d = 0; d < 4; ++d) buf[d] = w[(i + d * stride) % n_float4];
#pragma unroll
        for (int d = 0; d < 4; ++d) acc += buf[d].x + buf[d].y + buf[d].z + buf[d].w;
        i = (i + 4 * stride) % n_float4;
      }
      n += 32;
      if (threadIdx.x == 0 && poll(stop)) go = 0;
      __syncthreads();
    }
    if (threadIdx.x == 0) trips[wg] = n;
    if (acc == 1234.5678f) sink[wg] = acc;
    return;
  }
  const bool is_producer = (wg == 0);
  if (threadIdx.x == 0) out_cycles[iters + (is_producer ? 0 : 1)] = xcc_id();
  __shared__ unsigned long long t0;
  for (int i = 1; i <= iters; ++i) {
    if (is_producer) {
      if (threadIdx.x == 0) t0 = clk();
      __syncthreads();
      if (threadIdx.x == 0) {
        signal(ping);
        wait_for(pong, (unsigned)i);
        out_cycles[i - 1] = clk() - t0;
      }
      __syncthreads();
    } else {
      if (threadIdx.x == 0) {
        wait_for(ping, (unsigned)i);
        signal(pong);
      }
      __syncthreads();
    }
  }
  if (is_producer && threadIdx.x == 0) {
    __hip_atomic_store(stop, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
}

// ---------------------------------------------------------------- (b) XCD-local mirror
//
// The design's claim: a worker should poll a flag written on its own chiplet,
// not a counter on another one. This pins producer and consumer to the same
// XCD and runs the same protocol as (a); the gap between the two is the most
// the scheduler's mirroring can save per event.

__global__ __launch_bounds__(kBlock) void xcd_local_flag(
    unsigned* __restrict__ flag, unsigned long long* __restrict__ out_cycles,
    int iters) {
  const int wg = blockIdx.x;
  // Workgroups are dispatched round-robin over XCDs, so wg and wg+kXCDs land
  // on the same chiplet. xcc_id() is recorded by the host-side check below.
  const bool is_writer = (wg == 0);
  const bool is_reader = (wg == kXCDs);
  if (!is_writer && !is_reader) return;
  if (threadIdx.x == 0) out_cycles[iters + (is_writer ? 0 : 1)] = xcc_id();

  __shared__ unsigned long long t0;
  for (int i = 1; i <= iters; ++i) {
    if (is_writer) {
      if (threadIdx.x == 0) {
        t0 = clk();
        signal(flag);
        wait_for(flag + 16, (unsigned)i);   // separate cache line
        out_cycles[i - 1] = clk() - t0;
      }
      __syncthreads();
    } else {
      if (threadIdx.x == 0) {
        wait_for(flag, (unsigned)i);
        signal(flag + 16);
      }
      __syncthreads();
    }
  }
}

// ---------------------------------------------------------------- (c) streamed GEMV
//
// One workgroup per CU streams a slice of a big weight matrix straight into
// registers (no LDS staging) and does the fp32 FMAs. DEPTH controls how many
// dwordx4 loads are in flight before the first s_waitcnt, which is the knob
// §12 cares about: 18 GB/s per CU x ~1 us latency = ~18 KB in flight.

template <int DEPTH>
__global__ __launch_bounds__(kBlock, 1) void gemv_stream(
    const float4* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ y, long long n_float4) {
  const long long stride = (long long)gridDim.x * blockDim.x;
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;

  float acc = 0.f;
  float4 buf[DEPTH];

  while (i + (DEPTH - 1) * stride < n_float4) {
#pragma unroll
    for (int d = 0; d < DEPTH; ++d) buf[d] = w[i + d * stride];  // all issued
#pragma unroll
    for (int d = 0; d < DEPTH; ++d) {                            // then consumed
      acc += buf[d].x + buf[d].y + buf[d].z + buf[d].w;
    }
    i += DEPTH * stride;
  }
  for (; i < n_float4; i += stride) {
    float4 v = w[i];
    acc += v.x + v.y + v.z + v.w;
  }
  // Keep the accumulator live without a second pass over memory.
  if (acc == 1234.5678f) y[blockIdx.x] = acc + x[0];
}

// ---------------------------------------------------------------- (f) the expert phase, isolated
//
// The kernel's own gemv_gate_up_rows / gemv_rows (src/kernels/gemv.h) over
// 27 layers of fake experts — 8 units of 17.3 MB per layer, 3.7 GB in all so
// the 256 MB Infinity Cache cannot hide it — with the exact (xcd, worker)
// row split the persistent kernel uses. If this reaches the streaming
// ceiling of (c) the losses inside the kernel are interference; if it caps
// lower, the bound is this code path's access pattern.

#include "../src/kernels/gemv.h"

constexpr int kHidden = 2048, kInter = 1408, kLayersF = 27, kUnits = 8;
constexpr size_t kUnitElems = (size_t)3 * kInter * kHidden;   // gate/up interleaved + down

// Per block and layer, the block's own busy time (thread 0's clock) goes to
// `busy[L * grid + block]`: the mean over blocks is the throughput view,
// the max over blocks is what a phase costs the persistent kernel, whose
// next task waits for the slowest worker.
// sync: 0 = none (throughput view); 1 = the persistent kernel's handshake
// between gate_up and down — every block releases (buffer_wbl2) and bumps an
// XCD-local counter, then all blocks of the unit wait for it and acquire
// (buffer_inv); 2 = the design's §4 variant — blocks drain their stores
// (s_waitcnt) and bump the counter, only the *last* one releases; waiters
// acquire as before. The counters are monotonic across layers and launches.
__global__ __launch_bounds__(kBlock, 1) void expert_phase(
    const __hip_bfloat16* __restrict__ experts, float* __restrict__ h,
    float* __restrict__ out, unsigned long long* __restrict__ busy,
    unsigned* __restrict__ ctr, unsigned base, int layers, int do_down, int sync) {
  __shared__ float x[kHidden];
  for (int i = threadIdx.x; i < kHidden; i += kBlock) x[i] = 0.001f * (i % 7);
  __syncthreads();
  const int xcd = blockIdx.x % kXCDs;           // which unit this block streams
  const int worker = blockIdx.x / kXCDs;        // 0..37 (38 per unit here)
  const int n_workers = kCUsPerXCD;
  for (int L = 0; L < layers; ++L) {
    const unsigned long long t0 = clk();
    const __hip_bfloat16* unit = experts + ((size_t)L * kUnits + xcd) * kUnitElems;
    float* hx = h + (size_t)xcd * kInter;
    fleet::gemv_gate_up_rows(unit, x, hx, kInter, kHidden, 0, 1, worker, n_workers);
    if (do_down) {
      __syncthreads();
      if (sync) {
        const unsigned target = base + (unsigned)(L + 1) * n_workers;
        if (threadIdx.x == 0) {
          if (sync == 1) {
            signal(&ctr[xcd * 16]);                             // release + atomic
          } else {
            __builtin_amdgcn_s_waitcnt(0);                      // stores are in L2
            const unsigned old = __hip_atomic_fetch_add(&ctr[xcd * 16], 1u,
                                     __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            if (old + 1 == target) fence_release();             // last one flushes
          }
          wait_for(&ctr[xcd * 16], target);                     // poll + acquire
        }
        __syncthreads();
      }
      fleet::gemv_rows(unit + (size_t)2 * kInter * kHidden, kInter, hx,
                       out + (size_t)xcd * kHidden, nullptr, fleet::EPI_BF16, 1.f,
                       kHidden, kInter, 0, 1, worker, n_workers);
    }
    __syncthreads();
    if (threadIdx.x == 0) busy[(size_t)L * gridDim.x + blockIdx.x] = clk() - t0;
  }
}

static double run_expert_phase(int do_down, int sync, double ns) {
  const size_t elems = (size_t)kLayersF * kUnits * kUnitElems;
  __hip_bfloat16* w; float *h, *out; unsigned long long* busy; unsigned* ctr;
  HIP_CHECK(hipMalloc(&w, elems * sizeof(__hip_bfloat16)));
  HIP_CHECK(hipMalloc(&h, kUnits * kInter * sizeof(float)));
  HIP_CHECK(hipMalloc(&out, kUnits * kHidden * sizeof(float)));
  HIP_CHECK(hipMalloc(&busy, (size_t)kLayersF * kGrid * sizeof(unsigned long long)));
  HIP_CHECK(hipMalloc(&ctr, kXCDs * 16 * sizeof(unsigned)));
  HIP_CHECK(hipMemset(ctr, 0, kXCDs * 16 * sizeof(unsigned)));
  HIP_CHECK(hipMemset(w, 0x3c, elems * sizeof(__hip_bfloat16)));   // ~1.0 in bf16
  hipEvent_t a, b;
  HIP_CHECK(hipEventCreate(&a)); HIP_CHECK(hipEventCreate(&b));
  const unsigned per_launch = (unsigned)kLayersF * kCUsPerXCD;
  hipLaunchKernelGGL(expert_phase, dim3(kGrid), dim3(kBlock), 0, 0, w, h, out, busy, ctr, 0u, kLayersF, do_down, sync);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipEventRecord(a));
  hipLaunchKernelGGL(expert_phase, dim3(kGrid), dim3(kBlock), 0, 0, w, h, out, busy, ctr, per_launch, kLayersF, do_down, sync);
  HIP_CHECK(hipEventRecord(b));
  HIP_CHECK(hipEventSynchronize(b));
  float ms = 0; HIP_CHECK(hipEventElapsedTime(&ms, a, b));
  std::vector<unsigned long long> hb((size_t)kLayersF * kGrid);
  HIP_CHECK(hipMemcpy(hb.data(), busy, hb.size() * sizeof(hb[0]), hipMemcpyDeviceToHost));
  double mean_us = 0, max_us = 0;
  for (int L = 0; L < kLayersF; ++L) {
    unsigned long long mx = 0, sum = 0;
    for (int g = 0; g < kGrid; ++g) { sum += hb[(size_t)L * kGrid + g]; mx = std::max(mx, hb[(size_t)L * kGrid + g]); }
    mean_us += sum / (double)kGrid * ns / 1e3; max_us += mx * ns / 1e3;
  }
  mean_us /= kLayersF; max_us /= kLayersF;
  const double bytes = (double)kLayersF * kUnits * (do_down ? 3.0 : 2.0) * kInter * kHidden * 2;
  static const char* names[] = {"no handshake", "per-worker release", "last-worker release"};
  std::printf("    %-22s %-20s %6.1f us/layer, %.2f TB/s; per block: mean %.1f us, slowest %.1f us\n",
              do_down ? "gate_up + down" : "gate_up only", names[sync], ms * 1e3 / kLayersF,
              bytes / (ms * 1e-3) / 1e12, mean_us, max_us);
  HIP_CHECK(hipFree(w)); HIP_CHECK(hipFree(h)); HIP_CHECK(hipFree(out)); HIP_CHECK(hipFree(busy));
  HIP_CHECK(hipFree(ctr));
  return bytes / (ms * 1e-3) / 1e12;
}

// ---------------------------------------------------------------- driver

struct Stats {
  double median, p95, mean;
};

static Stats summarise(std::vector<unsigned long long> v, double ns_per_tick) {
  std::sort(v.begin(), v.end());
  Stats s{};
  s.median = v[v.size() / 2] * ns_per_tick;
  s.p95 = v[(size_t)(v.size() * 0.95)] * ns_per_tick;
  double sum = 0;
  for (auto t : v) sum += t * ns_per_tick;
  s.mean = sum / v.size();
  return s;
}

static double tick_ns() {
  // s_memrealtime runs at a fixed 100 MHz on gfx9.
  int rate_khz = 0;
  HIP_CHECK(hipDeviceGetAttribute(&rate_khz, hipDeviceAttributeWallClockRate, 0));
  return rate_khz > 0 ? 1e6 / rate_khz : 10.0;
}

static Stats run_event(int scheme, int iters, double ns, bool* cross_xcd) {
  unsigned *ping, *pong;
  unsigned long long* cyc;
  HIP_CHECK(hipMalloc(&cyc, (iters + 2) * sizeof(unsigned long long)));
  if (scheme == SCHEME_UNCACHED) {
    // Uncached device memory: the atomic resolves past the XCD L2.
    HIP_CHECK(hipExtMallocWithFlags((void**)&ping, 64, hipDeviceMallocUncached));
    HIP_CHECK(hipExtMallocWithFlags((void**)&pong, 64, hipDeviceMallocUncached));
  } else {
    HIP_CHECK(hipMalloc(&ping, 64));
    HIP_CHECK(hipMalloc(&pong, 64));
  }
  HIP_CHECK(hipMemset(ping, 0, 64));
  HIP_CHECK(hipMemset(pong, 0, 64));

  hipLaunchKernelGGL(event_pingpong, dim3(kGrid), dim3(kBlock), 0, 0, ping, pong,
                     cyc, iters);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<unsigned long long> h(iters + 2);
  HIP_CHECK(hipMemcpy(h.data(), cyc, h.size() * sizeof(h[0]), hipMemcpyDeviceToHost));
  *cross_xcd = h[iters] != h[iters + 1];
  h.resize(iters);
  HIP_CHECK(hipFree(ping)); HIP_CHECK(hipFree(pong)); HIP_CHECK(hipFree(cyc));
  return summarise(std::move(h), ns);
}

// (d): returns the number of stale words seen over all iterations.
// uncached: payload and flags in MTYPE-UC memory, protocol without fences (d'').
static unsigned run_payload(int iters, bool* cross_xcd, bool uncached = false) {
  unsigned *payload, *flags, *errors;
  unsigned long long* ids;
  if (uncached) {
    HIP_CHECK(hipExtMallocWithFlags((void**)&payload, kPayloadWords * sizeof(unsigned),
                                    hipDeviceMallocUncached));
    HIP_CHECK(hipExtMallocWithFlags((void**)&flags, 256, hipDeviceMallocUncached));
  } else {
    HIP_CHECK(hipMalloc(&payload, kPayloadWords * sizeof(unsigned)));
    HIP_CHECK(hipMalloc(&flags, 256));
  }
  HIP_CHECK(hipMalloc(&errors, sizeof(unsigned)));
  HIP_CHECK(hipMalloc(&ids, 2 * sizeof(unsigned long long)));
  HIP_CHECK(hipMemset(payload, 0, kPayloadWords * sizeof(unsigned)));
  HIP_CHECK(hipMemset(flags, 0, 256));
  HIP_CHECK(hipMemset(errors, 0, sizeof(unsigned)));
  if (uncached) {
    hipLaunchKernelGGL(payload_visibility<false>, dim3(kGrid), dim3(kBlock), 0, 0,
                       payload, flags, flags + 16, errors, ids, iters);
  } else {
    hipLaunchKernelGGL(payload_visibility<true>, dim3(kGrid), dim3(kBlock), 0, 0,
                       payload, flags, flags + 16, errors, ids, iters);
  }
  HIP_CHECK(hipDeviceSynchronize());
  unsigned bad = 0;
  unsigned long long h_ids[2];
  HIP_CHECK(hipMemcpy(&bad, errors, sizeof(bad), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(h_ids, ids, sizeof(h_ids), hipMemcpyDeviceToHost));
  *cross_xcd = h_ids[0] != h_ids[1];
  HIP_CHECK(hipFree(payload)); HIP_CHECK(hipFree(flags));
  HIP_CHECK(hipFree(errors)); HIP_CHECK(hipFree(ids));
  return bad;
}

// (e): (a) with the other 302 workgroups streaming `bytes` of HBM. Cooperative
// launch so all blocks are resident (the streamers spin until the pair is
// done). Reports the achieved load bandwidth and whether the pair spanned XCDs.
static Stats run_event_loaded(int iters, double ns, size_t bytes, double* load_tbs,
                              bool* cross_xcd) {
  unsigned *ping, *pong, *stop;
  unsigned long long *cyc, *trips;
  float4* w; float* sink;
  HIP_CHECK(hipMalloc(&ping, 64)); HIP_CHECK(hipMalloc(&pong, 64));
  HIP_CHECK(hipMalloc(&stop, 64));
  HIP_CHECK(hipMalloc(&cyc, (iters + 2) * sizeof(unsigned long long)));
  HIP_CHECK(hipMalloc(&trips, kGrid * sizeof(unsigned long long)));
  HIP_CHECK(hipMalloc(&w, bytes)); HIP_CHECK(hipMalloc(&sink, kGrid * sizeof(float)));
  HIP_CHECK(hipMemset(ping, 0, 64)); HIP_CHECK(hipMemset(pong, 0, 64));
  HIP_CHECK(hipMemset(stop, 0, 64)); HIP_CHECK(hipMemset(w, 1, bytes));
  HIP_CHECK(hipMemset(trips, 0, kGrid * sizeof(unsigned long long)));

  int per_cu = 0;
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &per_cu, (const void*)event_pingpong_loaded, kBlock, 0));
  if (per_cu < 1) { std::fprintf(stderr, "(e): kernel does not fit one block per CU\n"); std::exit(1); }
  long long n4 = (long long)(bytes / sizeof(float4));
  void* args[] = {&ping, &pong, &cyc, (void*)&iters, &w, &n4, &sink, &stop, &trips};
  hipEvent_t a, b;
  HIP_CHECK(hipEventCreate(&a)); HIP_CHECK(hipEventCreate(&b));
  HIP_CHECK(hipEventRecord(a));
  HIP_CHECK(hipLaunchCooperativeKernel((const void*)event_pingpong_loaded, dim3(kGrid),
                                       dim3(kBlock), args, 0, nullptr));
  HIP_CHECK(hipEventRecord(b));
  HIP_CHECK(hipEventSynchronize(b));
  float ms = 0; HIP_CHECK(hipEventElapsedTime(&ms, a, b));

  std::vector<unsigned long long> h(iters + 2), tr(kGrid);
  HIP_CHECK(hipMemcpy(h.data(), cyc, h.size() * sizeof(h[0]), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(tr.data(), trips, tr.size() * sizeof(tr[0]), hipMemcpyDeviceToHost));
  *cross_xcd = h[iters] != h[iters + 1];
  double loads = 0;
  for (auto t : tr) loads += (double)t * 4 * kBlock;              // float4 loads
  *load_tbs = loads * sizeof(float4) / (ms * 1e-3) / 1e12;
  h.resize(iters);
  HIP_CHECK(hipEventDestroy(a)); HIP_CHECK(hipEventDestroy(b));
  HIP_CHECK(hipFree(ping)); HIP_CHECK(hipFree(pong)); HIP_CHECK(hipFree(stop));
  HIP_CHECK(hipFree(cyc)); HIP_CHECK(hipFree(trips)); HIP_CHECK(hipFree(w));
  HIP_CHECK(hipFree(sink));
  return summarise(std::move(h), ns);
}

static Stats run_xcd_local(int iters, double ns, bool* same_xcd) {
  unsigned* flag;
  unsigned long long* cyc;
  HIP_CHECK(hipMalloc(&flag, 256));
  HIP_CHECK(hipMemset(flag, 0, 256));
  HIP_CHECK(hipMalloc(&cyc, (iters + 2) * sizeof(unsigned long long)));
  hipLaunchKernelGGL(xcd_local_flag, dim3(kGrid), dim3(kBlock), 0, 0, flag, cyc, iters);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<unsigned long long> h(iters + 2);
  HIP_CHECK(hipMemcpy(h.data(), cyc, h.size() * sizeof(h[0]), hipMemcpyDeviceToHost));
  *same_xcd = h[iters] == h[iters + 1];   // both sides recorded their XCC_ID
  h.resize(iters);
  HIP_CHECK(hipFree(flag)); HIP_CHECK(hipFree(cyc));
  return summarise(std::move(h), ns);
}

template <int DEPTH>
static double run_gemv(size_t bytes, int reps) {
  const long long n4 = bytes / sizeof(float4);
  float4* w; float *x, *y;
  HIP_CHECK(hipMalloc(&w, bytes));
  HIP_CHECK(hipMalloc(&x, 4096 * sizeof(float)));
  HIP_CHECK(hipMalloc(&y, kGrid * sizeof(float)));
  HIP_CHECK(hipMemset(w, 1, bytes));

  hipEvent_t a, b;
  HIP_CHECK(hipEventCreate(&a)); HIP_CHECK(hipEventCreate(&b));
  hipLaunchKernelGGL((gemv_stream<DEPTH>), dim3(kGrid), dim3(kBlock), 0, 0, w, x, y, n4);
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipEventRecord(a));
  for (int i = 0; i < reps; ++i)
    hipLaunchKernelGGL((gemv_stream<DEPTH>), dim3(kGrid), dim3(kBlock), 0, 0, w, x, y, n4);
  HIP_CHECK(hipEventRecord(b));
  HIP_CHECK(hipEventSynchronize(b));
  float ms = 0; HIP_CHECK(hipEventElapsedTime(&ms, a, b));
  HIP_CHECK(hipFree(w)); HIP_CHECK(hipFree(x)); HIP_CHECK(hipFree(y));
  HIP_CHECK(hipEventDestroy(a)); HIP_CHECK(hipEventDestroy(b));
  return (double)bytes * reps / (ms * 1e-3) / 1e12;  // TB/s
}

int main(int argc, char** argv) {
  const char* json_path = nullptr;
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--json") && i + 1 < argc) json_path = argv[++i];

  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  const double ns = tick_ns();
  std::printf("device: %s  CUs %d  arch %s\n", prop.name, prop.multiProcessorCount,
              prop.gcnArchName);
  std::printf("grid: %d workgroups x %d threads   tick %.2f ns\n\n", kGrid, kBlock, ns);

  const int iters = 2000;
  std::printf("(d) cross-XCD payload visibility: 4 waves store 32 KB, thread 0 releases, "
              "thread 0 acquires on another XCD, every word checked, %d rounds\n", iters);
  bool cross_d = false;
  const unsigned stale = run_payload(iters, &cross_d);
  const bool payload_ok = !stale && cross_d;
  std::printf("    stale words: %u of %llu  [%s]%s\n\n", stale,
              (unsigned long long)iters * kPayloadWords, payload_ok ? "PASS" : "FAIL",
              cross_d ? "" : "   !! producer and consumer were on the SAME XCD: not a cross-XCD result");
  if (stale) {
    std::printf("    the release/acquire protocol does not publish the workgroup's stores; "
                "stop here — every number below is meaningless until this passes\n");
  }

  std::printf("(d'') the same payload in uncached (MTYPE UC) memory with NO fences: producer drains "
              "stores + relaxed atomic, consumer polls, %d rounds (the --uncached-acts protocol)\n", iters);
  bool cross_uc = false;
  const unsigned stale_uc = run_payload(iters, &cross_uc, /*uncached=*/true);
  const bool payload_uc_ok = !stale_uc && cross_uc;
  std::printf("    stale words: %u of %llu  [%s]%s\n\n", stale_uc,
              (unsigned long long)iters * kPayloadWords, payload_uc_ok ? "PASS" : "FAIL",
              cross_uc ? "" : "   !! same XCD: not a cross-XCD result");

  std::printf("(d') %d producers on one XCD, drained stores, ONE last-arriver flush; a same-XCD consumer "
              "(L1-only acquire) and a cross-XCD consumer (L1+L2 acquire) verify every word, %d rounds\n",
              kProducers, iters / 4);
  unsigned bad_local = 0, bad_cross = 0;
  run_payload_lastflush(iters / 4, &bad_local, &bad_cross);
  const bool lastflush_ok = bad_local == 0 && bad_cross == 0;
  std::printf("    stale words: same-XCD consumer %u, cross-XCD consumer %u, of %llu each  [%s]\n\n",
              bad_local, bad_cross, (unsigned long long)(iters / 4) * kProducers * kPayloadWords,
              lastflush_ok ? "PASS" : "FAIL");

  std::printf("(a) cross-XCD event round-trip (ping-pong, us; release+atomic+poll+acquire)\n");
  bool cross_a = false, cross_b = false;
  Stats wbl2 = run_event(SCHEME_CACHED, iters, ns, &cross_a);
  Stats wt = run_event(SCHEME_UNCACHED, iters, ns, &cross_b);
  std::printf("    (i)  counters in device memory   median %6.2f  p95 %6.2f\n",
              wbl2.median / 1e3, wbl2.p95 / 1e3);
  std::printf("    (ii) counters uncached           median %6.2f  p95 %6.2f\n",
              wt.median / 1e3, wt.p95 / 1e3);
  if (!cross_a || !cross_b) {
    std::printf("    !! blocks 0 and 1 landed on the SAME XCD; this is not a "
                "cross-XCD number (round-robin placement assumption failed)\n");
  }
  std::printf("    -> design picks the cheaper: %s  (FLEET_UNCACHED_EVENTS=1 selects ii)\n\n",
              wt.median < wbl2.median ? "(ii)" : "(i)");

  std::printf("(b) same-XCD flag round-trip (us)\n");
  bool same_xcd = false;
  Stats loc = run_xcd_local(iters, ns, &same_xcd);
  std::printf("    median %6.3f  p95 %6.3f   (vs cross-XCD %.2f)%s\n\n",
              loc.median / 1e3, loc.p95 / 1e3,
              std::min(wbl2.median, wt.median) / 1e3,
              same_xcd ? "" : "   !! producer and consumer were NOT on the same XCD; "
                              "the round-robin placement assumption failed");

  const size_t bytes = 2ull << 30;  // 2 GiB, larger than the 256 MB LLC
  std::printf("(e) cross-XCD event round-trip while 302 workgroups stream HBM (us)\n");
  double load_tbs = 0; bool cross_e = false;
  Stats loaded = run_event_loaded(iters, ns, bytes, &load_tbs, &cross_e);
  std::printf("    median %6.2f  p95 %6.2f   (idle: %.2f)  under %.2f TB/s of streaming load"
              "  <- the value for design.md §9%s\n\n",
              loaded.median / 1e3, loaded.p95 / 1e3, wbl2.median / 1e3, load_tbs,
              cross_e ? "" : "\n    !! the pair was on the SAME XCD");

  std::printf("(c) register-streamed read bandwidth by in-flight depth\n");
  double bw[6] = {};
  bw[0] = run_gemv<1>(bytes, 5);   std::printf("    depth  1: %5.2f TB/s\n", bw[0]);
  bw[1] = run_gemv<2>(bytes, 5);   std::printf("    depth  2: %5.2f TB/s\n", bw[1]);
  bw[2] = run_gemv<4>(bytes, 5);   std::printf("    depth  4: %5.2f TB/s\n", bw[2]);
  bw[3] = run_gemv<8>(bytes, 5);   std::printf("    depth  8: %5.2f TB/s\n", bw[3]);
  bw[4] = run_gemv<12>(bytes, 5);  std::printf("    depth 12: %5.2f TB/s\n", bw[4]);
  bw[5] = run_gemv<16>(bytes, 5);  std::printf("    depth 16: %5.2f TB/s\n", bw[5]);
  const double best = *std::max_element(bw, bw + 6);
  std::printf("    -> best %.2f TB/s = %.0f%% of 5.3 peak "
              "(design assumes 4.0 achievable)\n", best, best / 5.3 * 100);

  std::printf("\n(f) the kernel's expert GEMVs in isolation (gemv.h, 8 x 17.3 MB units/layer, 27 layers)\n");
  const double f_gu = run_expert_phase(0, 0, ns);
  const double f_all = run_expert_phase(1, 0, ns);
  const double f_sync1 = run_expert_phase(1, 1, ns);
  const double f_sync2 = run_expert_phase(1, 2, ns);
  std::printf("    -> %.0f%% of (c)'s ceiling for gate_up alone, %.0f%% with down; with the kernel's "
              "per-worker handshake %.0f%%, with the last-worker flush %.0f%%\n",
              f_gu / best * 100, f_all / best * 100, f_sync1 / best * 100, f_sync2 / best * 100);

  if (json_path) {
    FILE* f = std::fopen(json_path, "w");
    std::fprintf(f,
        "{\n  \"device\": \"%s\",\n  \"arch\": \"%s\",\n  \"cus\": %d,\n"
        "  \"payload_stale_words\": %u, \"payload_cross_xcd\": %s,\n"
        "  \"payload_uncached_stale_words\": %u, \"payload_uncached_cross_xcd\": %s,\n"
        "  \"event_cached_us\": {\"median\": %.3f, \"p95\": %.3f},\n"
        "  \"event_uncached_us\": {\"median\": %.3f, \"p95\": %.3f},\n"
        "  \"event_loaded_us\": {\"median\": %.3f, \"p95\": %.3f, \"load_tbs\": %.3f, \"cross_xcd\": %s},\n"
        "  \"xcd_local_us\": {\"median\": %.4f, \"p95\": %.4f},\n"
        "  \"gemv_tbs\": [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f],\n"
        "  \"gemv_best_tbs\": %.3f\n}\n",
        prop.name, prop.gcnArchName, prop.multiProcessorCount,
        stale, cross_d ? "true" : "false",
        stale_uc, cross_uc ? "true" : "false",
        wbl2.median / 1e3, wbl2.p95 / 1e3, wt.median / 1e3, wt.p95 / 1e3,
        loaded.median / 1e3, loaded.p95 / 1e3, load_tbs, cross_e ? "true" : "false",
        loc.median / 1e3, loc.p95 / 1e3,
        bw[0], bw[1], bw[2], bw[3], bw[4], bw[5], best);
    std::fclose(f);
    std::printf("\nwrote %s\n", json_path);
  }
  return (payload_ok && lastflush_ok && payload_uc_ok) ? 0 : 1;
}
