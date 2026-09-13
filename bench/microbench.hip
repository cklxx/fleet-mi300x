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
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");   // buffer_inv on gfx942
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
        signal(flag_a);                               // release + atomic, thread 0 only
        wait_for(flag_b, (unsigned)i);
      }
      __syncthreads();
    } else {
      if (threadIdx.x == 0) wait_for(flag_a, (unsigned)i);   // poll + acquire, thread 0 only
      __syncthreads();
      unsigned bad = 0;
      for (int k = threadIdx.x; k < kPayloadWords; k += blockDim.x) {
        bad += payload[k] != mix((unsigned)i, (unsigned)k);
      }
      if (bad) atomicAdd(errors, bad);
      __syncthreads();
      if (threadIdx.x == 0) signal(flag_b);
    }
  }
}

// ---------------------------------------------------------------- (e) event cost under load
//
// Blocks 0/1 ping-pong exactly as in (a); every other block streams a large
// buffer into registers until the producer raises `stop`. That is the
// bandwidth environment a real event lives in.
__global__ __launch_bounds__(kBlock) void event_pingpong_loaded(
    unsigned* __restrict__ ping, unsigned* __restrict__ pong,
    unsigned long long* __restrict__ out_cycles, int iters,
    const float4* __restrict__ w, long long n_float4, float* __restrict__ sink,
    unsigned* __restrict__ stop) {
  const int wg = blockIdx.x;
  if (wg >= 2) {
    const long long stride = (long long)(gridDim.x - 2) * blockDim.x;
    long long i = (long long)(wg - 2) * blockDim.x + threadIdx.x;
    float acc = 0.f;
    while (!poll(stop)) {
      float4 buf[4];
#pragma unroll
      for (int d = 0; d < 4; ++d) buf[d] = w[(i + d * stride) % n_float4];
#pragma unroll
      for (int d = 0; d < 4; ++d) acc += buf[d].x + buf[d].y + buf[d].z + buf[d].w;
      i = (i + 4 * stride) % n_float4;
    }
    if (acc == 1234.5678f) sink[wg] = acc;
    return;
  }
  const bool is_producer = (wg == 0);
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
static unsigned run_payload(int iters, bool* cross_xcd) {
  unsigned *payload, *flags, *errors;
  unsigned long long* ids;
  HIP_CHECK(hipMalloc(&payload, kPayloadWords * sizeof(unsigned)));
  HIP_CHECK(hipMalloc(&flags, 256));
  HIP_CHECK(hipMalloc(&errors, sizeof(unsigned)));
  HIP_CHECK(hipMalloc(&ids, 2 * sizeof(unsigned long long)));
  HIP_CHECK(hipMemset(payload, 0, kPayloadWords * sizeof(unsigned)));
  HIP_CHECK(hipMemset(flags, 0, 256));
  HIP_CHECK(hipMemset(errors, 0, sizeof(unsigned)));
  hipLaunchKernelGGL(payload_visibility, dim3(kGrid), dim3(kBlock), 0, 0,
                     payload, flags, flags + 16, errors, ids, iters);
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

// (e): (a) with the other 302 workgroups streaming `bytes` of HBM.
static Stats run_event_loaded(int iters, double ns, size_t bytes) {
  unsigned *ping, *pong, *stop;
  unsigned long long* cyc;
  float4* w; float* sink;
  HIP_CHECK(hipMalloc(&ping, 64)); HIP_CHECK(hipMalloc(&pong, 64));
  HIP_CHECK(hipMalloc(&stop, 64));
  HIP_CHECK(hipMalloc(&cyc, iters * sizeof(unsigned long long)));
  HIP_CHECK(hipMalloc(&w, bytes)); HIP_CHECK(hipMalloc(&sink, kGrid * sizeof(float)));
  HIP_CHECK(hipMemset(ping, 0, 64)); HIP_CHECK(hipMemset(pong, 0, 64));
  HIP_CHECK(hipMemset(stop, 0, 64)); HIP_CHECK(hipMemset(w, 1, bytes));
  hipLaunchKernelGGL(event_pingpong_loaded, dim3(kGrid), dim3(kBlock), 0, 0,
                     ping, pong, cyc, iters, w, (long long)(bytes / sizeof(float4)),
                     sink, stop);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<unsigned long long> h(iters);
  HIP_CHECK(hipMemcpy(h.data(), cyc, iters * sizeof(h[0]), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(ping)); HIP_CHECK(hipFree(pong)); HIP_CHECK(hipFree(stop));
  HIP_CHECK(hipFree(cyc)); HIP_CHECK(hipFree(w)); HIP_CHECK(hipFree(sink));
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
  std::printf("    stale words: %u of %llu  [%s]%s\n\n", stale,
              (unsigned long long)iters * kPayloadWords, stale ? "FAIL" : "PASS",
              cross_d ? "" : "   !! producer and consumer were on the SAME XCD");
  if (stale) {
    std::printf("    the release/acquire protocol does not publish the workgroup's stores; "
                "stop here — every number below is meaningless until this passes\n");
  }

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
  Stats loaded = run_event_loaded(iters, ns, bytes);
  std::printf("    median %6.2f  p95 %6.2f   (idle: %.2f)  <- the value for design.md §9\n\n",
              loaded.median / 1e3, loaded.p95 / 1e3, wbl2.median / 1e3);

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

  if (json_path) {
    FILE* f = std::fopen(json_path, "w");
    std::fprintf(f,
        "{\n  \"device\": \"%s\",\n  \"arch\": \"%s\",\n  \"cus\": %d,\n"
        "  \"payload_stale_words\": %u, \"payload_cross_xcd\": %s,\n"
        "  \"event_cached_us\": {\"median\": %.3f, \"p95\": %.3f},\n"
        "  \"event_uncached_us\": {\"median\": %.3f, \"p95\": %.3f},\n"
        "  \"event_loaded_us\": {\"median\": %.3f, \"p95\": %.3f},\n"
        "  \"xcd_local_us\": {\"median\": %.4f, \"p95\": %.4f},\n"
        "  \"gemv_tbs\": [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f],\n"
        "  \"gemv_best_tbs\": %.3f\n}\n",
        prop.name, prop.gcnArchName, prop.multiProcessorCount,
        stale, cross_d ? "true" : "false",
        wbl2.median / 1e3, wbl2.p95 / 1e3, wt.median / 1e3, wt.p95 / 1e3,
        loaded.median / 1e3, loaded.p95 / 1e3,
        loc.median / 1e3, loc.p95 / 1e3,
        bw[0], bw[1], bw[2], bw[3], bw[4], bw[5], best);
    std::fclose(f);
    std::printf("\nwrote %s\n", json_path);
  }
  return stale ? 1 : 0;
}
