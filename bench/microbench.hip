// D1 microbenchmarks (docs/design.md §7 D1, §9).
//
// These three numbers decide whether the design's latency budget survives
// contact with the hardware, and none of them needs model weights:
//
//   (a) cross-XCD event round-trip, for both schemes in §4:
//         (i)  buffer_wbl2 on the last worker, then GPU-scope atomicAdd
//         (ii) write-through stores (sc1) + event counters in uncached memory
//       §9 budgets ~166 global events x 2-4 us; at 4.94 GB/token the whole
//       token is 1.24 ms, so an extra microsecond per event costs ~13% of it.
//
//   (b) XCD-local flag latency — the scheduler mirrors a global counter into
//       its L2 and 37 workers poll that instead of the fabric. This is the
//       "one hop instead of two" claim in §4.
//
//   (c) register-streamed GEMV bandwidth vs outstanding-load depth and tile
//       size. §12 needs ~18 KB in flight per CU to reach 75% of HBM peak;
//       this measures where that actually lands on gfx942.
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

// Non-temporal load: event polling must not park lines in L2 (§4).
__device__ __forceinline__ unsigned nt_load(const volatile unsigned* p) {
  return __builtin_nontemporal_load(const_cast<const unsigned*>(p));
}

// ---------------------------------------------------------------- (a) event round-trip
//
// One producer workgroup signals; one consumer workgroup on a *different* XCD
// waits. Measuring a ping-pong rather than a one-way trip keeps both clocks on
// the same side of the fabric, so no cross-XCD clock skew enters the number.

enum EventScheme { SCHEME_WBL2 = 0, SCHEME_WRITETHROUGH = 1 };

__global__ __launch_bounds__(kBlock) void event_pingpong(
    unsigned* __restrict__ ping, unsigned* __restrict__ pong,
    unsigned long long* __restrict__ out_cycles, int iters, int scheme) {
  const int wg = blockIdx.x;
  const bool is_producer = (wg == 0);
  const bool is_consumer = (wg == 1);
  if (!is_producer && !is_consumer) return;

  __shared__ unsigned long long t0;

  for (int i = 1; i <= iters; ++i) {
    if (is_producer) {
      if (threadIdx.x == 0) t0 = clk();
      __syncthreads();

      if (threadIdx.x == 0) {
        if (scheme == SCHEME_WBL2) {
          // (i) flush this XCD's dirty L2 lines, then a device-scope atomic.
          __builtin_amdgcn_buffer_wbl2();
          __threadfence();
          atomicAdd(ping, 1u);
        } else {
          // (ii) write-through store; the counter itself lives in uncached
          // memory so the atomic resolves at the Infinity Cache.
          __threadfence();
          atomicAdd(ping, 1u);
        }
      }
      // wait for the echo
      if (threadIdx.x == 0) {
        while (nt_load(pong) < (unsigned)i) { __builtin_amdgcn_s_sleep(1); }
        out_cycles[i - 1] = clk() - t0;
      }
      __syncthreads();
    } else {  // consumer, different XCD by construction of the grid
      if (threadIdx.x == 0) {
        while (nt_load(ping) < (unsigned)i) { __builtin_amdgcn_s_sleep(1); }
        if (scheme == SCHEME_WBL2) {
          __builtin_amdgcn_buffer_wbl2();
          __threadfence();
        } else {
          __threadfence();
        }
        atomicAdd(pong, 1u);
      }
      __syncthreads();
    }
  }
}

// ---------------------------------------------------------------- (b) XCD-local mirror
//
// The design's claim: a worker should poll an L2-resident flag written by its
// own XCD's scheduler, not the global counter over the fabric. This measures
// the L2 hop alone, with producer and consumer pinned to the same XCD.

__global__ __launch_bounds__(kBlock) void xcd_local_flag(
    unsigned* __restrict__ flag, unsigned long long* __restrict__ out_cycles,
    int iters) {
  const int wg = blockIdx.x;
  // Workgroups are dispatched round-robin over XCDs, so wg and wg+kXCDs land
  // on the same chiplet.
  const bool is_writer = (wg == 0);
  const bool is_reader = (wg == kXCDs);
  if (!is_writer && !is_reader) return;

  __shared__ unsigned long long t0;
  for (int i = 1; i <= iters; ++i) {
    if (is_writer) {
      if (threadIdx.x == 0) {
        t0 = clk();
        __threadfence_block();
        atomicAdd(flag, 1u);            // stays in this XCD's L2
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        while (nt_load(flag + 1) < (unsigned)i) { __builtin_amdgcn_s_sleep(1); }
        out_cycles[i - 1] = clk() - t0;
      }
      __syncthreads();
    } else {
      if (threadIdx.x == 0) {
        while (*(volatile unsigned*)flag < (unsigned)i) {
          __builtin_amdgcn_s_sleep(1);
        }
        atomicAdd(flag + 1, 1u);
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

static Stats run_event(int scheme, int iters, double ns) {
  unsigned *ping, *pong;
  unsigned long long* cyc;
  HIP_CHECK(hipMalloc(&cyc, iters * sizeof(unsigned long long)));
  if (scheme == SCHEME_WRITETHROUGH) {
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
                     cyc, iters, scheme);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<unsigned long long> h(iters);
  HIP_CHECK(hipMemcpy(h.data(), cyc, iters * sizeof(h[0]), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(ping)); HIP_CHECK(hipFree(pong)); HIP_CHECK(hipFree(cyc));
  return summarise(std::move(h), ns);
}

static Stats run_xcd_local(int iters, double ns) {
  unsigned* flag;
  unsigned long long* cyc;
  HIP_CHECK(hipMalloc(&flag, 64));
  HIP_CHECK(hipMemset(flag, 0, 64));
  HIP_CHECK(hipMalloc(&cyc, iters * sizeof(unsigned long long)));
  hipLaunchKernelGGL(xcd_local_flag, dim3(kGrid), dim3(kBlock), 0, 0, flag, cyc, iters);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<unsigned long long> h(iters);
  HIP_CHECK(hipMemcpy(h.data(), cyc, iters * sizeof(h[0]), hipMemcpyDeviceToHost));
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
  std::printf("(a) cross-XCD event round-trip (ping-pong, us)\n");
  Stats wbl2 = run_event(SCHEME_WBL2, iters, ns);
  Stats wt = run_event(SCHEME_WRITETHROUGH, iters, ns);
  std::printf("    (i)  wbl2 + atomicAdd        median %6.2f  p95 %6.2f\n",
              wbl2.median / 1e3, wbl2.p95 / 1e3);
  std::printf("    (ii) write-through/uncached  median %6.2f  p95 %6.2f\n",
              wt.median / 1e3, wt.p95 / 1e3);
  std::printf("    -> design picks the cheaper: %s\n\n",
              wt.median < wbl2.median ? "(ii)" : "(i)");

  std::printf("(b) XCD-local flag round-trip (us)\n");
  Stats loc = run_xcd_local(iters, ns);
  std::printf("    median %6.3f  p95 %6.3f   (vs global %.2f)\n\n",
              loc.median / 1e3, loc.p95 / 1e3,
              std::min(wbl2.median, wt.median) / 1e3);

  std::printf("(c) register-streamed read bandwidth by in-flight depth\n");
  const size_t bytes = 2ull << 30;  // 2 GiB, larger than the 256 MB LLC
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
        "  \"event_wbl2_us\": {\"median\": %.3f, \"p95\": %.3f},\n"
        "  \"event_writethrough_us\": {\"median\": %.3f, \"p95\": %.3f},\n"
        "  \"xcd_local_us\": {\"median\": %.4f, \"p95\": %.4f},\n"
        "  \"gemv_tbs\": [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f],\n"
        "  \"gemv_best_tbs\": %.3f\n}\n",
        prop.name, prop.gcnArchName, prop.multiProcessorCount,
        wbl2.median / 1e3, wbl2.p95 / 1e3, wt.median / 1e3, wt.p95 / 1e3,
        loc.median / 1e3, loc.p95 / 1e3,
        bw[0], bw[1], bw[2], bw[3], bw[4], bw[5], best);
    std::fclose(f);
    std::printf("\nwrote %s\n", json_path);
  }
  return 0;
}
