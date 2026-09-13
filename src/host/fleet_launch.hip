// Host launcher: build the per-worker queues, allocate device state, run the
// persistent kernel once per token.
//
// docs/design.md §4: descriptors are built on the host, immutable, and grouped
// by (xcd, worker) before launch. A worker therefore never consults a scheduler
// to find its next task — it walks its own slice of `queue_index`. That is the
// "one synchronisation hop instead of two" the design trades for.
//
// §4 also requires all 304 workgroups to be co-resident: a scheduler waiting on
// a workgroup that was never dispatched deadlocks. hipLaunchCooperativeKernel
// fails the launch instead of hanging, which turns a subtle runtime hang into a
// loud startup error.
//
// Build: hipcc --offload-arch=gfx942 -O3 -std=c++17 src/host/fleet_launch.hip \
//              src/kernels/fleet_kernel.hip -o build/fleet_decode

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include "../runtime/fleet_runtime.h"
#include "../runtime/fleet_types.h"

#define HIP_CHECK(expr)                                                        \
    do {                                                                       \
        hipError_t _e = (expr);                                                \
        if (_e != hipSuccess) {                                                \
            std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__,       \
                         #expr, hipGetErrorString(_e));                        \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

using namespace fleet;

// Kernel entry point; parameter types come from fleet_types.h so the
// host and device agree on layout by construction.
extern "C" __global__ void fleet_decode_step(RuntimeState, ModelDims, Weights,
                                             Activations, KVCache, int,
                                             uint32_t*);

// ---------------------------------------------------------------- queues

// Group descriptors by (xcd, worker) into a CSR-style index. Order within a
// worker's slice is preserved from the host graph, which is what makes the
// ahead-of-time wait ordering valid (taskgraph.py validates that no task waits
// on an event produced later in the queue).
struct Queues {
    std::vector<int32_t> offset;   // [kXCDs * kWorkersPerXCD + 1]
    std::vector<int32_t> index;    // task ids, grouped
};

static Queues build_queues(const std::vector<TaskDescriptor>& tasks) {
    const int nq = kXCDs * kWorkersPerXCD;
    Queues q;
    q.offset.assign(nq + 1, 0);

    for (const auto& t : tasks) {
        if (t.xcd < 0 || t.worker < 0) continue;
        q.offset[t.xcd * kWorkersPerXCD + t.worker + 1]++;
    }
    for (int i = 0; i < nq; ++i) q.offset[i + 1] += q.offset[i];

    q.index.resize(q.offset[nq]);
    std::vector<int32_t> cursor(q.offset.begin(), q.offset.end() - 1);
    for (size_t i = 0; i < tasks.size(); ++i) {
        const auto& t = tasks[i];
        if (t.xcd < 0 || t.worker < 0) continue;
        q.index[cursor[t.xcd * kWorkersPerXCD + t.worker]++] = (int32_t)i;
    }
    return q;
}

static std::vector<TaskDescriptor> load_descriptors(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (bytes % (long)sizeof(TaskDescriptor)) {
        std::fprintf(stderr, "%s is %ld bytes, not a multiple of %zu — host and "
                     "device descriptor layouts disagree\n",
                     path, bytes, sizeof(TaskDescriptor));
        std::exit(1);
    }
    std::vector<TaskDescriptor> v(bytes / sizeof(TaskDescriptor));
    if (std::fread(v.data(), 1, bytes, f) != (size_t)bytes) { std::exit(1); }
    std::fclose(f);
    return v;
}

// ---------------------------------------------------------------- YaRN tables
//
// cos/sin depend only on position, never on data, so they are computed once on
// the host and uploaded. Recomputing YaRN on device would mean a second
// implementation of the trickiest formula in the model — and a second thing
// that can silently disagree with HuggingFace.

static double yarn_get_mscale(double scale, double mscale) {
    return scale <= 1.0 ? 1.0 : 0.1 * mscale * std::log(scale) + 1.0;
}

static void yarn_tables(int dim, int pos, double base, double scale,
                        int original_max, double beta_fast, double beta_slow,
                        double mscale, double mscale_all_dim,
                        std::vector<float>& cos_out,
                        std::vector<float>& sin_out) {
    const int half = dim / 2;
    std::vector<double> inv_freq(half);

    auto corr_dim = [&](double rot) {
        return dim * std::log(original_max / (rot * 2 * M_PI)) / (2 * std::log(base));
    };
    double low = std::floor(corr_dim(beta_fast));
    double high = std::ceil(corr_dim(beta_slow));
    low = low < 0 ? 0 : low;
    high = high > dim - 1 ? dim - 1 : high;
    if (low == high) high += 0.001;

    for (int i = 0; i < half; ++i) {
        const double e = (double)(2 * i) / dim;
        const double extra = 1.0 / std::pow(base, e);
        const double inter = 1.0 / (scale * std::pow(base, e));
        double ramp = ((double)i - low) / (high - low);
        ramp = ramp < 0 ? 0 : (ramp > 1 ? 1 : ramp);
        const double mask = 1.0 - ramp;
        inv_freq[i] = inter * (1 - mask) + extra * mask;
    }

    const double m = yarn_get_mscale(scale, mscale) /
                     yarn_get_mscale(scale, mscale_all_dim);
    cos_out.resize(dim);
    sin_out.resize(dim);
    for (int i = 0; i < half; ++i) {
        const double a = pos * inv_freq[i];
        cos_out[i] = cos_out[i + half] = (float)(std::cos(a) * m);
        sin_out[i] = sin_out[i + half] = (float)(std::sin(a) * m);
    }
}

// ---------------------------------------------------------------- main

int main(int argc, char** argv) {
    const char* graph_path = "build/taskgraph_d2.bin";
    int tokens = 32, context = 1024;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--graph") && i + 1 < argc) graph_path = argv[++i];
        else if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) tokens = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--context") && i + 1 < argc) context = atoi(argv[++i]);
    }

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("device: %s  arch %s  CUs %d\n", prop.name, prop.gcnArchName,
                prop.multiProcessorCount);
    if (prop.multiProcessorCount != kGrid) {
        std::printf("  note: grid is %d (8 XCDs x 38 CUs); device reports %d CUs\n",
                    kGrid, prop.multiProcessorCount);
    }

    auto tasks = load_descriptors(graph_path);
    auto q = build_queues(tasks);
    std::printf("graph: %zu descriptors, %zu queued across %d workers\n",
                tasks.size(), q.index.size(), kXCDs * kWorkersPerXCD);

    int max_event = 0;
    for (const auto& t : tasks) {
        if (t.signal_event > max_event) max_event = t.signal_event;
    }
    const int n_events = max_event + 1;
    std::printf("events: %d\n", n_events);
    if (n_events > kMaxEvents) {
        std::fprintf(stderr, "n_events %d exceeds kMaxEvents %d\n", n_events, kMaxEvents);
        return 1;
    }

    // ---- device state
    RuntimeState rt{};
    TaskDescriptor* d_tasks;
    int32_t *d_off, *d_idx;
    HIP_CHECK(hipMalloc(&d_tasks, tasks.size() * sizeof(TaskDescriptor)));
    HIP_CHECK(hipMalloc(&d_off, q.offset.size() * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&d_idx, q.index.size() * sizeof(int32_t)));
    HIP_CHECK(hipMemcpy(d_tasks, tasks.data(),
                        tasks.size() * sizeof(TaskDescriptor), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_off, q.offset.data(),
                        q.offset.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_idx, q.index.data(),
                        q.index.size() * sizeof(int32_t), hipMemcpyHostToDevice));

    // Event counters: scheme (ii) from §4 wants them in uncached memory so the
    // atomic resolves at the Infinity Cache without an L2 flush. D1
    // microbenchmark (a) decides; until it has run, prefer the portable path.
    const bool uncached = std::getenv("FLEET_UNCACHED_EVENTS") != nullptr;
    uint32_t *d_global, *d_flags, *d_counters, *d_done;
    const size_t ev_bytes = (size_t)n_events * sizeof(uint32_t);
    if (uncached) {
        HIP_CHECK(hipExtMallocWithFlags((void**)&d_global, ev_bytes,
                                        hipDeviceMallocUncached));
    } else {
        HIP_CHECK(hipMalloc(&d_global, ev_bytes));
    }
    HIP_CHECK(hipMalloc(&d_flags, (size_t)kXCDs * kMaxEvents * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&d_counters, (size_t)kXCDs * kMaxEvents * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&d_done, sizeof(uint32_t)));
    HIP_CHECK(hipMemset(d_global, 0, ev_bytes));
    HIP_CHECK(hipMemset(d_flags, 0, (size_t)kXCDs * kMaxEvents * sizeof(uint32_t)));
    HIP_CHECK(hipMemset(d_counters, 0, (size_t)kXCDs * kMaxEvents * sizeof(uint32_t)));

    rt.tasks = d_tasks;
    rt.queue_offset = d_off;
    rt.queue_index = d_idx;
    rt.global_events = d_global;
    rt.xcd_flags = d_flags;
    rt.xcd_counters = d_counters;
    rt.n_events = n_events;
    rt.use_uncached_counters = uncached ? 1 : 0;

    std::printf("event counters: %s\n",
                uncached ? "uncached (scheme ii)" : "cached + wbl2 (scheme i)");

    // ---- residency check before anything expensive happens
    void* args[] = {&rt};
    int blocks_per_cu = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_cu, (const void*)fleet_decode_step, kBlock, 0));
    std::printf("occupancy: %d workgroup(s)/CU at %d threads "
                "(need >= 1 for grid %d)\n", blocks_per_cu, kBlock, kGrid);
    if (blocks_per_cu < 1) {
        std::fprintf(stderr, "cooperative launch would fail: the kernel does not "
                     "fit one workgroup per CU. Check register pressure with "
                     "-Rpass-analysis=kernel-resource-usage.\n");
        return 1;
    }

    std::printf("\nready: %d tokens over %d-token context\n", tokens, context);
    std::printf("weights, activations and KV cache are bound by "
                "src/host/pack_weights.py output; wire that in before launching.\n");

    HIP_CHECK(hipFree(d_tasks)); HIP_CHECK(hipFree(d_off)); HIP_CHECK(hipFree(d_idx));
    HIP_CHECK(hipFree(d_global)); HIP_CHECK(hipFree(d_flags));
    HIP_CHECK(hipFree(d_counters)); HIP_CHECK(hipFree(d_done));
    return 0;
}
