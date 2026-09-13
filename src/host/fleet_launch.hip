// Host launcher: build the per-worker queues, allocate device state, bind the
// packed weights and the converted cache, and run the persistent kernel once
// per token.
//
// docs/design.md §4: descriptors are built on the host, immutable, and grouped
// by (xcd, worker) before launch. A worker therefore never consults a scheduler
// to find its next task — it walks its own slice of `queue_index`. That is the
// "one synchronisation hop instead of two" the design trades for.
//
// §4 also requires all 304 workgroups to be co-resident: a scheduler waiting on
// a workgroup that was never dispatched deadlocks. hipLaunchCooperativeKernel
// fails the launch instead of hanging, which turns a subtle runtime hang into a
// loud startup error. The kernel adds its own checks on top (an XCD with the
// wrong number of blocks, a wait that never completes) and reports them
// through an abort code this launcher decodes.
//
// Inputs (all produced by the scripts in src/host/):
//   --graph    build/taskgraph_d2.bin      taskgraph.py --emit
//   --weights  build/weights.bin           pack_weights.py (+ .manifest)
//   --cache    build/fleet_cache.bin       reference_run.py (bf16 [L][max_pos][576])
//   --golden   build/golden_tokens.txt     reference_run.py (HF greedy tokens)
//   --hidden   build/golden_hidden.bin     reference_run.py (fp32 [layers][hidden],
//              each layer's output on the first decode step; compared layer by
//              layer against the kernel's own outputs on token 0, §6)
//
// Modes:
//   --smoke    run the protocol only: every task body is skipped, so this
//              needs no weights and measures pure synchronisation cost
//   default    free-running greedy decode, compared token by token with HF
//   --teacher-force
//              feed HF's token at every step so a divergence is isolated to
//              the step it appears in
//
// Build: hipcc --offload-arch=gfx942 -O3 -std=c++17 -Isrc src/host/fleet_launch.hip \
//              src/kernels/fleet_kernel.hip -o build/fleet_decode

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
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

extern "C" hipError_t fleet_occupancy(int* blocks_per_cu);
extern "C" hipError_t fleet_probe_xcds(uint32_t* d_hist, hipStream_t stream);
extern "C" hipError_t fleet_launch_step(const RuntimeState*, const ModelDims*,
                                        const Weights*, const Activations*,
                                        const KVCache*, int token, hipStream_t);

[[noreturn]] static void die(const std::string& msg) {
    std::fprintf(stderr, "error: %s\n", msg.c_str());
    std::exit(1);
}

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
        if (t.xcd < 0 || t.xcd >= kXCDs || t.worker < 0 || t.worker >= kWorkersPerXCD) {
            die("descriptor " + std::to_string(t.index) + " has (xcd, worker) = (" +
                std::to_string(t.xcd) + ", " + std::to_string(t.worker) + ")");
        }
        q.offset[t.xcd * kWorkersPerXCD + t.worker + 1]++;
    }
    for (int i = 0; i < nq; ++i) q.offset[i + 1] += q.offset[i];
    q.index.resize(q.offset[nq]);
    std::vector<int32_t> cursor(q.offset.begin(), q.offset.end() - 1);
    for (size_t i = 0; i < tasks.size(); ++i) {
        const auto& t = tasks[i];
        q.index[cursor[t.xcd * kWorkersPerXCD + t.worker]++] = (int32_t)i;
    }
    return q;
}

static std::vector<TaskDescriptor> load_descriptors(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) die("cannot open " + path);
    const long bytes = (long)f.tellg();
    f.seekg(0);
    if (bytes <= 0 || bytes % (long)sizeof(TaskDescriptor)) {
        die(path + " is " + std::to_string(bytes) + " bytes, not a multiple of " +
            std::to_string(sizeof(TaskDescriptor)) +
            " — host and device descriptor layouts disagree");
    }
    std::vector<TaskDescriptor> v(bytes / sizeof(TaskDescriptor));
    f.read((char*)v.data(), bytes);
    return v;
}

static std::map<int, std::string> load_event_labels(const std::string& graph_path) {
    std::map<int, std::string> labels;
    std::ifstream f(graph_path + ".events.txt");
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        int id, scope, producers;
        std::string label;
        if (ss >> id >> scope >> producers >> label) labels[id] = label;
    }
    return labels;
}

// ---------------------------------------------------------------- manifest

struct Manifest {
    struct Tensor { size_t offset, bytes; };
    std::map<std::string, double> scalars;
    std::map<std::string, Tensor> tensors;

    double scalar(const std::string& k) const {
        auto it = scalars.find(k);
        if (it == scalars.end()) die("manifest has no scalar '" + k + "'");
        return it->second;
    }
    bool has(const std::string& k) const { return tensors.count(k) != 0; }
    const Tensor& tensor(const std::string& k) const {
        auto it = tensors.find(k);
        if (it == tensors.end()) die("manifest has no tensor '" + k + "'");
        return it->second;
    }
};

static Manifest load_manifest(const std::string& path) {
    std::ifstream f(path);
    if (!f) die("cannot open " + path + " (run src/host/pack_weights.py)");
    Manifest m;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string kind, name;
        ss >> kind >> name;
        if (kind == "scalar") {
            double v; ss >> v; m.scalars[name] = v;
        } else if (kind == "tensor") {
            size_t off, bytes; ss >> off >> bytes; m.tensors[name] = {off, bytes};
        }
    }
    return m;
}

// ---------------------------------------------------------------- YaRN tables
//
// cos/sin depend only on position, never on data, so they are computed once on
// the host and uploaded. Recomputing YaRN on device would mean a second
// implementation of the trickiest formula in the model — and a second thing
// that can silently disagree with HuggingFace. This mirrors
// reference_decode.py's yarn_cos_sin line for line.

struct RopeParams {
    double base, factor, beta_fast, beta_slow, mscale, mscale_all_dim;
    int dim, original_max;
};

static double yarn_get_mscale(double scale, double mscale) {
    return scale <= 1.0 ? 1.0 : 0.1 * mscale * std::log(scale) + 1.0;
}

static void yarn_tables(const RopeParams& r, int max_pos,
                        std::vector<float>& cos_out, std::vector<float>& sin_out) {
    const int dim = r.dim, half = dim / 2;
    auto corr_dim = [&](double rot) {
        return dim * std::log(r.original_max / (rot * 2 * M_PI)) / (2 * std::log(r.base));
    };
    double low = std::floor(corr_dim(r.beta_fast));
    double high = std::ceil(corr_dim(r.beta_slow));
    low = std::max(low, 0.0);
    high = std::min(high, (double)(dim - 1));
    if (low == high) high += 0.001;

    std::vector<double> inv_freq(half);
    for (int i = 0; i < half; ++i) {
        const double e = (double)(2 * i) / dim;
        const double extra = 1.0 / std::pow(r.base, e);
        const double inter = 1.0 / (r.factor * std::pow(r.base, e));
        double ramp = ((double)i - low) / (high - low);
        ramp = std::min(std::max(ramp, 0.0), 1.0);
        const double mask = 1.0 - ramp;
        inv_freq[i] = inter * (1 - mask) + extra * mask;
    }
    const double m = yarn_get_mscale(r.factor, r.mscale) /
                     yarn_get_mscale(r.factor, r.mscale_all_dim);
    cos_out.assign((size_t)max_pos * dim, 0.f);
    sin_out.assign((size_t)max_pos * dim, 0.f);
    for (int pos = 0; pos < max_pos; ++pos) {
        for (int i = 0; i < half; ++i) {
            const double a = pos * inv_freq[i];
            cos_out[(size_t)pos * dim + i] = cos_out[(size_t)pos * dim + i + half] = (float)(std::cos(a) * m);
            sin_out[(size_t)pos * dim + i] = sin_out[(size_t)pos * dim + i + half] = (float)(std::sin(a) * m);
        }
    }
}

// ---------------------------------------------------------------- helpers

template <typename T>
static T* dev_alloc(size_t n) {
    T* p = nullptr;
    HIP_CHECK(hipMalloc(&p, std::max<size_t>(1, n) * sizeof(T)));
    HIP_CHECK(hipMemset(p, 0, std::max<size_t>(1, n) * sizeof(T)));
    return p;
}

template <typename T>
static T* dev_upload(const std::vector<T>& v) {
    T* p = dev_alloc<T>(v.size());
    if (!v.empty()) HIP_CHECK(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    return p;
}

// Stream a large file straight into device memory, 256 MB at a time through
// pinned host memory. 31 GB at NVMe speed is the slow part of startup.
static void upload_file(const std::string& path, void* dst, size_t expect_bytes) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) die("cannot open " + path);
    const size_t bytes = (size_t)f.tellg();
    if (bytes != expect_bytes) {
        die(path + " is " + std::to_string(bytes) + " bytes, expected " +
            std::to_string(expect_bytes));
    }
    f.seekg(0);
    const size_t chunk = 256ull << 20;
    void* pinned = nullptr;
    HIP_CHECK(hipHostMalloc(&pinned, chunk));
    for (size_t off = 0; off < bytes; off += chunk) {
        const size_t n = std::min(chunk, bytes - off);
        f.read((char*)pinned, n);
        HIP_CHECK(hipMemcpy((char*)dst + off, pinned, n, hipMemcpyHostToDevice));
        std::printf("\r  %s: %.1f / %.1f GB", path.c_str(), (off + n) / 1e9, bytes / 1e9);
        std::fflush(stdout);
    }
    std::printf("\n");
    HIP_CHECK(hipHostFree(pinned));
}

static const __hip_bfloat16* tensor_ptr(const __hip_bfloat16* blob, const Manifest& m,
                                        const std::string& name) {
    if (!m.has(name)) return nullptr;
    return blob + m.tensor(name).offset / sizeof(__hip_bfloat16);
}

static const __hip_bfloat16* const* layer_table(const __hip_bfloat16* blob,
                                                const Manifest& m, int layers,
                                                const std::string& suffix) {
    std::vector<const __hip_bfloat16*> h(layers);
    for (int i = 0; i < layers; ++i) h[i] = tensor_ptr(blob, m, "L" + std::to_string(i) + "." + suffix);
    return dev_upload(h);
}

static std::vector<int> load_tokens(const std::string& path) {
    std::vector<int> v;
    std::ifstream f(path);
    int t;
    while (f >> t) v.push_back(t);
    return v;
}

static void describe_abort(uint32_t code, const std::map<int, std::string>& labels) {
    if (code == kAbortNone) return;
    const uint32_t kind = code & 0xF0000000u;
    if (kind == kAbortXcdDistribution) {
        std::fprintf(stderr, "abort: XCD %u received %u workgroups, expected %d — "
                     "the round-robin placement assumption in design.md §4 does "
                     "not hold on this device/partition mode\n",
                     (code >> 8) & 0xff, code & 0xff, kCUsPerXCD);
    } else if (kind == kAbortWaitTimeout) {
        const int ev = (int)(code & 0xffff);
        auto it = labels.find(ev);
        std::fprintf(stderr, "abort: a worker timed out waiting on event %d (%s) — "
                     "its producers never all signalled; check the graph's producer "
                     "counts and the XCD-locality of that event\n", ev,
                     it == labels.end() ? "unlabelled" : it->second.c_str());
    } else if (kind == kAbortBarrierTimeout) {
        std::fprintf(stderr, "abort: the grid barrier timed out — fewer than %d "
                     "workgroups became resident\n", kGrid);
    } else {
        std::fprintf(stderr, "abort: unknown code 0x%08x\n", code);
    }
}

// ---------------------------------------------------------------- main

int main(int argc, char** argv) {
    std::string graph_path = "build/taskgraph_d2.bin";
    std::string weights_path = "build/weights.bin";
    std::string cache_path = "build/fleet_cache.bin";
    std::string golden_path = "build/golden_tokens.txt";
    std::string hidden_path = "build/golden_hidden.bin";
    std::string json_path, dump_path, trace_path;
    int tokens = 32, context = 1024, first_token = -1;
    bool smoke = false, teacher = false;
    uint32_t spin_limit = 1u << 22;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) die("missing value after " + a); return argv[++i]; };
        if (a == "--graph") graph_path = next();
        else if (a == "--weights") weights_path = next();
        else if (a == "--cache") cache_path = next();
        else if (a == "--golden") golden_path = next();
        else if (a == "--hidden") hidden_path = next();
        else if (a == "--dump-layers") dump_path = next();   // write our [layers][hidden] fp32
        else if (a == "--trace") trace_path = next();        // per-task timeline of token 1
        else if (a == "--json") json_path = next();
        else if (a == "--tokens") tokens = std::atoi(next().c_str());
        else if (a == "--context") context = std::atoi(next().c_str());
        else if (a == "--first-token") first_token = std::atoi(next().c_str());
        else if (a == "--spin-limit") spin_limit = (uint32_t)std::atol(next().c_str());
        else if (a == "--smoke") smoke = true;
        else if (a == "--teacher-force") teacher = true;
        else die("unknown argument " + a);
    }

    // ---- device
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("device: %s  arch %s  CUs %d\n", prop.name, prop.gcnArchName,
                prop.multiProcessorCount);
    int coop = 0;
    HIP_CHECK(hipDeviceGetAttribute(&coop, hipDeviceAttributeCooperativeLaunch, 0));
    if (!coop) die("device does not support cooperative launch");
    if (prop.multiProcessorCount != kGrid) {
        std::printf("  note: grid is %d (8 XCDs x 38 CUs); device reports %d CUs\n",
                    kGrid, prop.multiProcessorCount);
    }

    // ---- graph
    auto tasks = load_descriptors(graph_path);
    auto q = build_queues(tasks);
    auto labels = load_event_labels(graph_path);
    int max_event = -1, kv_chunks = 1;
    std::vector<char> is_global;
    for (const auto& t : tasks) {
        max_event = std::max(max_event, (int)t.signal_event);
        if (t.kind == TASK_ATTENTION) kv_chunks = std::max(kv_chunks, (int)t.n_split);
        if (t.signal_event >= 0 && t.signal_scope == SCOPE_GLOBAL) {
            if ((int)is_global.size() <= t.signal_event) is_global.resize(t.signal_event + 1, 0);
            is_global[t.signal_event] = 1;
        }
    }
    const int n_events = max_event + 1;
    std::vector<int32_t> global_ids;   // what the schedulers mirror; nothing else
    for (int e = 0; e < (int)is_global.size(); ++e) if (is_global[e]) global_ids.push_back(e);
    const int done_event = tasks.back().signal_event;
    if (n_events > kMaxEvents) die("n_events " + std::to_string(n_events) + " exceeds kMaxEvents");
    if (tasks.back().kind != TASK_ARGMAX) die("last descriptor is not ARGMAX; done_event would be wrong");
    if (done_event < 0 || done_event >= (int)is_global.size() || !is_global[done_event]) {
        die("final event " + std::to_string(done_event) + " is not a global-scope event; "
            "the schedulers would never see it and never stop");
    }
    std::printf("graph: %zu descriptors across %d workers, %d events (%zu global, mirrored), "
                "kv_chunks %d, final event %d (%s)\n", tasks.size(), kXCDs * kWorkersPerXCD,
                n_events, global_ids.size(), kv_chunks, done_event,
                labels.count(done_event) ? labels[done_event].c_str() : "?");

    // ---- runtime state
    const bool uncached = std::getenv("FLEET_UNCACHED_EVENTS") != nullptr;
    RuntimeState rt{};
    rt.tasks = dev_upload(tasks);
    rt.queue_offset = dev_upload(q.offset);
    rt.queue_index = dev_upload(q.index);
    if (uncached) {
        // Scheme (ii) from §4: counters in uncached memory so the atomic
        // resolves at the Infinity Cache. D1 microbenchmark (a) decides.
        uint32_t* p = nullptr;
        HIP_CHECK(hipExtMallocWithFlags((void**)&p, (size_t)kMaxEvents * sizeof(uint32_t),
                                        hipDeviceMallocUncached));
        HIP_CHECK(hipMemset(p, 0, (size_t)kMaxEvents * sizeof(uint32_t)));
        rt.global_events = p;
    } else {
        rt.global_events = dev_alloc<uint32_t>(kMaxEvents);
    }
    rt.xcd_flags = dev_alloc<uint32_t>((size_t)kXCDs * kMaxEvents);
    rt.xcd_counters = dev_alloc<uint32_t>((size_t)kXCDs * kMaxEvents);
    uint32_t* misc = dev_alloc<uint32_t>(16);   // arrivals + abort, reset per launch
    rt.xcd_arrivals = misc;
    rt.grid_arrivals = misc + kXCDs;
    rt.abort = misc + kXCDs + 1;
    rt.global_event_ids = dev_upload(global_ids);
    rt.n_global_events = (int32_t)global_ids.size();
    rt.n_events = n_events;
    rt.done_event = done_event;
    rt.use_uncached_counters = uncached ? 1 : 0;
    rt.smoke = smoke ? 1 : 0;
    rt.dump_layers = 0;   // set for token 0 below
    rt.spin_limit = spin_limit;
    uint64_t* d_trace = trace_path.empty() ? nullptr : dev_alloc<uint64_t>(tasks.size() * 4);
    rt.trace = nullptr;   // enabled for token 1 only (token 0 carries the warm-up)
    rt.epoch = 0;   // set per token below; a zero epoch makes every wait trivially true
    std::printf("event counters: %s\n",
                uncached ? "uncached (scheme ii)" : "cached + release fence (scheme i)");

    // ---- model
    ModelDims d{};
    Weights w{};
    Activations act{};
    KVCache kv{};
    std::vector<int> golden;
    const int max_pos = context + tokens;
    float *d_cos = nullptr, *d_sin = nullptr;

    if (!smoke) {
        const Manifest m = load_manifest(weights_path + ".manifest");
        d.hidden = (int)m.scalar("hidden");
        d.heads = (int)m.scalar("heads");
        d.qk_nope = (int)m.scalar("qk_nope");
        d.qk_rope = (int)m.scalar("qk_rope");
        d.q_head_dim = d.qk_nope + d.qk_rope;
        d.v_head = (int)m.scalar("v_head");
        d.kv_lora = (int)m.scalar("kv_lora");
        d.moe_inter = (int)m.scalar("moe_inter");
        d.dense_inter = (int)m.scalar("dense_inter");
        d.n_routed = (int)m.scalar("n_routed");
        d.top_k = (int)m.scalar("top_k");
        d.vocab = (int)m.scalar("vocab");
        d.layers = (int)m.scalar("layers");
        d.first_k_dense = (int)m.scalar("first_k_dense");
        d.rms_eps = (float)m.scalar("rms_eps");
        d.routed_scaling = (float)m.scalar("routed_scaling");

        RopeParams rp{m.scalar("rope_theta"), m.scalar("rope_factor"),
                      m.scalar("beta_fast"), m.scalar("beta_slow"),
                      m.scalar("mscale"), m.scalar("mscale_all_dim"),
                      d.qk_rope, (int)m.scalar("rope_original_max")};
        {
            // reference_decode.py ModelConfig.softmax_scale
            double s = std::pow((double)d.q_head_dim, -0.5);
            if (rp.mscale_all_dim != 0.0) {
                const double ms = yarn_get_mscale(rp.factor, rp.mscale_all_dim);
                s = s * ms * ms;
            }
            d.softmax_scale = (float)s;
        }

        // The kernel's LDS layout and attention lane mapping are sized for
        // this model; refuse anything else rather than overflow silently.
        if (d.hidden > kMaxHidden || d.kv_lora != kMaxKvLora || d.qk_rope != kMaxQkRope ||
            d.moe_inter > kMaxMoeInter || d.n_routed > kMaxRouted || d.top_k > kMaxTopK ||
            d.heads % kXCDs != 0 || (d.heads * d.q_head_dim) % kXCDs != 0 ||
            (int)m.scalar("n_shared") * d.moe_inter != 2 * d.moe_inter) {
            die("model dims exceed the kernel's compile-time bounds (fleet_types.h)");
        }
        std::printf("model: hidden %d heads %d kv_lora %d rope %d experts %d/%d top-%d "
                    "layers %d (dense < %d) vocab %d  softmax_scale %.6f\n",
                    d.hidden, d.heads, d.kv_lora, d.qk_rope, d.n_routed, 2, d.top_k,
                    d.layers, d.first_k_dense, d.vocab, d.softmax_scale);

        // weights: one flat blob, per-layer pointer tables
        const size_t blob_bytes = (size_t)m.scalar("bytes");
        __hip_bfloat16* blob = nullptr;
        HIP_CHECK(hipMalloc(&blob, blob_bytes));
        upload_file(weights_path, blob, blob_bytes);
        w.layer_qkv = layer_table(blob, m, d.layers, "qkv");
        w.layer_kv_b = layer_table(blob, m, d.layers, "kv_b");
        w.layer_o_proj = layer_table(blob, m, d.layers, "o_proj");
        w.layer_norm_in = layer_table(blob, m, d.layers, "norm_in");
        w.layer_norm_post = layer_table(blob, m, d.layers, "norm_post");
        w.layer_kv_a_norm = layer_table(blob, m, d.layers, "kv_a_norm");
        w.layer_router = layer_table(blob, m, d.layers, "router");
        w.layer_experts = layer_table(blob, m, d.layers, "experts");
        w.layer_shared = layer_table(blob, m, d.layers, "shared");
        w.layer_dense = layer_table(blob, m, d.layers, "dense");
        w.embed = tensor_ptr(blob, m, "embed");
        w.lm_head = tensor_ptr(blob, m, "lm_head");
        w.final_norm = tensor_ptr(blob, m, "final_norm");
        w.expert_stride = (int64_t)m.scalar("expert_stride");
        if (!w.embed || !w.lm_head || !w.final_norm) die("manifest is missing embed/lm_head/final_norm");
        for (int L = 0; L < d.layers; ++L) {
            const std::string p = "L" + std::to_string(L) + ".";
            for (const char* s : {"qkv", "kv_b", "o_proj", "norm_in", "norm_post", "kv_a_norm"}) {
                if (!m.has(p + s)) die("manifest is missing " + p + s);
            }
            const bool moe = L >= d.first_k_dense;
            if (moe && !(m.has(p + "router") && m.has(p + "experts") && m.has(p + "shared"))) die("manifest is missing MoE tensors for " + p);
            if (!moe && !m.has(p + "dense")) die("manifest is missing " + p + "dense");
        }

        // activations (§5)
        act.x = dev_alloc<float>(d.hidden);
        act.x_norm = dev_alloc<float>(d.hidden);
        act.q = dev_alloc<float>((size_t)d.heads * d.q_head_dim + d.kv_lora + d.qk_rope);
        act.kv_a = act.q + (size_t)d.heads * d.q_head_dim;
        act.attn_partial = dev_alloc<float>((size_t)d.heads * kv_chunks * kPartialStride);
        act.o = dev_alloc<float>((size_t)d.heads * d.v_head);
        act.expert_h = dev_alloc<float>(std::max((size_t)kXCDs * d.moe_inter, (size_t)d.dense_inter));
        act.expert_out = dev_alloc<float>((size_t)kXCDs * d.hidden);
        act.logits = dev_alloc<float>(d.vocab);
        act.topk_ids = dev_alloc<int32_t>(kMaxTopK);
        act.topk_w = dev_alloc<float>(kMaxTopK);
        act.next_token = dev_alloc<int32_t>(1);
        act.layer_dump = dev_alloc<float>((size_t)d.layers * d.hidden);

        std::vector<float> cos_h, sin_h;
        yarn_tables(rp, max_pos, cos_h, sin_h);
        d_cos = dev_upload(cos_h);
        d_sin = dev_upload(sin_h);

        // KV cache: [layers][max_pos][576] bf16, prefill rows filled by
        // reference_run.py, decode rows zero until the kernel appends them.
        // max_pos comes from the file (reference_run wrote context + decode
        // rows); this run may decode fewer tokens than that, never more.
        kv.row = d.kv_lora + d.qk_rope;
        {
            std::ifstream f(cache_path, std::ios::binary | std::ios::ate);
            if (!f) die("cannot open " + cache_path + " (reference_run.py writes it)");
            const size_t bytes = (size_t)f.tellg();
            const size_t per_row = (size_t)d.layers * kv.row * sizeof(__hip_bfloat16);
            if (bytes % per_row) die(cache_path + " size is not a whole number of positions");
            kv.max_pos = (int)(bytes / per_row);
        }
        if (max_pos > kv.max_pos) {
            die("cache holds " + std::to_string(kv.max_pos) + " positions but context + tokens = " +
                std::to_string(max_pos) + "; re-run reference_run.py with a larger --decode");
        }
        const size_t cache_bytes = (size_t)d.layers * kv.max_pos * kv.row * sizeof(__hip_bfloat16);
        __hip_bfloat16* cache_dev = nullptr;
        HIP_CHECK(hipMalloc(&cache_dev, cache_bytes));
        upload_file(cache_path, cache_dev, cache_bytes);
        kv.data = cache_dev;

        golden = load_tokens(golden_path);
        if (golden.empty() && first_token < 0) die("no golden tokens at " + golden_path + " and no --first-token");
        if (first_token < 0) first_token = golden[0];
    }

    // Golden per-layer states (optional): compared after token 0.
    std::vector<float> golden_hidden;
    if (!smoke) {
        std::ifstream f(hidden_path, std::ios::binary | std::ios::ate);
        if (f) {
            const size_t bytes = (size_t)f.tellg();
            if (bytes == (size_t)d.layers * d.hidden * sizeof(float)) {
                golden_hidden.resize((size_t)d.layers * d.hidden);
                f.seekg(0);
                f.read((char*)golden_hidden.data(), bytes);
            } else {
                std::printf("note: %s is %zu bytes, expected %zu; skipping the per-layer check\n",
                            hidden_path.c_str(), bytes, (size_t)d.layers * d.hidden * sizeof(float));
            }
        } else {
            std::printf("note: no %s; per-layer check skipped (reference_run.py writes it)\n",
                        hidden_path.c_str());
        }
    }

    // ---- residency check before anything expensive happens
    int blocks_per_cu = 0;
    HIP_CHECK(fleet_occupancy(&blocks_per_cu));
    std::printf("occupancy: %d workgroup(s)/CU at %d threads (need >= 1 for grid %d)\n",
                blocks_per_cu, kBlock, kGrid);
    if (blocks_per_cu < 1) {
        die("cooperative launch would fail: the kernel does not fit one workgroup "
            "per CU. Check register pressure with -Rpass-analysis=kernel-resource-usage.");
    }

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // ---- placement probe: kGrid blocks, counted per XCC_ID
    {
        uint32_t* d_hist = dev_alloc<uint32_t>(16);
        HIP_CHECK(fleet_probe_xcds(d_hist, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        uint32_t hist[16];
        HIP_CHECK(hipMemcpy(hist, d_hist, sizeof(hist), hipMemcpyDeviceToHost));
        std::printf("placement: %d workgroups by XCC_ID:", kGrid);
        bool ok = true;
        for (int x = 0; x < 16; ++x) {
            if (hist[x] || x < kXCDs) std::printf(" [%d]=%u", x, hist[x]);
            if (x < kXCDs ? hist[x] != (uint32_t)kCUsPerXCD : hist[x] != 0) ok = false;
        }
        std::printf("%s\n", ok ? "  (8 x 38, as assumed)" : "");
        if (!ok) {
            die("workgroups are not placed 38 per XCD on this device/partition mode; "
                "the per-XCD queues in design.md §4 assume they are (SPX mode, 304 CUs)");
        }
        HIP_CHECK(hipFree(d_hist));
    }

    // ---- decode loop: one launch per token (v1 in §4)
    hipEvent_t ev0, ev1;
    HIP_CHECK(hipEventCreate(&ev0));
    HIP_CHECK(hipEventCreate(&ev1));

    std::vector<float> ms(tokens);
    std::vector<int> produced(tokens, -1);
    int token = first_token, matched = 0, compared = 0, first_div = -1;
    std::printf("\n%s: %d tokens over a %d-token context%s\n",
                smoke ? "smoke run (protocol only)" : (teacher ? "teacher-forced decode" : "free-running decode"),
                tokens, context, teacher ? " (HF token fed each step)" : "");

    int layers_ok = -1;   // consecutive layers within the §6 gate, from layer 0
    for (int i = 0; i < tokens; ++i) {
        rt.epoch = (uint32_t)(i + 1);       // never reset; targets are epoch x producers
        d.seq_len = context + i + 1;        // rows valid after this step
        rt.dump_layers = (!smoke && i == 0) ? 1 : 0;
        rt.trace = (d_trace && i == 1) ? d_trace : nullptr;
        if (!smoke) {
            act.cos = d_cos + (size_t)(context + i) * d.qk_rope;
            act.sin = d_sin + (size_t)(context + i) * d.qk_rope;
        }
        HIP_CHECK(hipMemsetAsync(misc, 0, 16 * sizeof(uint32_t), stream));
        HIP_CHECK(hipEventRecord(ev0, stream));
        HIP_CHECK(fleet_launch_step(&rt, &d, &w, &act, &kv, token, stream));
        HIP_CHECK(hipEventRecord(ev1, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipEventElapsedTime(&ms[i], ev0, ev1));

        uint32_t abort_code = 0;
        HIP_CHECK(hipMemcpy(&abort_code, rt.abort, sizeof(abort_code), hipMemcpyDeviceToHost));
        if (abort_code != kAbortNone) {
            std::printf("token %2d: %8.3f ms  ABORT\n", i, ms[i]);
            describe_abort(abort_code, labels);
            return 2;
        }

        if (smoke) {
            std::printf("token %2d: %8.3f ms  (protocol only)\n", i, ms[i]);
            continue;
        }
        int next = -1;
        HIP_CHECK(hipMemcpy(&next, act.next_token, sizeof(next), hipMemcpyDeviceToHost));
        produced[i] = next;
        const bool have_ref = (size_t)(i + 1) < golden.size();
        std::string verdict;
        if (have_ref) {
            ++compared;
            if (next == golden[i + 1]) { ++matched; verdict = "ok"; }
            else { verdict = "MISMATCH (HF " + std::to_string(golden[i + 1]) + ")"; if (first_div < 0) first_div = i; }
        }
        std::printf("token %2d: %8.3f ms  in %6d -> out %6d  %s\n", i, ms[i], token, next, verdict.c_str());
        token = (teacher && have_ref) ? golden[i + 1] : next;

        // ---- per-layer comparison on the first step (design.md §6): the
        // drift curve of Fleet output feeding Fleet, layer by layer, against
        // HF's states for the same step. Gate: max rel <= 2e-2, cosine >= 0.999.
        if (i == 0 && (!golden_hidden.empty() || !dump_path.empty())) {
            std::vector<float> ours((size_t)d.layers * d.hidden);
            HIP_CHECK(hipMemcpy(ours.data(), act.layer_dump, ours.size() * sizeof(float),
                                hipMemcpyDeviceToHost));
            if (!dump_path.empty()) {
                std::ofstream o(dump_path, std::ios::binary);
                o.write((const char*)ours.data(), ours.size() * sizeof(float));
                std::printf("wrote %s (fp32 [%d][%d])\n", dump_path.c_str(), d.layers, d.hidden);
            }
            if (!golden_hidden.empty()) {
                std::printf("\nper-layer output vs HF, first decode step (gate: max rel <= 2e-2, cosine >= 0.999)\n");
                layers_ok = 0;
                bool chain = true;
                for (int L = 0; L < d.layers; ++L) {
                    const float* a = ours.data() + (size_t)L * d.hidden;
                    const float* b = golden_hidden.data() + (size_t)L * d.hidden;
                    double dot = 0, na = 0, nb = 0, max_abs = 0, max_ref = 0;
                    for (int j = 0; j < d.hidden; ++j) {
                        dot += (double)a[j] * b[j]; na += (double)a[j] * a[j]; nb += (double)b[j] * b[j];
                        max_abs = std::max(max_abs, (double)std::fabs(a[j] - b[j]));
                        max_ref = std::max(max_ref, (double)std::fabs(b[j]));
                    }
                    const double rel = max_abs / std::max(max_ref, 1e-12);
                    const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
                    const bool ok = rel <= 2e-2 && cos >= 0.999;
                    if (chain && ok) ++layers_ok; else chain = false;
                    std::printf("  layer %2d %s: max rel %9.2e  cosine %.6f  [%s]\n", L,
                                L < d.first_k_dense ? "dense" : "moe  ", rel, cos, ok ? "ok" : "FAIL");
                }
                std::printf("  consecutive layers within the gate: %d of %d\n\n", layers_ok, d.layers);
            }
        }
    }

    // ---- per-task timeline of token 1 (design.md §10): where the time goes.
    // Each descriptor has (began waiting, wait satisfied, signalled, xcd) in
    // 100 MHz ticks. Aggregated per task kind and per layer; the raw rows are
    // written for bench/gantt.py.
    if (d_trace && tokens > 1) {
        std::vector<uint64_t> tr(tasks.size() * 4);
        HIP_CHECK(hipMemcpy(tr.data(), d_trace, tr.size() * sizeof(uint64_t), hipMemcpyDeviceToHost));
        uint64_t t0 = ~0ull, t1 = 0;
        for (size_t i = 0; i < tasks.size(); ++i) {
            if (tr[4 * i] == 0) continue;
            t0 = std::min(t0, tr[4 * i]); t1 = std::max(t1, tr[4 * i + 2]);
        }
        static const char* kind_names[] = {"QKV_FUSED", "ATTENTION", "MERGE_UV", "O_PROJ",
            "NORM_ROUTER", "EXPERT_GATE_UP", "EXPERT_DOWN", "REDUCE", "DENSE_GATE_UP",
            "DENSE_DOWN", "EMBED", "LM_HEAD", "ARGMAX"};
        // per kind: busy (sum of run time), waiting (sum of wait time), and the
        // phase span (first start to last end across all its descriptors).
        struct Agg { double busy = 0, wait = 0; uint64_t first = ~0ull, last = 0; int n = 0; };
        std::map<int, Agg> by_kind;
        std::map<int, std::pair<uint64_t, uint64_t>> by_layer;   // span per layer
        for (size_t i = 0; i < tasks.size(); ++i) {
            if (tr[4 * i] == 0) continue;
            const auto& t = tasks[i];
            Agg& a = by_kind[t.kind];
            a.busy += (double)(tr[4 * i + 2] - tr[4 * i + 1]);
            a.wait += (double)(tr[4 * i + 1] - tr[4 * i]);
            a.first = std::min(a.first, tr[4 * i + 1]); a.last = std::max(a.last, tr[4 * i + 2]); ++a.n;
            auto& L = by_layer[t.layer];
            if (L.first == 0) L = {tr[4 * i + 1], tr[4 * i + 2]};
            L.first = std::min(L.first, tr[4 * i + 1]); L.second = std::max(L.second, tr[4 * i + 2]);
        }
        const double us = 0.01;   // 100 MHz ticks -> us
        std::printf("\ntrace of token 1: %.3f ms from first wait to last signal\n", (t1 - t0) * us / 1e3);
        std::printf("  %-15s %6s %10s %10s %10s\n", "kind", "tasks", "busy-sum", "wait-sum", "avg-busy");
        for (auto& [k, a] : by_kind) {
            std::printf("  %-15s %6d %8.2f ms %8.2f ms %8.1f us\n", kind_names[k], a.n,
                        a.busy * us / 1e3, a.wait * us / 1e3, a.busy * us / a.n);
        }
        std::printf("  per-layer span (first task ready -> last task signalled): ");
        double sum_layers = 0;
        for (auto& [L, sp] : by_layer) {
            if (L < 0) continue;
            const double span = (sp.second - sp.first) * us;
            sum_layers += span;
            if (L <= 2 || L == 26) std::printf("L%d %.0f us  ", L, span);
        }
        std::printf("... sum %.2f ms\n", sum_layers / 1e3);
        std::ofstream o(trace_path, std::ios::binary);
        o.write((const char*)tr.data(), tr.size() * sizeof(uint64_t));
        std::printf("wrote %s (%zu descriptors x 4 uint64: wait-start, ready, done, xcd; 10 ns ticks)\n",
                    trace_path.c_str(), tasks.size());
    }

    // ---- summary
    std::vector<float> sorted(ms);
    std::sort(sorted.begin(), sorted.end());
    const float median = sorted[sorted.size() / 2];
    const float p95 = sorted[std::min(sorted.size() - 1, (size_t)(0.95 * sorted.size()))];
    double total = 0;
    for (float v : ms) total += v;
    std::printf("\nper-token latency: median %.3f ms  p95 %.3f ms  mean %.3f ms  "
                "(%.1f tok/s)\n", median, p95, total / tokens, 1000.0 * tokens / total);
    if (!smoke) {
        std::printf("tokens: %d/%d match HF greedy", matched, compared);
        if (first_div >= 0) std::printf(", first divergence at step %d", first_div);
        if (compared < tokens) {
            std::printf("  (%d step(s) had no reference token: reference_run.py "
                        "writes decode+1 tokens; --tokens must not exceed that)",
                        tokens - compared);
        }
        std::printf("\n");
    }
    if (!json_path.empty()) {
        std::ofstream j(json_path);
        j << "{\n  \"mode\": \"" << (smoke ? "smoke" : (teacher ? "teacher" : "free")) << "\",\n"
          << "  \"tokens\": " << tokens << ", \"context\": " << context << ",\n"
          << "  \"median_ms\": " << median << ", \"p95_ms\": " << p95 << ",\n"
          << "  \"matched\": " << matched << ", \"compared\": " << compared << ",\n"
          << "  \"layers_within_gate\": " << layers_ok << ", \"layers\": " << d.layers << ",\n  \"ms\": [";
        for (int i = 0; i < tokens; ++i) j << (i ? ", " : "") << ms[i];
        j << "],\n  \"produced\": [";
        for (int i = 0; i < tokens; ++i) j << (i ? ", " : "") << produced[i];
        j << "]\n}\n";
        std::printf("wrote %s\n", json_path.c_str());
    }
    // Success means every launch was compared and every comparison matched.
    return (smoke || (compared == tokens && matched == compared)) ? 0 : 3;
}
