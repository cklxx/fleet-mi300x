// The persistent kernel: one launch per token, 304 co-resident workgroups.
//
// docs/design.md §3, §4. Every workgroup enters here, discovers which chiplet
// it landed on, and becomes either that XCD's scheduler (one CU) or one of its
// 37 workers. Workers walk their pre-built queue: wait on the task's dependent
// event, run the task, signal completion. No host involvement, no kernel
// launches between operators — that is the entire point.
//
// This file holds the control structure and the small tasks. The bandwidth-
// critical bodies (GEMV, attention) live in separate headers so they can be
// tuned and unit-tested without touching the scheduling logic.

#include "runtime/fleet_runtime.h"
#include "runtime/fleet_types.h"
#include "gemv.h"
#include "attention.h"
#include "expert.h"

namespace fleet {

// ---------------------------------------------------------------- small tasks
//
// These are the "CU-task x1" entries in §3: microsecond-scale, single
// workgroup, and only standalone because they need the whole vector after a
// cross-XCD step.

__device__ void task_embed(const ModelDims& d, const Weights& w,
                           Activations& act, int token) {
    const __hip_bfloat16* row = w.embed + (int64_t)token * d.hidden;
    for (int i = threadIdx.x; i < d.hidden; i += blockDim.x) {
        act.x[i] = __bfloat162float(row[i]);
    }
}

// RMSNorm over the hidden vector into a separate output. Recomputed in the
// prologue of every GEMV task that needs it (§3 fusion 1): the vector is 4 KB,
// so recomputation is far cheaper than an extra global event.
__device__ inline void rms_norm_inplace(const ModelDims& d,
                                        const __hip_bfloat16* __restrict__ gamma,
                                        const float* __restrict__ in,
                                        float* __restrict__ out, float* smem) {
    float local = 0.f;
    for (int i = threadIdx.x; i < d.hidden; i += blockDim.x) {
        const float v = in[i];
        local += v * v;
    }
    const float ssq = block_reduce_ordered(local, smem);
    __syncthreads();
    const float inv = rsqrtf(ssq / d.hidden + d.rms_eps);
    for (int i = threadIdx.x; i < d.hidden; i += blockDim.x) {
        out[i] = in[i] * inv * __bfloat162float(gamma[i]);
    }
    __syncthreads();
}

// RMSNorm over the full hidden vector, then the router. Kept together because
// the router needs the normed vector and nothing else does before the experts.
__device__ void task_norm_router(const ModelDims& d, const Weights& w,
                                 Activations& act, int layer, bool moe) {
    __shared__ float ssq[fleet::kBlock / 64];
    __shared__ float inv_rms;

    float local = 0.f;
    for (int i = threadIdx.x; i < d.hidden; i += blockDim.x) {
        const float v = act.x[i];
        local += v * v;
    }
    // wave reduce, then across waves
    for (int off = 32; off; off >>= 1) local += __shfl_down(local, off);
    if ((threadIdx.x & 63) == 0) ssq[threadIdx.x >> 6] = local;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f;
        for (int i = 0; i < blockDim.x / 64; ++i) s += ssq[i];
        inv_rms = rsqrtf(s / d.hidden + d.rms_eps);
    }
    __syncthreads();

    const __hip_bfloat16* g = w.layer_norm_post[layer];
    for (int i = threadIdx.x; i < d.hidden; i += blockDim.x) {
        act.x_norm[i] = act.x[i] * inv_rms * __bfloat162float(g[i]);
    }
    __syncthreads();
    if (!moe) return;

    // Router: fp32 throughout, matching HF (softmax over 64 logits, top-k with
    // no renormalisation because norm_topk_prob is false and scale is 1.0).
    __shared__ float logits[64];
    const __hip_bfloat16* rw = w.layer_router[layer];
    for (int e = threadIdx.x; e < d.n_routed; e += blockDim.x) {
        float acc = 0.f;
        const __hip_bfloat16* row = rw + (int64_t)e * d.hidden;
        for (int i = 0; i < d.hidden; ++i) acc += __bfloat162float(row[i]) * act.x_norm[i];
        logits[e] = acc;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float mx = logits[0];
        for (int e = 1; e < d.n_routed; ++e) mx = fmaxf(mx, logits[e]);
        float sum = 0.f;
        for (int e = 0; e < d.n_routed; ++e) { logits[e] = __expf(logits[e] - mx); sum += logits[e]; }
        const float inv = 1.f / sum;
        for (int e = 0; e < d.n_routed; ++e) logits[e] *= inv;

        // top-k by selection: k is 6, so a scan beats a sort.
        bool taken[64] = {false};
        for (int k = 0; k < d.top_k; ++k) {
            int best = -1;
            float bv = -1.f;
            for (int e = 0; e < d.n_routed; ++e) {
                if (!taken[e] && logits[e] > bv) { bv = logits[e]; best = e; }
            }
            taken[best] = true;
            act.topk_ids[k] = best;
            act.topk_w[k] = bv;          // unnormalised, scale 1.0
        }
    }
}

// Fixed-order reduction of the 8 expert partials plus the residual. No float
// atomics anywhere: the determinism rule in §4 is what makes the bitwise
// reproducibility check in §6 meaningful.
__device__ void task_reduce(const ModelDims& d, Activations& act) {
    for (int i = threadIdx.x; i < d.hidden; i += blockDim.x) {
        float acc = 0.f;
        for (int x = 0; x < fleet::kXCDs; ++x) acc += act.expert_out[x * d.hidden + i];
        act.x[i] = __bfloat162float(__float2bfloat16(act.x[i] + acc));
    }
}

__device__ void task_argmax(const ModelDims& d, Activations& act) {
    __shared__ float best_val[fleet::kBlock];
    __shared__ int best_idx[fleet::kBlock];

    float bv = -INFINITY;
    int bi = 0;
    for (int i = threadIdx.x; i < d.vocab; i += blockDim.x) {
        if (act.logits[i] > bv) { bv = act.logits[i]; bi = i; }
    }
    best_val[threadIdx.x] = bv;
    best_idx[threadIdx.x] = bi;
    __syncthreads();
    for (int s = blockDim.x >> 1; s; s >>= 1) {
        if (threadIdx.x < s) {
            // Tie-break on the lower index so the result is reproducible.
            const bool take = best_val[threadIdx.x + s] > best_val[threadIdx.x] ||
                              (best_val[threadIdx.x + s] == best_val[threadIdx.x] &&
                               best_idx[threadIdx.x + s] < best_idx[threadIdx.x]);
            if (take) {
                best_val[threadIdx.x] = best_val[threadIdx.x + s];
                best_idx[threadIdx.x] = best_idx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *act.next_token = best_idx[0];
}

// ---------------------------------------------------------------- dispatch

__device__ void run_task(const TaskDescriptor& t, const ModelDims& d,
                         const Weights& w, Activations& act, KVCache& kv,
                         int token, int xcd, int worker, float* smem) {
    const int L = t.layer;
    const int nw = fleet::kWorkersPerXCD;

    switch (t.kind) {
        case TASK_EMBED:       task_embed(d, w, act, token); break;
        case TASK_NORM_ROUTER: task_norm_router(d, w, act, L,
                                                L >= d.first_k_dense); break;
        case TASK_REDUCE:      task_reduce(d, act); break;
        case TASK_ARGMAX:      task_argmax(d, act); break;

        // q_proj ‖ kv_a as one GEMV: 3072 q columns + 576 kv_a columns split
        // evenly over 8 XCDs (456 each = 2 heads' worth plus kv_a). RMSNorm is
        // recomputed per worker in the prologue rather than costing an event.
        case TASK_QKV_FUSED: {
            rms_norm_inplace(d, w.layer_norm_in[L], act.x, act.x_norm, smem);
            const int N = d.heads * d.q_head_dim + d.kv_lora + d.qk_rope;
            gemv_chiplet(w.layer_q_proj[L], act.x_norm, act.q, N, d.hidden,
                         xcd, fleet::kXCDs, worker, nw, smem);
            break;
        }

        // Attention: kv post + q-absorb in the prologue, then this task's slice
        // of the cache. XCD k owns heads 2k, 2k+1 so the merge stays local.
        case TASK_ATTENTION: {
            const int row = d.kv_lora + d.qk_rope;
            float* cache_row_f = act.x_norm;   // scratch, already consumed
            kv_post(act.kv_a, w.layer_kv_a_norm[L], act.cos, act.sin,
                    cache_row_f, d.kv_lora, d.qk_rope, d.rms_eps, smem,
                    /*write_cache=*/t.head == 0 && t.kv_chunk == 0,
                    kv.data + ((int64_t)L * kv.max_pos + d.seq_len - 1) * kv.row);
            float* q_c = act.q_c + (int64_t)t.head * d.kv_lora;
            q_absorb(act.q + (int64_t)t.head * d.q_head_dim, w.layer_kv_b[L],
                     q_c, t.head, d.qk_nope, d.v_head, d.kv_lora);
            const int stride = 2 + d.kv_lora;
            attention_chunk(kv.data + (int64_t)L * kv.max_pos * kv.row, q_c,
                            act.q + (int64_t)t.head * d.q_head_dim + d.qk_nope,
                            d.softmax_scale, d.seq_len, d.kv_lora, d.qk_rope,
                            kv.row, t.kv_chunk, max(1, (int)t.n_split),
                            act.attn_partial +
                                ((int64_t)t.head * max(1, (int)t.n_split) + t.kv_chunk) * stride,
                            smem);
            break;
        }

        case TASK_MERGE_UV: {
            const int stride = 2 + d.kv_lora;
            merge_and_uv(act.attn_partial + (int64_t)t.head * max(1, (int)t.n_split) * stride,
                         w.layer_kv_b[L], act.o, t.head, max(1, (int)t.n_split),
                         d.kv_lora, d.qk_nope, d.v_head, stride, act.x_norm);
            break;
        }

        // o_proj needs all 16 heads, so it is the layer's one unavoidable
        // global event; the residual add is folded into its epilogue.
        case TASK_O_PROJ:
            gemv_chiplet_epi(w.layer_o_proj[L], act.o, act.x, act.x,
                             d.hidden, d.heads * d.v_head, xcd, fleet::kXCDs,
                             worker, nw, EPI_RESIDUAL, smem);
            break;

        case TASK_EXPERT_GATE_UP: {
            const ExpertUnit u = resolve_unit(
                t.expert_slot, w.layer_experts[L], w.layer_shared[L],
                act.topk_ids, act.topk_w, d.hidden, d.moe_inter,
                d.moe_inter * 2, w.expert_stride);
            expert_gate_up(u, act.x_norm, act.expert_h + (int64_t)xcd * d.moe_inter * 2,
                           d.hidden, worker, nw, smem);
            break;
        }

        case TASK_EXPERT_DOWN: {
            const ExpertUnit u = resolve_unit(
                t.expert_slot, w.layer_experts[L], w.layer_shared[L],
                act.topk_ids, act.topk_w, d.hidden, d.moe_inter,
                d.moe_inter * 2, w.expert_stride);
            expert_down(u, act.expert_h + (int64_t)xcd * d.moe_inter * 2,
                        act.expert_out + (int64_t)xcd * d.hidden,
                        d.hidden, worker, nw, smem);
            break;
        }

        case TASK_DENSE_GATE_UP:
            dense_gate_up(w.layer_dense[L], act.x_norm, act.expert_h, d.hidden,
                          d.dense_inter, xcd, fleet::kXCDs, worker, nw, smem);
            break;

        case TASK_DENSE_DOWN:
            gemv_chiplet_epi(w.layer_dense[L] + (int64_t)2 * d.dense_inter * d.hidden,
                             act.expert_h, act.x, act.x, d.hidden, d.dense_inter,
                             xcd, fleet::kXCDs, worker, nw, EPI_RESIDUAL, smem);
            break;

        case TASK_LM_HEAD:
            rms_norm_inplace(d, w.final_norm, act.x, act.x_norm, smem);
            gemv_chiplet(w.lm_head, act.x_norm, act.logits, d.vocab, d.hidden,
                         xcd, fleet::kXCDs, worker, nw, smem);
            break;

        default:
            break;
    }
}

// ---------------------------------------------------------------- entry point
//
// Launched with hipLaunchCooperativeKernel so all 304 workgroups are guaranteed
// co-resident: a scheduler that waits for an absent worker would deadlock, and
// the cooperative launch removes that failure mode by construction rather than
// by hoping the occupancy works out.

extern "C" __global__ __launch_bounds__(fleet::kBlock, 1)
void fleet_decode_step(RuntimeState rt, ModelDims d, Weights w,
                       Activations act, KVCache kv, int token,
                       uint32_t* done_flag) {
    const int xcd = xcd_id();

    if (is_scheduler()) {
        run_scheduler(rt, xcd, done_flag);
        return;
    }

    // One reduction scratchpad per workgroup, reused by every task body.
    __shared__ float smem[fleet::kBlock];

    const int worker = worker_id();
    for (int slot = 0;; ++slot) {
        const TaskDescriptor* t = fetch_task(rt, xcd, worker, slot);
        if (t == nullptr) break;

        if (t->wait_event >= 0) {
            // n_split on the *producing* event is what we must reach; the host
            // stored it on every consumer's descriptor so the kernel never has
            // to walk the event table.
            wait_event(rt, t->wait_event, (EventScope)t->signal_scope, xcd,
                       t->n_split);
        }
        __syncthreads();

        run_task(*t, d, w, act, kv, token, xcd, worker, smem);
        __syncthreads();

        if (t->signal_event >= 0 && threadIdx.x == 0) {
            signal_event(rt, t->signal_event, (EventScope)t->signal_scope, xcd);
        }
    }

    // The last worker out tells the schedulers to stop mirroring.
    if (worker == 0 && threadIdx.x == 0) {
        atomicAdd(done_flag, 1u);
    }
}

}  // namespace fleet
