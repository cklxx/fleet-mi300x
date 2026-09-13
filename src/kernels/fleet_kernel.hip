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
//
// A Chiplet-task (§3) reaches the device as one descriptor per worker: the
// host emits 37 descriptors per XCD for it, each carrying the worker id, and
// the row split is a pure function of (xcd, worker, wave) in gemv.h. The
// event the Chiplet-task signals therefore has 8 x 37 = 296 producers.

#include "runtime/fleet_runtime.h"
#include "runtime/fleet_types.h"
#include "gemv.h"
#include "attention.h"
#include "expert.h"

namespace fleet {

// Per-workgroup LDS. Every task body that needs scratch gets it here, never
// from a global buffer another workgroup might be using at the same time.
struct Scratch {
    float smem[kBlock];                          // block reductions
    float vec[kMaxHidden];                       // staged GEMV operand
    float row[kMaxKvLora + kMaxQkRope];          // this token's cache row
    float q_c[kMaxKvLora];
    float q_pe[kMaxQkRope];
    float part[kWaves * kPartialStride];         // per-wave attention partials
    float logits[kMaxRouted];
};

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

// RMSNorm of the hidden vector into LDS. Recomputed in the prologue of every
// GEMV task that needs it (§3 fusion 1): the vector is 8 KB, so recomputation
// is far cheaper than an extra global event, and the GEMV then streams its
// operand from LDS.
//
// HF's DeepseekV2RMSNorm in bf16: variance and scaling in fp32, the scaled
// value cast to bf16, then multiplied by the bf16 weight — two roundings,
// reproduced exactly here (reference_decode.py rms_norm with bf16=True).
__device__ inline void rms_norm_to_lds(const ModelDims& d,
                                       const __hip_bfloat16* __restrict__ gamma,
                                       const float* __restrict__ in,
                                       float* __restrict__ out, float* smem) {
    float local = 0.f;
    for (int i = threadIdx.x; i < d.hidden; i += blockDim.x) {
        const float v = in[i];
        local += v * v;
    }
    const float ssq = block_reduce_ordered(local, smem);
    const float inv = rsqrtf(ssq / d.hidden + d.rms_eps);
    for (int i = threadIdx.x; i < d.hidden; i += blockDim.x) {
        out[i] = bf16_round(bf16_round(in[i] * inv) * __bfloat162float(gamma[i]));
    }
    __syncthreads();
}

// RMSNorm over the full hidden vector, then the router. Kept together because
// the router needs the normed vector and nothing else does before the experts.
// The normed vector is published to act.x_norm for the expert tasks.
__device__ void task_norm_router(const ModelDims& d, const Weights& w,
                                 Activations& act, Scratch& s, int layer, bool moe) {
    rms_norm_to_lds(d, w.layer_norm_post[layer], act.x, s.vec, s.smem);
    for (int i = threadIdx.x; i < d.hidden; i += blockDim.x) act.x_norm[i] = s.vec[i];
    if (!moe) return;

    // Router: fp32 throughout, matching HF (softmax over 64 logits, top-k with
    // no renormalisation because norm_topk_prob is false).
    gemv_rows(w.layer_router[layer], d.hidden, s.vec, s.logits, nullptr, EPI_NONE,
              1.f, d.n_routed, d.hidden, 0, 1, 0, 1);
    __syncthreads();

    if (threadIdx.x == 0) {
        float mx = s.logits[0];
        for (int e = 1; e < d.n_routed; ++e) mx = fmaxf(mx, s.logits[e]);
        float sum = 0.f;
        for (int e = 0; e < d.n_routed; ++e) { s.logits[e] = __expf(s.logits[e] - mx); sum += s.logits[e]; }
        const float inv = 1.f / sum;
        for (int e = 0; e < d.n_routed; ++e) s.logits[e] *= inv;

        // top-k by selection: k is 6, so a scan beats a sort. Ties go to the
        // lower index, matching torch.topk's stable order on equal scores.
        bool taken[kMaxRouted] = {false};
        for (int k = 0; k < d.top_k; ++k) {
            int best = -1;
            float bv = -1.f;
            for (int e = 0; e < d.n_routed; ++e) {
                if (!taken[e] && s.logits[e] > bv) { bv = s.logits[e]; best = e; }
            }
            taken[best] = true;
            act.topk_ids[k] = best;
            act.topk_w[k] = bv;          // unnormalised; scaling applied in resolve_unit
        }
    }
}

// Fixed-order reduction of the 8 expert partials plus the residual. No float
// atomics anywhere: the determinism rule in §4 is what makes the bitwise
// reproducibility check in §6 meaningful. The MoE output is a bf16 tensor in
// HF and the residual add is a bf16 add, hence the two roundings.
__device__ void task_reduce(const ModelDims& d, Activations& act) {
    for (int i = threadIdx.x; i < d.hidden; i += blockDim.x) {
        float acc = 0.f;
        for (int x = 0; x < kXCDs; ++x) acc += act.expert_out[x * d.hidden + i];
        act.x[i] = bf16_round(act.x[i] + bf16_round(acc));
    }
}

// Greedy pick over the bf16 logits (lm_head writes them bf16-rounded, as HF's
// lm_head does); ties go to the lower index, as torch.argmax does.
__device__ void task_argmax(const ModelDims& d, Activations& act) {
    __shared__ float best_val[kBlock];
    __shared__ int best_idx[kBlock];

    float bv = -INFINITY;
    int bi = 0;
    for (int i = threadIdx.x; i < d.vocab; i += blockDim.x) {
        if (act.logits[i] > bv) { bv = act.logits[i]; bi = i; }
    }
    best_val[threadIdx.x] = bv;
    best_idx[threadIdx.x] = bi;
    __syncthreads();
    for (int s = (int)blockDim.x >> 1; s; s >>= 1) {
        if ((int)threadIdx.x < s) {
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

// Layer-boundary capture (§6): x at the entry of layer L's QKV task is the
// output of layer L-1, and x at lm_head's entry is the last layer's. One
// worker copies it out; every other worker only reads x here, so the copy
// races with nothing. Enabled by the launcher for the first token only,
// which is the step the golden per-layer states were captured for.
__device__ __forceinline__ void dump_layer_output(const ModelDims& d, const Activations& act,
                                                  int layer_done) {
    float* dst = act.layer_dump + (int64_t)layer_done * d.hidden;
    for (int i = threadIdx.x; i < d.hidden; i += blockDim.x) dst[i] = act.x[i];
}

__device__ void run_task(const TaskDescriptor& t, const ModelDims& d,
                         const Weights& w, Activations& act, KVCache& kv,
                         Scratch& s, int token, int xcd, int worker, int dump) {
    const int L = t.layer;
    const int nw = kWorkersPerXCD;
    const int chunks = max(1, (int)t.n_split);
    const bool dumper = dump && xcd == 0 && worker == 0;

    switch (t.kind) {
        case TASK_EMBED:       task_embed(d, w, act, token); break;
        case TASK_NORM_ROUTER: task_norm_router(d, w, act, s, L,
                                                L >= d.first_k_dense); break;
        case TASK_REDUCE:      task_reduce(d, act); break;
        case TASK_ARGMAX:      task_argmax(d, act); break;

        // q_proj ‖ kv_a as one GEMV: 3072 q columns + 576 kv_a columns split
        // evenly over 8 XCDs (456 each = 2 heads' worth plus kv_a). RMSNorm is
        // recomputed per worker in the prologue rather than costing an event.
        case TASK_QKV_FUSED: {
            if (dumper && L > 0) dump_layer_output(d, act, L - 1);
            rms_norm_to_lds(d, w.layer_norm_in[L], act.x, s.vec, s.smem);
            const int N = d.heads * d.q_head_dim + d.kv_lora + d.qk_rope;
            gemv_rows(w.layer_qkv[L], d.hidden, s.vec, act.q, nullptr, EPI_BF16,
                      1.f, N, d.hidden, xcd, kXCDs, worker, nw);
            break;
        }

        // Attention: kv post + q-absorb (+ q_pe RoPE) in the prologue, all
        // into LDS, then this task's slice of the cache. XCD k owns heads 2k,
        // 2k+1 so the merge stays local.
        case TASK_ATTENTION: {
            __hip_bfloat16* cache_layer = kv.data + (int64_t)L * kv.max_pos * kv.row;
            kv_post(act.kv_a, w.layer_kv_a_norm[L], act.cos, act.sin, s.row,
                    d.kv_lora, d.qk_rope, d.rms_eps, s.smem,
                    /*write_cache=*/t.head == 0 && t.kv_chunk == 0,
                    cache_layer + (int64_t)(d.seq_len - 1) * kv.row);
            q_absorb(act.q + (int64_t)t.head * d.q_head_dim, w.layer_kv_b[L],
                     act.cos, act.sin, s.q_c, s.q_pe, t.head,
                     d.qk_nope, d.qk_rope, d.v_head, d.kv_lora);
            attention_chunk(cache_layer, s.q_c, s.q_pe, s.row, d.softmax_scale,
                            d.seq_len, kv.row, t.kv_chunk, chunks,
                            act.attn_partial +
                                ((int64_t)t.head * chunks + t.kv_chunk) * kPartialStride,
                            s.part);
            break;
        }

        case TASK_MERGE_UV:
            merge_and_uv(act.attn_partial + (int64_t)t.head * chunks * kPartialStride,
                         w.layer_kv_b[L], act.o, t.head, chunks,
                         d.kv_lora, d.qk_nope, d.v_head, s.q_c);
            break;

        // o_proj needs all 16 heads, so it is the layer's one unavoidable
        // global event; the residual add is folded into its epilogue.
        case TASK_O_PROJ:
            stage_vector(act.o, s.vec, d.heads * d.v_head);
            gemv_rows(w.layer_o_proj[L], d.heads * d.v_head, s.vec, act.x, act.x,
                      EPI_RESIDUAL, 1.f, d.hidden, d.heads * d.v_head,
                      xcd, kXCDs, worker, nw);
            break;

        case TASK_EXPERT_GATE_UP: {
            const ExpertUnit u = resolve_unit(
                t.expert_slot, w.layer_experts[L], w.layer_shared[L],
                act.topk_ids, act.topk_w, d.routed_scaling, d.hidden, d.moe_inter,
                d.moe_inter * 2, w.expert_stride);
            stage_vector(act.x_norm, s.vec, d.hidden);
            expert_gate_up(u, s.vec, act.expert_h + (int64_t)xcd * d.moe_inter,
                           d.hidden, worker, nw);
            break;
        }

        case TASK_EXPERT_DOWN: {
            const ExpertUnit u = resolve_unit(
                t.expert_slot, w.layer_experts[L], w.layer_shared[L],
                act.topk_ids, act.topk_w, d.routed_scaling, d.hidden, d.moe_inter,
                d.moe_inter * 2, w.expert_stride);
            stage_vector(act.expert_h + (int64_t)xcd * d.moe_inter, s.vec, u.inter);
            expert_down(u, s.vec, act.expert_out + (int64_t)xcd * d.hidden,
                        d.hidden, worker, nw);
            break;
        }

        case TASK_DENSE_GATE_UP:
            stage_vector(act.x_norm, s.vec, d.hidden);
            dense_gate_up(w.layer_dense[L], s.vec, act.expert_h, d.hidden,
                          d.dense_inter, xcd, kXCDs, worker, nw);
            break;

        // h is 10944 wide here: read from L2 rather than staged.
        case TASK_DENSE_DOWN:
            gemv_rows(w.layer_dense[L] + (int64_t)2 * d.dense_inter * d.hidden,
                      d.dense_inter, act.expert_h, act.x, act.x, EPI_RESIDUAL, 1.f,
                      d.hidden, d.dense_inter, xcd, kXCDs, worker, nw);
            break;

        case TASK_LM_HEAD:
            if (dumper) dump_layer_output(d, act, d.layers - 1);
            rms_norm_to_lds(d, w.final_norm, act.x, s.vec, s.smem);
            gemv_rows(w.lm_head, d.hidden, s.vec, act.logits, nullptr, EPI_BF16,
                      1.f, d.vocab, d.hidden, xcd, kXCDs, worker, nw);
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

extern "C" __global__ __launch_bounds__(kBlock, 1)
void fleet_decode_step(RuntimeState rt, ModelDims d, Weights w,
                       Activations act, KVCache kv, int token) {
    __shared__ Scratch s;
    __shared__ int slot_smem;

    const int xcd = xcd_id();
    const int worker = claim_role(rt, xcd, &slot_smem);
    if (aborted(rt)) return;          // wrong XCD distribution: everyone leaves

    if (worker < 0) {
        run_scheduler(rt, xcd);
        return;
    }

    __shared__ int ok_smem;
    for (int slot = 0;; ++slot) {
        const TaskDescriptor* t = fetch_task(rt, xcd, worker, slot);
        if (t == nullptr) break;

        if (threadIdx.x == 0) {
            ok_smem = wait_event(rt, t->wait_event, (EventScope)t->wait_scope, xcd,
                                 t->wait_count) ? 1 : 0;
        }
        __syncthreads();
        if (!ok_smem) break;          // abort raised somewhere: drain, exit

        if (!rt.smoke) run_task(*t, d, w, act, kv, s, token, xcd, worker, rt.dump_layers);
        __syncthreads();

        if (threadIdx.x == 0) {
            signal_event(rt, t->signal_event, (EventScope)t->signal_scope, xcd);
        }
    }
}

}  // namespace fleet

// ---------------------------------------------------------------- host entry
//
// The launch lives in this translation unit so the host stub the runtime
// registers for fleet_decode_step is the one being launched; the launcher
// only ever sees these two C functions.

// Pre-flight: where do kGrid workgroups actually land? One ordinary launch,
// each block counts itself under its HW_REG_XCC_ID. The launcher prints the
// histogram and refuses to continue unless every XCD got kCUsPerXCD blocks,
// which is the placement assumption design.md §4 makes and claim_role() would
// otherwise only report after a cooperative launch has been paid for.
__global__ void fleet_probe_xcc(uint32_t* hist) {
    if (threadIdx.x == 0) atomicAdd(&hist[fleet::xcd_id() & 15], 1u);
}

extern "C" hipError_t fleet_probe_xcds(uint32_t* d_hist, hipStream_t stream) {
    hipLaunchKernelGGL(fleet_probe_xcc, dim3(fleet::kGrid), dim3(fleet::kBlock), 0,
                       stream, d_hist);
    return hipGetLastError();
}

extern "C" hipError_t fleet_occupancy(int* blocks_per_cu) {
    return hipOccupancyMaxActiveBlocksPerMultiprocessor(
        blocks_per_cu, (const void*)fleet::fleet_decode_step, fleet::kBlock, 0);
}

extern "C" hipError_t fleet_launch_step(const fleet::RuntimeState* rt,
                                        const fleet::ModelDims* d,
                                        const fleet::Weights* w,
                                        const fleet::Activations* act,
                                        const fleet::KVCache* kv, int token,
                                        hipStream_t stream) {
    void* args[] = {(void*)rt, (void*)d, (void*)w, (void*)act, (void*)kv, &token};
    return hipLaunchCooperativeKernel((const void*)fleet::fleet_decode_step,
                                      dim3(fleet::kGrid), dim3(fleet::kBlock),
                                      args, 0, stream);
}
