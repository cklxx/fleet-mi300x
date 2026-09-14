// The persistent kernel: one launch per token sequence, 304 co-resident
// workgroups.
//
// docs/design.md §3, §4. Every workgroup enters here, discovers which chiplet
// it landed on, and becomes either that XCD's scheduler (one CU) or one of its
// 37 workers. For each token of the launch, workers walk their pre-built
// queue: wait on the task's dependent event, run the task, signal completion.
// No host involvement, no kernel launches between operators or between
// tokens — that is the entire point.
//
// This file holds the control structure and the small tasks. The bandwidth-
// critical bodies (GEMV, attention) live in separate headers so they can be
// tuned and unit-tested without touching the scheduling logic.
//
// A Chiplet-task (§3) reaches the device as one descriptor per worker: the
// host emits 37 descriptors per XCD for it, each carrying the worker id, and
// the row split is a pure function of (xcd, worker, wave) in gemv.h.

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
    float vec[kMaxHidden];                       // staged GEMV operand / folded x
    float row[kMaxKvLora + kMaxQkRope];          // this token's cache row
    float q_c[kMaxKvLora];                       // q_c, then the merge's o_c
    float q_pe[kMaxQkRope];
    float part[kWaves * kPartialStride];         // per-wave attention partials
    float logits[kMaxRouted];
    int   ids[kMaxTopK];                         // this XCD's routing choice
    float ws[kMaxTopK];
};

// One token's loop variables, derived from the token index of the launch.
struct TokenCtx {
    int tok;            // 0 .. n_tokens-1 within this launch
    uint32_t epoch;     // rt.epoch0 + tok
    int pos;            // absolute position of this token
    int seq_len;        // rows valid after this step = pos + 1
    const float* cos;   // YaRN row for pos
    const float* sin;
    bool dump;          // copy layer outputs (first token of the launch)
    bool trace;         // per-task timeline for this token
};

// ---------------------------------------------------------------- small tasks

__device__ void task_embed(const ModelDims& d, const Weights& w,
                           Activations& act, const TokenCtx& tc) {
    if (threadIdx.x == 0 && act.token_time) {
        act.token_time[2 * tc.tok] = __builtin_amdgcn_s_memrealtime();
    }
    const int token = act.token_in[tc.tok];
    const __hip_bfloat16* row = w.embed + (int64_t)token * d.hidden;
    for (int i = threadIdx.x; i < d.hidden; i += blockDim.x) {
        act.x[i] = __bfloat162float(row[i]);
    }
}

// RMSNorm of an LDS vector in place. HF's DeepseekV2RMSNorm in bf16: variance
// and scaling in fp32, the scaled value cast to bf16, then multiplied by the
// bf16 weight — two roundings (reference_decode.py rms_norm with bf16=True).
__device__ inline void rms_norm_lds(const ModelDims& d,
                                    const __hip_bfloat16* __restrict__ gamma,
                                    float* __restrict__ v, float* smem) {
    float local = 0.f;
#pragma unroll 8
    for (int i = threadIdx.x; i < d.hidden; i += kBlockThreads) local += v[i] * v[i];
    const float ssq = block_reduce_ordered(local, smem);
    const float inv = rsqrtf(ssq / d.hidden + d.rms_eps);
#pragma unroll 8
    for (int i = threadIdx.x; i < d.hidden; i += kBlockThreads) {
        v[i] = bf16_round(bf16_round(v[i] * inv) * __bfloat162float(gamma[i]));
    }
    __syncthreads();
}

// The residual stream at the entry of layer `layer` (or of lm_head when
// layer == d.layers), staged into LDS. x is already folded: the globally
// last down worker of the previous MoE layer did that (fold_partials).
__device__ inline void stage_x(const ModelDims& d, const Activations& act, int layer,
                               int xcd, int worker, float* __restrict__ vec,
                               const TokenCtx& tc) {
    stage_vector(act.x, vec, d.hidden);
    // Layer-boundary capture (§6): vec holds the output of layer-1.
    if (tc.dump && xcd == 0 && worker == 0 && layer > 0) {
        float* out = act.layer_dump + (int64_t)(layer - 1) * d.hidden;
        for (int i = threadIdx.x; i < d.hidden; i += blockDim.x) out[i] = vec[i];
    }
}

// The retired REDUCE task, done once per layer by the globally last down
// worker while every other worker waits on the down event: x = bf16(x +
// bf16(sum of the 8 partials)) — HF's bf16 MoE output and bf16 residual add.
// The partials came from other XCDs: acquire first (their XCDs' last
// arrivals flushed them before adding their shares).
__device__ inline void fold_partials(const ModelDims& d, const Activations& act) {
    if (threadIdx.x == 0) fence_acquire();
    __syncthreads();
#pragma unroll 8
    for (int i = threadIdx.x; i < d.hidden; i += kBlockThreads) {
        float acc = 0.f;
        for (int x = 0; x < kNumXCDs; ++x) acc += act.expert_out[x * d.hidden + i];
        act.x[i] = bf16_round(act.x[i] + bf16_round(acc));
    }
    __syncthreads();
}

// RMSNorm over the full hidden vector, then this worker's rows of the router
// — a Chiplet-task per XCD, so the experts of this XCD wait on an XCD-local
// event and one CU is not asked to stream 256 KB alone. Every worker norms
// the same x in the same order; worker 0 publishes it for the experts.
__device__ void task_norm_router(const ModelDims& d, const Weights& w,
                                 Activations& act, Scratch& s, int layer,
                                 int xcd, int worker, bool moe) {
    stage_vector(act.x, s.vec, d.hidden);
    rms_norm_lds(d, w.layer_norm_post[layer], s.vec, s.smem);
    if (worker == 0) {
        float* x_norm = act.x_norm + (int64_t)xcd * d.hidden;
        for (int i = threadIdx.x; i < d.hidden; i += blockDim.x) x_norm[i] = s.vec[i];
    }
    if (!moe) return;
    // Router logits: fp32 throughout, matching HF's fp32 gate.
    gemv_rows(w.layer_router[layer], d.hidden, s.vec,
              act.router_logits + (int64_t)xcd * d.n_routed, nullptr, EPI_NONE,
              1.f, d.n_routed, d.hidden, 0, 1, worker, kWorkersPerXCD);
}

// Softmax over the router logits and top-k, on wave 0 with one expert per
// lane (max, sum and each argmax pick are shuffle reductions), into LDS.
// Run by every expert task of the XCD: 64 floats, deterministic, and it
// saves a task and an event. Ties go to the lower index (torch.topk order).
__device__ inline void select_experts(const ModelDims& d, const float* __restrict__ logits,
                                      int* __restrict__ ids, float* __restrict__ ws) {
    if (threadIdx.x < kWaveLanes) {
        const int lane = threadIdx.x;
        const bool valid = lane < d.n_routed;
        float v = valid ? logits[lane] : -INFINITY;
        float mx = v;
#pragma unroll
        for (int off = kWaveLanes / 2; off; off >>= 1) mx = fmaxf(mx, __shfl_xor(mx, off, kWaveLanes));
        float p = valid ? __expf(v - mx) : 0.f;
        const float inv = 1.f / wave_sum(p);
        p *= inv;
        for (int k = 0; k < d.top_k; ++k) {
            float bv = p;
            int bi = valid ? lane : 0x7fffffff;
#pragma unroll
            for (int off = kWaveLanes / 2; off; off >>= 1) {
                const float ov = __shfl_xor(bv, off, kWaveLanes);
                const int oi = __shfl_xor(bi, off, kWaveLanes);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (lane == 0) { ids[k] = bi; ws[k] = bv; }   // unnormalised
            if (lane == bi) p = -INFINITY;                // taken
        }
    }
    __syncthreads();
}

// Greedy pick over the bf16 logits (lm_head writes them bf16-rounded, as HF's
// lm_head does); ties go to the lower index, as torch.argmax does. Feeds the
// next token unless the host pre-filled token_in (teacher forcing).
__device__ void task_argmax(const ModelDims& d, Activations& act, const RuntimeState& rt,
                            const TokenCtx& tc) {
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
    if (threadIdx.x == 0) {
        act.token_out[tc.tok] = best_idx[0];
        if (!rt.teacher) act.token_in[tc.tok + 1] = best_idx[0];
        if (act.token_time) act.token_time[2 * tc.tok + 1] = __builtin_amdgcn_s_memrealtime();
    }
}

// ---------------------------------------------------------------- dispatch
//
// Returns whether the task should signal its signal_event: always, except
// for FLAG_SIGNAL_LAST tasks, where only the last arrival does.

__device__ bool run_task(const TaskDescriptor& t, const RuntimeState& rt,
                         const ModelDims& d, const Weights& w, Activations& act,
                         KVCache& kv, Scratch& s, const TokenCtx& tc, int xcd,
                         int worker) {
    const int L = t.layer;
    const int nw = kWorkersPerXCD;
    const int chunks = max(1, (int)t.n_split);
    const int heads_per_xcd = d.heads / kNumXCDs;

    switch (t.kind) {
        case TASK_EMBED:       task_embed(d, w, act, tc); break;
        case TASK_NORM_ROUTER: task_norm_router(d, w, act, s, L, xcd, worker,
                                                L >= d.first_k_dense); break;
        case TASK_ARGMAX:      task_argmax(d, act, rt, tc); break;
        case TASK_REDUCE:      break;   // retired: folded into fold_x

        // Merge the head's kv_chunks partials (every merge task of the head
        // recomputes the 16 KB merge) and apply this task's slice of W_UV.
        case TASK_MERGE_UV: {
            const float* partials = act.attn_partial + (int64_t)t.head * chunks * kPartialStride;
            const int rows = d.v_head / chunks;
            merge_and_uv(partials, w.layer_kv_b[L], act.o, t.head, chunks,
                         d.kv_lora, d.qk_nope, d.v_head, s.q_c,
                         t.kv_chunk * rows, rows);
            break;
        }

        // This XCD's heads' q rows plus its own copy of kv_a, from the fused
        // [3072 + 576, hidden] matrix. RMSNorm (and the fold of the previous
        // layer's expert partials) is recomputed per worker in the prologue.
        case TASK_QKV_FUSED: {
            stage_x(d, act, L, xcd, worker, s.vec, tc);
            rms_norm_lds(d, w.layer_norm_in[L], s.vec, s.smem);
            const int q_rows = heads_per_xcd * d.q_head_dim;                // 384
            const int q0 = xcd * q_rows;
            gemv_rows(w.layer_qkv[L] + (int64_t)q0 * d.hidden, d.hidden, s.vec,
                      act.q + q0, nullptr, EPI_BF16, 1.f, q_rows, d.hidden,
                      0, 1, worker, nw);
            const int kv_rows = d.kv_lora + d.qk_rope;                      // 576
            const int kv0 = d.heads * d.q_head_dim;
            gemv_rows(w.layer_qkv[L] + (int64_t)kv0 * d.hidden, d.hidden, s.vec,
                      act.kv_a + (int64_t)xcd * kv_rows, nullptr, EPI_BF16, 1.f,
                      kv_rows, d.hidden, 0, 1, worker, nw);
            break;
        }

        // Attention: kv post + q-absorb (+ q_pe RoPE) in the prologue, all
        // into LDS, then this task's slice of the cache into a partial.
        case TASK_ATTENTION: {
            __hip_bfloat16* cache_layer = kv.data + (int64_t)L * kv.max_pos * kv.row;
            const float* kv_a = act.kv_a + (int64_t)xcd * (d.kv_lora + d.qk_rope);
            kv_post(kv_a, w.layer_kv_a_norm[L], tc.cos, tc.sin, s.row,
                    d.kv_lora, d.qk_rope, d.rms_eps, s.smem,
                    /*write_cache=*/t.head == 0 && t.kv_chunk == 0,   // one writer per layer
                    cache_layer + (int64_t)tc.pos * kv.row);
            q_absorb(act.q + (int64_t)t.head * d.q_head_dim, w.layer_kv_b[L],
                     tc.cos, tc.sin, s.q_c, s.q_pe, s.part, t.head,
                     d.qk_nope, d.qk_rope, d.v_head, d.kv_lora);
            float* partials = act.attn_partial + (int64_t)t.head * chunks * kPartialStride;
            attention_chunk(cache_layer, s.q_c, s.q_pe, s.row, d.softmax_scale,
                            tc.seq_len, kv.row, t.kv_chunk, chunks,
                            partials + (int64_t)t.kv_chunk * kPartialStride, s.part);
            break;
        }

        // o_proj needs all 16 heads, so it is the layer's one unavoidable
        // global event; the residual add is folded into its epilogue.
        case TASK_O_PROJ:
            stage_vector(act.o, s.vec, d.heads * d.v_head);
            gemv_rows(w.layer_o_proj[L], d.heads * d.v_head, s.vec, act.x,
                      act.x, EPI_RESIDUAL, 1.f, d.hidden, d.heads * d.v_head,
                      xcd, kNumXCDs, worker, nw);
            break;

        case TASK_EXPERT_GATE_UP: {
            select_experts(d, act.router_logits + (int64_t)xcd * d.n_routed, s.ids, s.ws);
            const ExpertUnit u = resolve_unit(
                t.expert_slot, w.layer_experts[L], w.layer_shared[L],
                s.ids, s.ws, d.routed_scaling, d.hidden, d.moe_inter, d.moe_inter * 2,
                w.expert_stride);
            stage_vector(act.x_norm + (int64_t)xcd * d.hidden, s.vec, d.hidden);
            expert_gate_up(u, s.vec, act.expert_h + (int64_t)xcd * d.moe_inter,
                           d.hidden, worker, nw);
            break;
        }

        case TASK_EXPERT_DOWN: {
            select_experts(d, act.router_logits + (int64_t)xcd * d.n_routed, s.ids, s.ws);
            const ExpertUnit u = resolve_unit(
                t.expert_slot, w.layer_experts[L], w.layer_shared[L],
                s.ids, s.ws, d.routed_scaling, d.hidden, d.moe_inter, d.moe_inter * 2,
                w.expert_stride);
            stage_vector(act.expert_h + (int64_t)xcd * d.moe_inter, s.vec, u.inter);
            expert_down(u, s.vec, act.expert_out + (int64_t)xcd * d.hidden,
                        d.hidden, worker, nw);
            break;
        }

        case TASK_DENSE_GATE_UP:
            stage_vector(act.x_norm + (int64_t)xcd * d.hidden, s.vec, d.hidden);
            dense_gate_up(w.layer_dense[L], s.vec, act.expert_h, d.hidden,
                          d.dense_inter, xcd, kNumXCDs, worker, nw);
            break;

        // h is 10944 wide here: read from L2 rather than staged.
        case TASK_DENSE_DOWN:
            gemv_rows(w.layer_dense[L] + (int64_t)2 * d.dense_inter * d.hidden,
                      d.dense_inter, act.expert_h, act.x, act.x,
                      EPI_RESIDUAL, 1.f, d.hidden, d.dense_inter, xcd, kNumXCDs,
                      worker, nw);
            break;

        case TASK_LM_HEAD:
            stage_x(d, act, d.layers, xcd, worker, s.vec, tc);
            rms_norm_lds(d, w.final_norm, s.vec, s.smem);
            gemv_rows(w.lm_head, d.hidden, s.vec, act.logits, nullptr, EPI_BF16,
                      1.f, d.vocab, d.hidden, xcd, kNumXCDs, worker, nw);
            break;

        default:
            break;
    }
    return true;
}

// ---------------------------------------------------------------- entry point
//
// Launched with hipLaunchCooperativeKernel so all 304 workgroups are guaranteed
// co-resident: a scheduler that waits for an absent worker would deadlock, and
// the cooperative launch removes that failure mode by construction rather than
// by hoping the occupancy works out.

extern "C" __global__ __launch_bounds__(kBlock, kBlocksPerCU)
void fleet_decode_step(RuntimeState rt, ModelDims d, Weights w,
                       Activations act, KVCache kv) {
    __shared__ Scratch s;
    __shared__ int slot_smem;
    __shared__ int ok_smem;

    const int xcd = xcd_id();
    const int worker = claim_role(rt, xcd, &slot_smem);
    if (aborted(rt)) return;          // wrong XCD distribution: everyone leaves

    if (worker < 0) {
        run_scheduler(rt, xcd);
        return;
    }

    for (int tok = 0; tok < rt.n_tokens; ++tok) {
        TokenCtx tc;
        tc.tok = tok;
        tc.epoch = rt.epoch0 + (uint32_t)tok;
        tc.pos = d.context + tok;
        tc.seq_len = tc.pos + 1;
        tc.cos = act.cos + (int64_t)tc.pos * d.qk_rope;
        tc.sin = act.sin + (int64_t)tc.pos * d.qk_rope;
        tc.dump = rt.dump_layers && tok == 0;
        tc.trace = rt.trace != nullptr && tok == rt.trace_token;

        bool stop = false;
        for (int slot = 0;; ++slot) {
            const TaskDescriptor* t = fetch_task(rt, xcd, worker, slot);
            if (t == nullptr) break;

            uint64_t t_start = 0, t_ready = 0;
            if (threadIdx.x == 0) {
                if (tc.trace) t_start = __builtin_amdgcn_s_memrealtime();
                bool ok = true;
                // Token boundary: the embed of token t waits for token t-1's
                // final event (the previous epoch of the graph's last event).
                if (t->kind == TASK_EMBED && tok > 0) {
                    ok = wait_event(rt, tc.epoch - 1, rt.done_event, SCOPE_GLOBAL, xcd, 1);
                }
                if (ok) {
                    ok = wait_event(rt, tc.epoch, t->wait_event, (EventScope)t->wait_scope,
                                    xcd, t->wait_count);
                }
                ok_smem = ok ? 1 : 0;
                if (tc.trace) t_ready = __builtin_amdgcn_s_memrealtime();
            }
            __syncthreads();
            if (!ok_smem) { stop = true; break; }   // abort raised somewhere

            bool do_signal = true;
            if (!rt.smoke) {
                do_signal = run_task(*t, rt, d, w, act, kv, s, tc, xcd, worker);
            } else if (t->flags & FLAG_SIGNAL_LAST) {
                __shared__ int last_smem2;
                if (threadIdx.x == 0) {
                    last_smem2 = arrive_last(rt, tc.epoch, t->local_event, xcd,
                                             max(1, (int)t->n_split)) ? 1 : 0;
                }
                __syncthreads();
                do_signal = last_smem2 != 0;
            }
            __syncthreads();

            __shared__ uint32_t total_smem;
            if (threadIdx.x == 0) {
                total_smem = 0;
                if (do_signal) {
                    total_smem = signal_event(rt, tc.epoch, t->signal_event,
                                              (EventScope)t->signal_scope, xcd,
                                              t->signal_xcd_count);
                }
            }
            __syncthreads();
            // FOLD_ON_LAST: this workgroup completed the epoch's producer count
            // (all of them minus the fold's own +1): fold, publish, add the +1.
            if ((t->flags & FLAG_FOLD_ON_LAST) &&
                total_smem == tc.epoch * (uint32_t)t->n_split + (tc.epoch - 1)) {
                if (!rt.smoke) fold_partials(d, act);
                if (threadIdx.x == 0) {
                    fence_release();
                    __hip_atomic_fetch_add(&rt.global_events[t->signal_event], 1u,
                                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                }
            }
            if (threadIdx.x == 0) {
                if (tc.trace) {
                    uint64_t* tr = rt.trace + (int64_t)t->index * 4;
                    tr[0] = t_start;                                  // began waiting
                    tr[1] = t_ready;                                  // wait satisfied
                    tr[2] = __builtin_amdgcn_s_memrealtime();         // signalled
                    tr[3] = (uint64_t)xcd;
                }
            }
        }
        if (stop) break;
    }
}

}  // namespace fleet

// ---------------------------------------------------------------- host entry
//
// The launch lives in this translation unit so the host stub the runtime
// registers for fleet_decode_step is the one being launched; the launcher
// only ever sees these C functions.

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
                                        const fleet::KVCache* kv,
                                        hipStream_t stream) {
    void* args[] = {(void*)rt, (void*)d, (void*)w, (void*)act, (void*)kv};
    return hipLaunchCooperativeKernel((const void*)fleet::fleet_decode_step,
                                      dim3(fleet::kGrid), dim3(fleet::kBlock),
                                      args, 0, stream);
}
