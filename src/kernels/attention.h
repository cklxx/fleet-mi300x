// Absorbed-MLA decode attention — the CU-task body.
//
// docs/design.md §2, §3. The cache holds one compressed row per position,
// `c ‖ k_pe` (576 floats), shared by all 16 heads, so a whole layer's cache is
// 1.2 MB and sits in each XCD's 4 MB L2. Materialising K and V per head would
// read 7.1x more per position and add ~2 GFLOP, which at batch 1 is pure loss.
//
// Three things are fused into this task's prologue to remove global events (§3):
//   1. kv post-processing: RMSNorm on c, RoPE on k_pe, and the cache append
//      are re-derived here from the raw 576-vector rather than costing an event
//   2. q-absorb: q_c[h] = q_nope[h] @ W_UK[h], each task reading its own
//      131 KB W_UK slice
//   3. head-to-XCD affinity: XCD k owns heads 2k and 2k+1, so the merge that
//      follows waits on an XCD-local counter instead of a global event
//
// Split-KV (D4) turns each head into kv_chunks tasks emitting flash-decoding
// partials (m, l, acc[512]); with kv_chunks == 1 the partial is the answer.
#pragma once

#include <hip/hip_runtime.h>
#include "gemv.h"

namespace fleet {

// Partial state per (head, chunk): running max, running sum, accumulator.
// Laid out so the merge task reads them contiguously.
struct AttnPartial {
    float m;            // max score seen
    float l;            // sum of exp(score - m)
    // acc[kv_lora] follows immediately in memory
};

__device__ __forceinline__ float* partial_acc(float* base, int kv_lora) {
    return base + 2;   // skip m, l
}

// DeepseekV2 RoPE: interleave-transpose, then the standard rotation.
// x[0..d) becomes [x0,x2,...,x_{d-2}, x1,x3,...,x_{d-1}] before rotate_half.
// Implementing the plain form instead is silently wrong, which is why
// tests/test_absorbed_equivalence.py checks the two differ.
__device__ __forceinline__ void apply_rope_interleaved(
        float* __restrict__ v, const float* __restrict__ cos,
        const float* __restrict__ sin, int d, int lane, int lanes) {
    const int half = d / 2;
    // Gather into the transposed order first; d is 64 here, so a small
    // scratch in registers per lane is enough.
    for (int i = lane; i < half; i += lanes) {
        const float a = v[2 * i];        // -> position i
        const float b = v[2 * i + 1];    // -> position i + half
        // rotate_half pairs (i, i+half): out_i = a*cos_i - b*sin_i
        //                                out_{i+half} = b*cos_{i+half} + a*sin_{i+half}
        const float o1 = a * cos[i] - b * sin[i];
        const float o2 = b * cos[i + half] + a * sin[i + half];
        v[i] = o1;
        v[i + half] = o2;
    }
}

// Prologue step 1: rebuild this token's cache row from the raw kv_a output.
// Every attention task recomputes it (2.3 KB fp32); one designated task writes
// it to the cache. Recomputation is free next to an extra global event.
__device__ inline void kv_post(
        const float* __restrict__ kv_a_raw, const __hip_bfloat16* __restrict__ kv_a_norm_w,
        const float* __restrict__ cos, const float* __restrict__ sin,
        float* __restrict__ row_out, int kv_lora, int qk_rope, float eps,
        float* smem, bool write_cache, __hip_bfloat16* __restrict__ cache_row) {
    // RMSNorm over the compressed part
    float local = 0.f;
    for (int i = threadIdx.x; i < kv_lora; i += blockDim.x) {
        const float v = kv_a_raw[i];
        local += v * v;
    }
    const float ssq = block_reduce_ordered(local, smem);
    __syncthreads();
    const float inv = rsqrtf(ssq / kv_lora + eps);

    for (int i = threadIdx.x; i < kv_lora; i += blockDim.x) {
        row_out[i] = kv_a_raw[i] * inv * __bfloat162float(kv_a_norm_w[i]);
    }
    for (int i = threadIdx.x; i < qk_rope; i += blockDim.x) {
        row_out[kv_lora + i] = kv_a_raw[kv_lora + i];
    }
    __syncthreads();

    apply_rope_interleaved(row_out + kv_lora, cos, sin, qk_rope,
                           threadIdx.x, blockDim.x);
    __syncthreads();

    if (write_cache) {
        for (int i = threadIdx.x; i < kv_lora + qk_rope; i += blockDim.x) {
            cache_row[i] = __float2bfloat16(row_out[i]);
        }
    }
}

// Prologue step 2: absorb W_UK into the query.
//   q_c[h] = q_nope[h] @ W_UK[h],  W_UK[h] = kv_b_proj rows [h*256, h*256+128)
// This is a [128] x [128, 512] product per head — 131 KB of weights, read once
// per task. With 4 KV chunks it is read 4x, 0.5 MB per layer, 0.3% of traffic.
__device__ inline void q_absorb(
        const float* __restrict__ q_nope, const __hip_bfloat16* __restrict__ kv_b,
        float* __restrict__ q_c, int head, int qk_nope, int v_head, int kv_lora) {
    const int row_stride = qk_nope + v_head;               // 256
    const __hip_bfloat16* w_uk = kv_b + (int64_t)head * row_stride * kv_lora;

    for (int j = threadIdx.x; j < kv_lora; j += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < qk_nope; ++i) {
            acc += q_nope[i] * __bfloat162float(w_uk[(int64_t)i * kv_lora + j]);
        }
        q_c[j] = acc;
    }
    __syncthreads();
}

// Main body: flash-decoding over this task's slice of the cache.
//
//   s[t] = scale * (q_c . c[t] + q_pe . k_pe[t])
//   online softmax over t, accumulating acc += p[t] * c[t]
//
// The cache is read as bf16 and widened in registers; each XCD pulls the
// layer's 1.2 MB once and the second head on the same XCD hits L2.
__device__ inline void attention_chunk(
        const __hip_bfloat16* __restrict__ cache, const float* __restrict__ q_c,
        const float* __restrict__ q_pe, float scale,
        int seq_len, int kv_lora, int qk_rope, int row,
        int chunk, int n_chunks, float* __restrict__ out, float* smem) {
    const int per_chunk = (seq_len + n_chunks - 1) / n_chunks;
    const int t_begin = chunk * per_chunk;
    const int t_end = min(seq_len, t_begin + per_chunk);

    float* acc = partial_acc(out, kv_lora);
    for (int j = threadIdx.x; j < kv_lora; j += blockDim.x) acc[j] = 0.f;
    __syncthreads();

    __shared__ float run_m, run_l;
    if (threadIdx.x == 0) { run_m = -INFINITY; run_l = 0.f; }
    __syncthreads();

    for (int t = t_begin; t < t_end; ++t) {
        const __hip_bfloat16* r = cache + (int64_t)t * row;

        // score: both halves of the row in one pass
        float part = 0.f;
        for (int j = threadIdx.x; j < kv_lora; j += blockDim.x) {
            part += q_c[j] * __bfloat162float(r[j]);
        }
        for (int j = threadIdx.x; j < qk_rope; j += blockDim.x) {
            part += q_pe[j] * __bfloat162float(r[kv_lora + j]);
        }
        const float s = block_reduce_ordered(part, smem) * scale;
        __syncthreads();

        // online softmax rescale
        __shared__ float p, rescale;
        if (threadIdx.x == 0) {
            const float new_m = fmaxf(run_m, s);
            rescale = __expf(run_m - new_m);
            p = __expf(s - new_m);
            run_l = run_l * rescale + p;
            run_m = new_m;
        }
        __syncthreads();

        for (int j = threadIdx.x; j < kv_lora; j += blockDim.x) {
            acc[j] = acc[j] * rescale + p * __bfloat162float(r[j]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) { out[0] = run_m; out[1] = run_l; }
    __syncthreads();
}

// Merge task: combine the chunks' partials, then absorb W_UV.
//   o_c[h] = (sum_chunks rescaled acc) / l
//   o[h]   = o_c[h] @ W_UV[h]^T,  W_UV[h] = kv_b rows [h*256+128, h*256+256)
// Runs on the same XCD as its head, so it waits on an XCD-local counter.
__device__ inline void merge_and_uv(
        const float* __restrict__ partials, const __hip_bfloat16* __restrict__ kv_b,
        float* __restrict__ o, int head, int n_chunks, int kv_lora,
        int qk_nope, int v_head, int partial_stride, float* scratch) {
    // Global max across chunks, then a rescaled sum. Fixed chunk order keeps
    // this bitwise reproducible.
    __shared__ float gmax, gsum;
    if (threadIdx.x == 0) {
        float m = -INFINITY;
        for (int c = 0; c < n_chunks; ++c) m = fmaxf(m, partials[c * partial_stride]);
        float l = 0.f;
        for (int c = 0; c < n_chunks; ++c) {
            l += partials[c * partial_stride + 1] *
                 __expf(partials[c * partial_stride] - m);
        }
        gmax = m; gsum = l;
    }
    __syncthreads();

    for (int j = threadIdx.x; j < kv_lora; j += blockDim.x) {
        float v = 0.f;
        for (int c = 0; c < n_chunks; ++c) {
            const float* p = partials + c * partial_stride;
            v += p[2 + j] * __expf(p[0] - gmax);
        }
        scratch[j] = v / gsum;
    }
    __syncthreads();

    const int row_stride = qk_nope + v_head;
    const __hip_bfloat16* w_uv =
        kv_b + (int64_t)head * row_stride * kv_lora + (int64_t)qk_nope * kv_lora;

    for (int i = threadIdx.x; i < v_head; i += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < kv_lora; ++j) {
            acc += __bfloat162float(w_uv[(int64_t)i * kv_lora + j]) * scratch[j];
        }
        o[head * v_head + i] = acc;
    }
}

}  // namespace fleet
