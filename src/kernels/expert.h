// MoE expert task bodies — gate_up then down, one balanced unit per XCD.
//
// docs/design.md §3: the 8 XCDs each stream exactly one unit, where a unit is
// either one of the 6 routed experts (17.3 MB) or one half of the shared expert
// pair (also 17.3 MB). That is a perfect 8-way split with no remainder, which
// is why the expert phase has no straggler XCD.
//
// Two details make this work inside an immutable task queue:
//
//   * Indirect expert index (§3): the descriptor stores the routing *slot* k,
//     not an expert id. The kernel reads topk_ids[k] at run time and forms the
//     weight pointer as base + id * stride. The queue never changes, so the
//     ahead-of-time launch stays valid even though routing is data-dependent.
//
//   * gate_up -> down is XCD-local (§3): h[2816] lives in that XCD's L2 and the
//     workers synchronise on an L2 counter, so the two GEMVs are not separated
//     by a device fence. §12 refines this to tile granularity — down starts
//     consuming K-chunks of h as they land rather than waiting for all of it.
#pragma once

#include <hip/hip_runtime.h>
#include "gemv.h"
#include "../runtime/fleet_runtime.h"

namespace fleet {

// A unit is either a routed expert (slot >= 0) or a shared half (slot < 0).
struct ExpertUnit {
    const __hip_bfloat16* gate_up;   // [2 * inter, hidden], rows [gate; up] interleaved
    const __hip_bfloat16* down;      // [hidden, inter]
    float weight;                    // routing weight, 1.0 for shared halves
    int inter;                       // rows this unit owns
};

// Resolve the descriptor's slot into actual pointers. Routed experts index
// through topk_ids; shared halves split the merged shared expert in two so all
// 8 XCDs carry the same 17.3 MB.
__device__ inline ExpertUnit resolve_unit(
        int expert_slot, const __hip_bfloat16* __restrict__ experts_base,
        const __hip_bfloat16* __restrict__ shared_base,
        const int32_t* __restrict__ topk_ids, const float* __restrict__ topk_w,
        int hidden, int moe_inter, int shared_inter, int64_t expert_stride) {
    ExpertUnit u;
    if (expert_slot >= 0) {
        const int id = topk_ids[expert_slot];           // indirect, read at run time
        const __hip_bfloat16* base = experts_base + (int64_t)id * expert_stride;
        u.gate_up = base;
        u.down = base + (int64_t)2 * moe_inter * hidden;
        u.weight = topk_w[expert_slot];
        u.inter = moe_inter;
    } else {
        const int half = -expert_slot - 1;              // 0 or 1
        const int rows = shared_inter / 2;
        u.gate_up = shared_base + (int64_t)half * rows * hidden;
        u.down = shared_base + (int64_t)2 * shared_inter * hidden
                 + (int64_t)half * rows;                // down is [hidden, inter]
        u.weight = 1.f;
        u.inter = rows;
    }
    return u;
}

// gate_up + SiLU(gate) * up, fused.
//
// The weights are repacked offline so each row pair (gate_i, up_i) is adjacent
// (§12): the two dot products land in the same wave with no cross-lane shuffle
// and no LDS round-trip, and the result is written straight as the h vector the
// down GEMV will read out of L2.
__device__ inline void expert_gate_up(
        const ExpertUnit& u, const float* __restrict__ x,
        float* __restrict__ h, int hidden, int worker, int n_workers,
        float* smem) {
    for (int n = worker; n < u.inter; n += n_workers) {
        // interleaved layout: gate row at 2n, up row at 2n+1
        const float g_part = dot_row_bf16(u.gate_up + (int64_t)(2 * n) * hidden,
                                          x, hidden, threadIdx.x, blockDim.x);
        const float g = block_reduce_ordered(g_part, smem);
        __syncthreads();
        const float u_part = dot_row_bf16(u.gate_up + (int64_t)(2 * n + 1) * hidden,
                                          x, hidden, threadIdx.x, blockDim.x);
        const float up = block_reduce_ordered(u_part, smem);
        __syncthreads();

        if (threadIdx.x == 0) {
            h[n] = (g / (1.f + __expf(-g))) * up;   // SiLU(gate) * up
        }
        __syncthreads();
    }
}

// down projection, scaled by the routing weight, accumulated into this XCD's
// partial. The reduce task later sums the 8 partials in a fixed order — no
// float atomics, so the result is bitwise reproducible (§4).
__device__ inline void expert_down(
        const ExpertUnit& u, const float* __restrict__ h,
        float* __restrict__ out, int hidden, int worker, int n_workers,
        float* smem) {
    for (int n = worker; n < hidden; n += n_workers) {
        const float part = dot_row_bf16(u.down + (int64_t)n * u.inter, h, u.inter,
                                        threadIdx.x, blockDim.x);
        const float sum = block_reduce_ordered(part, smem);
        if (threadIdx.x == 0) out[n] = u.weight * sum;
        __syncthreads();
    }
}

// Layer 0's dense MLP reuses the same machinery, but h is 10944 wide and does
// not fit an XCD-local buffer, so gate_up -> down costs one extra global event
// (§3). Keeping the same helpers means layer 0 needs no separate kernel path.
__device__ inline void dense_gate_up(
        const __hip_bfloat16* __restrict__ gate_up, const float* __restrict__ x,
        float* __restrict__ h, int hidden, int inter, int xcd, int n_xcds,
        int worker, int n_workers, float* smem) {
    const int per_xcd = (inter + n_xcds - 1) / n_xcds;
    const int begin = xcd * per_xcd;
    const int end = min(inter, begin + per_xcd);

    for (int n = begin + worker; n < end; n += n_workers) {
        const float g_part = dot_row_bf16(gate_up + (int64_t)(2 * n) * hidden,
                                          x, hidden, threadIdx.x, blockDim.x);
        const float g = block_reduce_ordered(g_part, smem);
        __syncthreads();
        const float u_part = dot_row_bf16(gate_up + (int64_t)(2 * n + 1) * hidden,
                                          x, hidden, threadIdx.x, blockDim.x);
        const float up = block_reduce_ordered(u_part, smem);
        __syncthreads();
        if (threadIdx.x == 0) h[n] = (g / (1.f + __expf(-g))) * up;
        __syncthreads();
    }
}

}  // namespace fleet
