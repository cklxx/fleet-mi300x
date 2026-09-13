// Register-streamed GEMV — the task body behind every large weight read.
//
// docs/design.md §3 maps q_proj‖kv_a, o_proj, the experts and lm_head onto
// Chiplet-tasks: the N dimension is split across the 8 XCDs, then across the 37
// workers of each XCD, K is chunked, and accumulation is fp32. At batch 1 there
// is no L2 reuse to win — m_tiles is 1 — so the gain is dispatch-count
// reduction and, above all, keeping enough loads in flight.
//
// §12: weights go HBM -> VGPR directly with global_load_dwordx4; gfx942's LDS
// DMA moves only 4 B/lane, so staging through LDS would need 4x the load
// instructions for the same bytes. Little's law sets the target depth:
//
//     18 GB/s per CU (5.3 TB/s / 296 workers) x ~1 us load-to-use ~= 18 KB
//
// which is why kStreamDepth defaults to 8 dwordx4 per wave with 4 waves; D1
// microbenchmark (c) measures where the knee actually is and this constant
// follows the measurement rather than the estimate.
#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace fleet {

// Loads in flight per wave before the first s_waitcnt. Overridden by the
// measured optimum from bench/microbench.hip (c).
#ifndef FLEET_STREAM_DEPTH
#define FLEET_STREAM_DEPTH 8
#endif
constexpr int kStreamDepth = FLEET_STREAM_DEPTH;

// bf16 pairs arrive as uint32; unpacking in registers costs no memory traffic.
__device__ __forceinline__ void unpack_bf16x2(uint32_t p, float& lo, float& hi) {
    lo = __uint_as_float((p & 0xFFFFu) << 16);
    hi = __uint_as_float(p & 0xFFFF0000u);
}

// One row of a bf16 [N, K] matrix times an fp32 [K] vector, fp32 accumulate.
// K is a multiple of 8 for every tensor in this model (smallest is 512).
__device__ __forceinline__ float dot_row_bf16(
        const __hip_bfloat16* __restrict__ row, const float* __restrict__ x,
        int K, int lane, int lanes) {
    const uint4* __restrict__ r4 = reinterpret_cast<const uint4*>(row);
    const int n4 = K / 8;              // 8 bf16 per uint4
    float acc = 0.f;

    int i = lane;
    uint4 buf[kStreamDepth];
    while (i + (kStreamDepth - 1) * lanes < n4) {
#pragma unroll
        for (int d = 0; d < kStreamDepth; ++d) buf[d] = r4[i + d * lanes];  // issue
#pragma unroll
        for (int d = 0; d < kStreamDepth; ++d) {                            // consume
            const int base = (i + d * lanes) * 8;
            float a, b;
            unpack_bf16x2(buf[d].x, a, b); acc += a * x[base + 0] + b * x[base + 1];
            unpack_bf16x2(buf[d].y, a, b); acc += a * x[base + 2] + b * x[base + 3];
            unpack_bf16x2(buf[d].z, a, b); acc += a * x[base + 4] + b * x[base + 5];
            unpack_bf16x2(buf[d].w, a, b); acc += a * x[base + 6] + b * x[base + 7];
        }
        i += kStreamDepth * lanes;
    }
    for (; i < n4; i += lanes) {
        const uint4 v = r4[i];
        const int base = i * 8;
        float a, b;
        unpack_bf16x2(v.x, a, b); acc += a * x[base + 0] + b * x[base + 1];
        unpack_bf16x2(v.y, a, b); acc += a * x[base + 2] + b * x[base + 3];
        unpack_bf16x2(v.z, a, b); acc += a * x[base + 4] + b * x[base + 5];
        unpack_bf16x2(v.w, a, b); acc += a * x[base + 6] + b * x[base + 7];
    }
    return acc;
}

// Reduce one row's partial sums across the workgroup in a fixed order.
// Fixed order matters: §4 forbids float atomics so that the whole decode is
// bitwise reproducible, which is what makes the determinism check in §6 real.
__device__ __forceinline__ float block_reduce_ordered(float v, float* smem) {
    smem[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    return smem[0];
}

// Chiplet-task GEMV: y[n_begin..n_end) = W[n_begin..n_end, :] @ x
//
// The N range is this XCD's slice; workers stride over rows inside it, one row
// per workgroup pass, all 256 threads cooperating on that row's K.
__device__ inline void gemv_chiplet(
        const __hip_bfloat16* __restrict__ w, const float* __restrict__ x,
        float* __restrict__ y, int N, int K,
        int xcd, int n_xcds, int worker, int n_workers, float* smem) {
    const int per_xcd = (N + n_xcds - 1) / n_xcds;
    const int n_begin = xcd * per_xcd;
    const int n_end = min(N, n_begin + per_xcd);

    for (int n = n_begin + worker; n < n_end; n += n_workers) {
        const float part = dot_row_bf16(w + (int64_t)n * K, x, K,
                                        threadIdx.x, blockDim.x);
        const float sum = block_reduce_ordered(part, smem);
        if (threadIdx.x == 0) y[n] = sum;
        __syncthreads();
    }
}

// Same, but with a fused epilogue. Used for:
//   * q_proj‖kv_a — RMSNorm recomputed in the prologue (§3 fusion 1)
//   * o_proj      — residual add folded in, saving a pass over x
//   * gate_up     — SiLU(gate) * up folded in (§3, wavefront-task fusion)
enum GemvEpilogue { EPI_NONE = 0, EPI_RESIDUAL = 1, EPI_SILU_MUL = 2 };

__device__ inline void gemv_chiplet_epi(
        const __hip_bfloat16* __restrict__ w, const float* __restrict__ x,
        float* __restrict__ y, const float* __restrict__ residual,
        int N, int K, int xcd, int n_xcds, int worker, int n_workers,
        GemvEpilogue epi, float* smem) {
    const int per_xcd = (N + n_xcds - 1) / n_xcds;
    const int n_begin = xcd * per_xcd;
    const int n_end = min(N, n_begin + per_xcd);

    for (int n = n_begin + worker; n < n_end; n += n_workers) {
        const float part = dot_row_bf16(w + (int64_t)n * K, x, K,
                                        threadIdx.x, blockDim.x);
        const float sum = block_reduce_ordered(part, smem);
        if (threadIdx.x == 0) {
            switch (epi) {
                case EPI_RESIDUAL:
                    // Round through bf16 at the residual point so the hidden
                    // state matches the reference's bf16 state exactly (§5).
                    y[n] = __bfloat162float(__float2bfloat16(residual[n] + sum));
                    break;
                case EPI_SILU_MUL: {
                    // gate_up is laid out [gate; up] interleaved by row (§12),
                    // so the paired row is N/2 away and already in cache.
                    const float g = sum;
                    y[n] = g / (1.f + __expf(-g));   // up multiplied by caller
                    break;
                }
                default:
                    y[n] = sum;
            }
        }
        __syncthreads();
    }
}

}  // namespace fleet
