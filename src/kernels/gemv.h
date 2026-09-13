// Register-streamed GEMV — the task body behind every large weight read.
//
// docs/design.md §3 maps q_proj‖kv_a, o_proj, the experts and lm_head onto
// Chiplet-tasks: the N dimension is split across the 8 XCDs, then across the 37
// workers of each XCD, and accumulation is fp32. At batch 1 there is no L2
// reuse to win — m_tiles is 1 — so the gain is dispatch-count reduction and,
// above all, keeping enough loads in flight.
//
// Structure: one *wave* owns a row pair. Its 64 lanes stride the row in
// 16-byte pieces (8 bf16 each), issue kStreamDepth loads per row before
// consuming any, and reduce with a fixed-order shuffle tree. Nothing in the
// row loop touches LDS or a workgroup barrier, so the four waves of a
// workgroup stream four independent row pairs and the CU keeps
// 4 waves x 2 rows x kStreamDepth x 1 KB = 32 KB in flight — above the 18 KB
// Little's-law target in §12. (The previous version put all 256 threads on
// one row: with K = 2048 that is exactly one load per lane and a full block
// reduction per row, which never reaches streaming depth at all.)
//
// §12: weights go HBM -> VGPR directly with global_load_dwordx4; gfx942's LDS
// DMA moves only 4 B/lane, so staging through LDS would need 4x the load
// instructions for the same bytes.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

namespace fleet {

// Loads in flight per row per lane before the first s_waitcnt. Overridden by
// the measured optimum from bench/microbench.hip (c).
#ifndef FLEET_STREAM_DEPTH
#define FLEET_STREAM_DEPTH 4
#endif
constexpr int kStreamDepth = FLEET_STREAM_DEPTH;
constexpr int kWaveLanes = 64;

// bf16 pairs arrive as uint32; unpacking in registers costs no memory traffic.
__device__ __forceinline__ void unpack_bf16x2(uint32_t p, float& lo, float& hi) {
    lo = __uint_as_float((p & 0xFFFFu) << 16);
    hi = __uint_as_float(p & 0xFFFF0000u);
}

__device__ __forceinline__ float bf16_round(float v) {
    return __bfloat162float(__float2bfloat16(v));
}

__device__ __forceinline__ float silu(float g) {
    return g / (1.f + __expf(-g));
}

// Fixed-order butterfly: the same tree every time, so the sum is bitwise
// reproducible (§4 forbids float atomics for exactly this reason).
__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
    for (int off = kWaveLanes / 2; off; off >>= 1) v += __shfl_xor(v, off, kWaveLanes);
    return v;
}

__device__ __forceinline__ void fma8(const uint4& v, const float* __restrict__ x8,
                                     float& acc) {
    float a, b;
    unpack_bf16x2(v.x, a, b); acc += a * x8[0] + b * x8[1];
    unpack_bf16x2(v.y, a, b); acc += a * x8[2] + b * x8[3];
    unpack_bf16x2(v.z, a, b); acc += a * x8[4] + b * x8[5];
    unpack_bf16x2(v.w, a, b); acc += a * x8[6] + b * x8[7];
}

// R rows (1 or 2) of a bf16 [*, K] matrix times an fp32 [K] vector, one wave,
// fp32 accumulate; every lane returns the full sums. K is a multiple of 8 for
// every tensor in this model (smallest is 512).
template <int R>
__device__ __forceinline__ void wave_dot(const __hip_bfloat16* const* rows,
                                         const float* __restrict__ x, int K,
                                         float* sums) {
    const int lane = threadIdx.x & (kWaveLanes - 1);
    const int n4 = K / 8;
    const uint4* r4[R];
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        r4[r] = reinterpret_cast<const uint4*>(rows[r]);
        acc[r] = 0.f;
    }

    int i = lane;
    uint4 buf[R][kStreamDepth];
    while (i + (kStreamDepth - 1) * kWaveLanes < n4) {
#pragma unroll
        for (int d = 0; d < kStreamDepth; ++d) {           // issue everything
#pragma unroll
            for (int r = 0; r < R; ++r) buf[r][d] = r4[r][i + d * kWaveLanes];
        }
#pragma unroll
        for (int d = 0; d < kStreamDepth; ++d) {           // then consume
            const float* x8 = x + (i + d * kWaveLanes) * 8;
#pragma unroll
            for (int r = 0; r < R; ++r) fma8(buf[r][d], x8, acc[r]);
        }
        i += kStreamDepth * kWaveLanes;
    }
    for (; i < n4; i += kWaveLanes) {                      // tail, < depth
        const float* x8 = x + i * 8;
#pragma unroll
        for (int r = 0; r < R; ++r) fma8(r4[r][i], x8, acc[r]);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) sums[r] = wave_sum(acc[r]);
}

// Reduce one value across the workgroup in a fixed order. Used by the
// prologues (RMSNorm, kv-norm), never inside a GEMV row loop.
__device__ __forceinline__ float block_reduce_ordered(float v, float* smem) {
    smem[threadIdx.x] = v;
    __syncthreads();
    for (int s = (int)blockDim.x >> 1; s; s >>= 1) {
        if ((int)threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    const float out = smem[0];
    __syncthreads();   // everyone has read smem[0] before it is reused
    return out;
}

// Copy a vector into LDS so the row loop reads its operand from LDS instead
// of re-fetching it from L2 for every row.
__device__ __forceinline__ void stage_vector(const float* __restrict__ src,
                                             float* __restrict__ dst, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
    __syncthreads();
}

// ---------------------------------------------------------------- row ranges
//
// A Chiplet-task's rows [0, N) are split evenly over the XCDs, then over the
// workers of each XCD by interleaving (worker w owns n_begin + w, + w +
// n_workers, ...), and each wave of the worker takes every kWaves-th row
// pair. The split is a pure function of (xcd, worker, wave), so the host
// graph never has to carry row ranges.

struct RowSlice {
    int begin, end;   // this XCD's [begin, end) of N
};

__device__ __forceinline__ RowSlice xcd_rows(int N, int xcd, int n_xcds) {
    const int per = (N + n_xcds - 1) / n_xcds;
    RowSlice s;
    s.begin = xcd * per;
    s.end = min(N, s.begin + per);
    return s;
}

enum GemvEpilogue { EPI_NONE = 0, EPI_RESIDUAL = 1 };

// y[n] = W[n, :] . x  for this (xcd, worker)'s rows of [0, N), with x already
// wherever the caller wants it read from (LDS for the ≤ 2048-wide operands).
// EPI_RESIDUAL: y[n] = bf16(residual[n] + sum), rounding through bf16 at the
// residual point so the hidden state matches the reference's bf16 state (§5).
__device__ inline void gemv_rows(
        const __hip_bfloat16* __restrict__ w, int ld,
        const float* __restrict__ x, float* __restrict__ y,
        const float* __restrict__ residual, GemvEpilogue epi, float scale,
        int N, int K, int xcd, int n_xcds, int worker, int n_workers) {
    const RowSlice s = xcd_rows(N, xcd, n_xcds);
    const int first = s.begin + worker;
    if (first >= s.end) return;
    const int cnt = (s.end - first + n_workers - 1) / n_workers;   // rows owned
    const int wave = threadIdx.x / kWaveLanes;
    const int lane = threadIdx.x % kWaveLanes;

    for (int k = 2 * wave; k < cnt; k += 2 * kWaves) {
        const int r0 = first + k * n_workers;
        const bool two = (k + 1) < cnt;
        const int r1 = two ? r0 + n_workers : r0;
        const __hip_bfloat16* rows[2] = {w + (int64_t)r0 * ld, w + (int64_t)r1 * ld};
        float sums[2];
        if (two) wave_dot<2>(rows, x, K, sums);
        else     wave_dot<1>(rows, x, K, sums);
        if (lane == 0) {
            const int nr = two ? 2 : 1;
            for (int r = 0; r < nr; ++r) {
                const int n = r ? r1 : r0;
                const float v = sums[r] * scale;
                y[n] = (epi == EPI_RESIDUAL) ? bf16_round(residual[n] + v) : v;
            }
        }
    }
}

// h[n] = SiLU(gate[n] . x) * (up[n] . x) for this worker's n in [0, inter).
// gate/up rows are interleaved offline (§12): row 2n is gate_n, row 2n+1 is
// up_n, so the pair is exactly what one wave streams together and the product
// is formed in registers with no shuffle or LDS round trip.
__device__ inline void gemv_gate_up_rows(
        const __hip_bfloat16* __restrict__ gate_up, const float* __restrict__ x,
        float* __restrict__ h, int inter, int K, int xcd, int n_xcds,
        int worker, int n_workers) {
    const RowSlice s = xcd_rows(inter, xcd, n_xcds);
    const int first = s.begin + worker;
    if (first >= s.end) return;
    const int cnt = (s.end - first + n_workers - 1) / n_workers;
    const int wave = threadIdx.x / kWaveLanes;
    const int lane = threadIdx.x % kWaveLanes;

    for (int k = wave; k < cnt; k += kWaves) {
        const int n = first + k * n_workers;
        const __hip_bfloat16* rows[2] = {gate_up + (int64_t)(2 * n) * K,
                                         gate_up + (int64_t)(2 * n + 1) * K};
        float sums[2];
        wave_dot<2>(rows, x, K, sums);
        if (lane == 0) h[n] = silu(sums[0]) * sums[1];
    }
}

}  // namespace fleet
