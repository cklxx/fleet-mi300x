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

// Loads in flight per row per lane before the first s_waitcnt. Set from
// bench/microbench.hip (c) on the MI300X: depth 4 streamed 3.76 TB/s, depth 8
// 4.25 TB/s (80% of peak), depth 12 and 16 fell off again.
#ifndef FLEET_STREAM_DEPTH
#define FLEET_STREAM_DEPTH 8
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

// R rows of a bf16 [*, K] matrix times an fp32 [K] vector, one wave, fp32
// accumulate, D 16-byte chunks per row in flight per lane; every lane returns
// the full sums. K is a multiple of 8 for every tensor in this model (smallest
// is 512). R x D is the number of loads in flight per lane, kept at 8 by the
// callers: for K = 2048 a row is 4 chunks per lane, so 2 rows x 4 chunks; for
// K = 512 it is 1 chunk, so 8 rows x 1.
//
// The final partial batch is issued in full before any of it is consumed
// (predicated, unrolled): the first version's plain tail loop loaded one
// chunk at a time, and with depth 8 and K = 2048 *every* row went through
// that tail, which is why raising the depth changed nothing on the MI300X.
template <int R, int D>
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

    // Double-buffered: batch i+1 is issued before batch i is consumed, so
    // consecutive batches overlap their memory latency instead of paying it
    // serially (measured on the MI300X: with one buffer the expert GEMVs
    // streamed at ~2.4 TB/s against a 4.25 TB/s ceiling).
    uint4 bufA[R][D], bufB[R][D];
    auto issue = [&](uint4 (&buf)[R][D], int i) {
#pragma unroll
        for (int d = 0; d < D; ++d) {
            if (i + d * kWaveLanes < n4) {
#pragma unroll
                for (int r = 0; r < R; ++r) buf[r][d] = r4[r][i + d * kWaveLanes];
            }
        }
    };
    auto consume = [&](const uint4 (&buf)[R][D], int i) {
#pragma unroll
        for (int d = 0; d < D; ++d) {
            if (i + d * kWaveLanes < n4) {
                const float* x8 = x + (i + d * kWaveLanes) * 8;
#pragma unroll
                for (int r = 0; r < R; ++r) fma8(buf[r][d], x8, acc[r]);
            }
        }
    };
    const int step = D * kWaveLanes;
    int i = lane;
    issue(bufA, i);
    while (true) {
        const int j = i + step;
        if (j < n4) issue(bufB, j);
        consume(bufA, i);
        if (j >= n4) break;
        i = j + step;
        if (i < n4) issue(bufA, i);
        consume(bufB, j);
        if (i >= n4) break;
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
// of re-fetching it from L2 for every row. The stride is the compile-time
// block size and the loop is unrolled so the (up to 8) loads per thread are
// all in flight together instead of paying one round trip each.
constexpr int kBlockThreads = 256;

__device__ __forceinline__ void stage_vector(const float* __restrict__ src,
                                             float* __restrict__ dst, int n) {
#pragma unroll 8
    for (int i = threadIdx.x; i < n; i += kBlockThreads) dst[i] = src[i];
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

// bf16 boundaries (reference_decode.py with bf16=True): HuggingFace runs the
// model in bf16, so every linear layer's output is rounded to bf16 before it
// is used, and the residual add is a bf16 add. Reproducing those roundings is
// what keeps routing decisions and argmax ties on the same side as HF.
//   EPI_NONE      y = sum                       (router: HF's gate runs in fp32)
//   EPI_BF16      y = bf16(scale * sum)         (every other linear output)
//   EPI_RESIDUAL  y = bf16(residual + bf16(sum)) (o_proj, dense down)
enum GemvEpilogue { EPI_NONE = 0, EPI_BF16 = 1, EPI_RESIDUAL = 2 };

// y[n] = W[n, :] . x  for this (xcd, worker)'s rows of [0, N), with x already
// wherever the caller wants it read from (LDS for the ≤ 2048-wide operands).
// Each wave takes RPI of the worker's rows per iteration (a short row means
// more rows per iteration, so the loads in flight stay at 8 per lane).
template <int RPI, int D>
__device__ __forceinline__ void gemv_rows_impl(
        const __hip_bfloat16* __restrict__ w, int ld,
        const float* __restrict__ x, float* __restrict__ y,
        const float* __restrict__ residual, GemvEpilogue epi, float scale,
        int K, int first, int cnt, int n_workers) {
    const int wave = threadIdx.x / kWaveLanes;
    const int lane = threadIdx.x % kWaveLanes;
    for (int k = RPI * wave; k < cnt; k += RPI * kWaves) {
        const int nr = min(RPI, cnt - k);
        const __hip_bfloat16* rows[RPI];
#pragma unroll
        for (int r = 0; r < RPI; ++r) {   // past the end: repeat the last row
            rows[r] = w + (int64_t)(first + (k + min(r, nr - 1)) * n_workers) * ld;
        }
        float sums[RPI];
        wave_dot<RPI, D>(rows, x, K, sums);
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < RPI; ++r) {
                if (r < nr) {
                    const int n = first + (k + r) * n_workers;
                    const float v = sums[r];
                    switch (epi) {
                        case EPI_BF16:     y[n] = bf16_round(scale * bf16_round(v)); break;
                        case EPI_RESIDUAL: y[n] = bf16_round(residual[n] + bf16_round(v)); break;
                        default:           y[n] = v;
                    }
                }
            }
        }
    }
}

__device__ inline void gemv_rows(
        const __hip_bfloat16* __restrict__ w, int ld,
        const float* __restrict__ x, float* __restrict__ y,
        const float* __restrict__ residual, GemvEpilogue epi, float scale,
        int N, int K, int xcd, int n_xcds, int worker, int n_workers) {
    const RowSlice s = xcd_rows(N, xcd, n_xcds);
    const int first = s.begin + worker;
    if (first >= s.end) return;
    const int cnt = (s.end - first + n_workers - 1) / n_workers;   // rows owned
    const int n4 = K / 8;   // 16-byte chunks per row; 64 lanes take 64 at a time
    // 16 loads in flight per lane (64 KB per workgroup) for every shape: the
    // tasks own 7–350 rows each, so what they pay is round trips, not bytes.
    if (n4 <= kWaveLanes) {             // K <= 512:  8 rows x 1 chunk  (merge W_UV)
        gemv_rows_impl<8, 1>(w, ld, x, y, residual, epi, scale, K, first, cnt, n_workers);
    } else if (n4 <= 2 * kWaveLanes) {  // K <= 1024: 8 rows x 2 chunks
        gemv_rows_impl<8, 2>(w, ld, x, y, residual, epi, scale, K, first, cnt, n_workers);
    } else if (n4 <= 4 * kWaveLanes) {  // K <= 2048: 4 rows x 4 chunks (q/kv_a, o_proj,
        gemv_rows_impl<4, 4>(w, ld, x, y, residual, epi, scale, K, first, cnt, n_workers);  // down, lm_head)
    } else {                            // longer rows: 2 rows x kStreamDepth
        gemv_rows_impl<2, kStreamDepth>(w, ld, x, y, residual, epi, scale, K, first, cnt, n_workers);
    }
}

// h[n] = bf16(bf16(SiLU(bf16(gate[n] . x))) * bf16(up[n] . x)) for this
// worker's n in [0, inter) — the roundings are HF's (gate_proj, act_fn and
// up_proj each produce a bf16 tensor, and their product is bf16).
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

    // Two (gate, up) pairs per wave iteration: 4 rows x 4 chunks = 16 loads
    // in flight per lane, the same depth as gemv_rows.
    for (int k = 2 * wave; k < cnt; k += 2 * kWaves) {
        const int n0 = first + k * n_workers;
        const bool two = (k + 1) < cnt;
        const int n1 = two ? n0 + n_workers : n0;
        const __hip_bfloat16* rows[4] = {gate_up + (int64_t)(2 * n0) * K,
                                         gate_up + (int64_t)(2 * n0 + 1) * K,
                                         gate_up + (int64_t)(2 * n1) * K,
                                         gate_up + (int64_t)(2 * n1 + 1) * K};
        float sums[4];
        wave_dot<4, 4>(rows, x, K, sums);
        if (lane == 0) {
            h[n0] = bf16_round(bf16_round(silu(bf16_round(sums[0]))) * bf16_round(sums[1]));
            if (two) h[n1] = bf16_round(bf16_round(silu(bf16_round(sums[2]))) * bf16_round(sums[3]));
        }
    }
}

}  // namespace fleet
