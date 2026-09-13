// Types shared by the kernel and the host launcher.
//
// These were originally defined inside fleet_kernel.hip, which meant
// fleet_launch.hip had to forward-declare them — and a forward declaration
// cannot be passed by value, so the launch signature would not compile. Both
// sides now include this header, so there is exactly one definition and the
// kernel's parameter layout cannot drift from what the host packs.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>   // __hip_bfloat16 is not part of hip_runtime.h
#include <stdint.h>

namespace fleet {

// Compile-time bounds for the LDS buffers the task bodies use. The host
// launcher refuses to run a model whose dims exceed them (fleet_launch.hip),
// so the kernel never has to check.
constexpr int kMaxHidden = 2048;   // x / x_norm / o staged in LDS
constexpr int kMaxKvLora = 512;    // q_c, cache row, merge scratch
constexpr int kMaxQkRope = 64;     // q_pe, k_pe
constexpr int kMaxMoeInter = 1408; // expert h staged in LDS for down
constexpr int kMaxRouted = 64;     // router logits
constexpr int kMaxTopK = 8;
constexpr int kPartialStride = 2 + kMaxKvLora;   // attention partial: m, l, acc[512]

// Static model shape, read from build/weights.manifest by the host.
struct ModelDims {
    int hidden, heads, q_head_dim, qk_nope, qk_rope, v_head;
    int kv_lora, moe_inter, dense_inter, n_routed, top_k, vocab;
    int layers, first_k_dense, seq_len;   // seq_len = rows valid after this step
    float rms_eps, softmax_scale, routed_scaling;
};

// Weight pointers into the flat blob produced by src/host/pack_weights.py.
// Experts are addressed as base + id * expert_stride so the task queue stays
// immutable while routing is data-dependent (docs/design.md §3).
struct Weights {
    const __hip_bfloat16* const* layer_qkv;       // [3072 + 576, hidden] fused
    const __hip_bfloat16* const* layer_kv_b;
    const __hip_bfloat16* const* layer_o_proj;
    const __hip_bfloat16* const* layer_norm_in;
    const __hip_bfloat16* const* layer_norm_post;
    const __hip_bfloat16* const* layer_kv_a_norm;
    const __hip_bfloat16* const* layer_router;
    const __hip_bfloat16* const* layer_experts;   // base of the 64-expert block
    const __hip_bfloat16* const* layer_shared;
    const __hip_bfloat16* const* layer_dense;
    const __hip_bfloat16* embed;
    const __hip_bfloat16* lm_head;
    const __hip_bfloat16* final_norm;
    int64_t expert_stride;                        // elements between experts
};

// Per-token activations, allocated once (docs/design.md §5, under 1 MB total).
// Everything a task needs only for itself lives in LDS instead (x_norm for
// the GEMV prologues, the rebuilt cache row, q_c, q_pe): no two workgroups
// ever share scratch through global memory.
struct Activations {
    float* __restrict__ x;             // [hidden], bf16-rounded at residuals
    float* __restrict__ x_norm;        // [hidden] post-attention norm (router task)
    float* __restrict__ q;             // [heads * q_head_dim + kv_lora + qk_rope]
    float* __restrict__ kv_a;          // = q + heads * q_head_dim, raw pre-norm
    float* __restrict__ attn_partial;  // [heads][chunks][2 + kv_lora] m, l, acc
    float* __restrict__ o;             // [heads * v_head]
    float* __restrict__ expert_h;      // [XCDs][moe_inter], or [dense_inter]
    float* __restrict__ expert_out;    // [XCDs][hidden] partials to reduce
    float* __restrict__ logits;        // [vocab]
    int32_t* __restrict__ topk_ids;    // [top_k]
    float* __restrict__ topk_w;        // [top_k]
    int32_t* __restrict__ next_token;
    const float* __restrict__ cos;     // [qk_rope] for this position; YaRN is
    const float* __restrict__ sin;     // position-only, so the host computes it
};

// Absorbed-MLA cache: one compressed row per position, shared by all heads.
struct KVCache {
    __hip_bfloat16* __restrict__ data;  // [layers][max_pos][kv_lora + qk_rope]
    int max_pos;
    int row;
};

}  // namespace fleet
