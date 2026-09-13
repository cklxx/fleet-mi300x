#!/usr/bin/env python3
"""Absorbed MLA must equal the naive materialised path, bit-for-bit in intent.

This test needs no model download and no HF: it builds random weights at the
real per-head shapes and checks that folding W_UK into the query and W_UV into
the output produces the same attention result as materialising K and V.

If this fails, every later boundary fails for the same reason, so it runs first.

    python3 tests/test_absorbed_equivalence.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "host"))

from reference_decode import (  # noqa: E402
    ModelConfig, apply_rope, softmax, yarn_cos_sin, yarn_get_mscale,
)

RNG = np.random.default_rng(0)


def naive_attention(cfg: ModelConfig, q_nope, q_pe, c, k_pe, kv_b):
    """Materialise K and V per head, then do ordinary attention."""
    H, Dk, Dv, L = cfg.heads, cfg.qk_nope, cfg.v_head, cfg.kv_lora_rank
    out = np.empty(H * Dv, dtype=np.float32)
    for h in range(H):
        base = h * (Dk + Dv)
        w_uk = kv_b[base: base + Dk]              # [Dk, L]
        w_uv = kv_b[base + Dk: base + Dk + Dv]    # [Dv, L]

        k_nope = c @ w_uk.T                        # [S, Dk]  materialised K
        v = c @ w_uv.T                             # [S, Dv]  materialised V

        s = (k_nope @ q_nope[h] + k_pe @ q_pe[h]) * cfg.softmax_scale
        p = softmax(s)
        out[h * Dv:(h + 1) * Dv] = p @ v
    return out


def absorbed_attention(cfg: ModelConfig, q_nope, q_pe, c, k_pe, kv_b):
    """Fold W_UK into q and W_UV into the output; the cache stays compressed."""
    H, Dk, Dv, L = cfg.heads, cfg.qk_nope, cfg.v_head, cfg.kv_lora_rank
    out = np.empty(H * Dv, dtype=np.float32)
    for h in range(H):
        base = h * (Dk + Dv)
        w_uk = kv_b[base: base + Dk]
        w_uv = kv_b[base + Dk: base + Dk + Dv]

        q_c = q_nope[h] @ w_uk                     # [L]
        s = (c @ q_c + k_pe @ q_pe[h]) * cfg.softmax_scale
        p = softmax(s)
        o_c = p @ c                                # [L]
        out[h * Dv:(h + 1) * Dv] = w_uv @ o_c
    return out


def test_absorbed_equals_naive() -> bool:
    cfg = ModelConfig(hidden=2048, heads=16, kv_lora_rank=512, qk_nope=128,
                      qk_rope=64, v_head=128, moe_inter=1408, n_routed=64, top_k=6)
    S = 37
    q_nope = RNG.standard_normal((cfg.heads, cfg.qk_nope), dtype=np.float32) * 0.05
    q_pe = RNG.standard_normal((cfg.heads, cfg.qk_rope), dtype=np.float32) * 0.05
    c = RNG.standard_normal((S, cfg.kv_lora_rank), dtype=np.float32) * 0.05
    k_pe = RNG.standard_normal((S, cfg.qk_rope), dtype=np.float32) * 0.05
    kv_b = RNG.standard_normal(
        (cfg.heads * (cfg.qk_nope + cfg.v_head), cfg.kv_lora_rank),
        dtype=np.float32) * 0.02

    a = naive_attention(cfg, q_nope, q_pe, c, k_pe, kv_b)
    b = absorbed_attention(cfg, q_nope, q_pe, c, k_pe, kv_b)

    rel = np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-9)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    # The two paths are algebraically identical, so the only difference is fp32
    # rounding: eps is 1.19e-7 and each output accumulates over 512 terms, so
    # ~1e-6 relative is the floor here, not a tolerance to be tightened.
    # (docs/design.md §6 asks for cosine >= 0.9999 at boundaries carrying bf16.)
    ok = rel <= 1e-5 and cos >= 1 - 1e-6
    print(f"  absorbed vs naive:   max rel {rel:.3e}  cosine {cos:.9f}  "
          f"[{'PASS' if ok else 'FAIL'}]")
    return ok


def test_cache_traffic_ratio() -> bool:
    """The point of absorbing: bytes read per position drop by design."""
    cfg = ModelConfig(hidden=2048, heads=16, kv_lora_rank=512, qk_nope=128,
                      qk_rope=64, v_head=128, moe_inter=1408, n_routed=64, top_k=6)
    absorbed = cfg.kv_lora_rank + cfg.qk_rope                       # 576, shared
    naive = cfg.heads * (cfg.qk_nope + cfg.v_head)                  # 4096, per head
    ratio = naive / absorbed
    ok = abs(ratio - 7.111) < 0.01
    print(f"  cache row: absorbed {absorbed} vs naive {naive} floats "
          f"-> {ratio:.2f}x less  [{'PASS' if ok else 'FAIL'}]")
    return ok


def test_rope_interleave() -> bool:
    """DeepseekV2 interleaves before rotate_half; plain RoPE gives a different
    answer. This guards against silently implementing the standard form."""
    d, S = 64, 8
    cos, sin = yarn_cos_sin(d, S, base=1e4, scale=40.0, original_max=4096,
                            beta_fast=32, beta_slow=1, mscale=0.707,
                            mscale_all_dim=0.707)
    x = RNG.standard_normal((1, d), dtype=np.float32)
    pos = 5
    ds = apply_rope(x, cos[pos][None, :], sin[pos][None, :])

    # Standard (non-interleaved) RoPE for contrast.
    half = d // 2
    plain = x * cos[pos] + np.concatenate([-x[:, half:], x[:, :half]], axis=-1) * sin[pos]

    differs = np.max(np.abs(ds - plain)) > 1e-3
    norm_kept = abs(np.linalg.norm(ds) / np.linalg.norm(x) - 1.0)
    # YaRN scales cos/sin by mscale/mscale_all_dim = 1.0 here, so RoPE is a
    # rotation and must preserve the norm.
    ok = differs and norm_kept < 1e-4
    print(f"  rope interleave:     differs from plain {differs}, "
          f"norm drift {norm_kept:.2e}  [{'PASS' if ok else 'FAIL'}]")
    return ok


def test_softmax_scale() -> bool:
    """scale = q_head_dim**-0.5 * mscale**2, mscale from yarn_get_mscale."""
    cfg = ModelConfig(hidden=2048, heads=16, kv_lora_rank=512, qk_nope=128,
                      qk_rope=64, v_head=128, moe_inter=1408, n_routed=64, top_k=6)
    m = yarn_get_mscale(40.0, 0.707)
    expected = (192 ** -0.5) * m * m
    ok = abs(cfg.softmax_scale - expected) < 1e-12
    print(f"  softmax_scale:       {cfg.softmax_scale:.6f} "
          f"(mscale {m:.6f})  [{'PASS' if ok else 'FAIL'}]")
    return ok


def main() -> int:
    print("absorbed-MLA equivalence (no GPU, no model download)")
    results = [
        test_softmax_scale(),
        test_rope_interleave(),
        test_cache_traffic_ratio(),
        test_absorbed_equals_naive(),
    ]
    ok = all(results)
    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
