#!/usr/bin/env python3
"""NumPy reference vs HuggingFace DeepseekV2, on a tiny random config.

This is the D0 exit criterion (docs/design.md §7): before any GPU time is spent,
the arithmetic the HIP tasks will implement must already match HF on CPU. A
tiny config keeps it to a second and needs no 31 GB download, while exercising
every shape relationship that matters — MLA head splits, the kv_b_proj row
layout, YaRN RoPE, and MoE routing with shared experts.

Boundaries are checked separately so a failure names the task that is wrong,
which is exactly how the GPU tests are structured later.

    python3 tests/test_reference_vs_hf.py
"""
from __future__ import annotations

import sys
import types
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "host"))

import torch  # noqa: E402

# The vendored modeling file targets transformers 4.x; 5.x dropped two feature
# probes it imports at module scope. Stubbing them is enough for numerics and
# is recorded in docs/ as a known environment limitation.
import transformers.utils as _u  # noqa: E402
import transformers.utils.import_utils as _iu  # noqa: E402
for _m in (_u, _iu):
    for _s in ("is_torch_fx_available", "is_flash_attn_2_available",
               "is_flash_attn_greater_or_equal_2_10"):
        if not hasattr(_m, _s):
            setattr(_m, _s, (lambda *a, **k: False))

from reference.configuration_deepseek import DeepseekV2Config  # noqa: E402
from reference.modeling_deepseek import (  # noqa: E402
    DeepseekV2Attention, DeepseekV2MoE, DeepseekV2RMSNorm,
)

import reference_decode as ref  # noqa: E402

TORCH_DTYPE = torch.float32
RNG = np.random.default_rng(1234)


def tiny_config() -> DeepseekV2Config:
    """Small but structurally identical to DeepSeek-Coder-V2-Lite-Base."""
    return DeepseekV2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=2,
        n_routed_experts=8,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        q_lora_rank=None,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        norm_topk_prob=False,
        routed_scaling_factor=1.0,
        scoring_func="softmax",
        topk_method="greedy",
        n_group=1,
        topk_group=1,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_scaling={"type": "yarn", "factor": 40.0,
                      "original_max_position_embeddings": 32,
                      "beta_fast": 32, "beta_slow": 1,
                      "mscale": 0.707, "mscale_all_dim": 0.707},
        max_position_embeddings=2048,
        attention_dropout=0.0,
        torch_dtype="float32",
    )


def cfg_to_ref(c: DeepseekV2Config) -> ref.ModelConfig:
    rs = c.rope_scaling
    return ref.ModelConfig(
        hidden=c.hidden_size, heads=c.num_attention_heads,
        kv_lora_rank=c.kv_lora_rank, qk_nope=c.qk_nope_head_dim,
        qk_rope=c.qk_rope_head_dim, v_head=c.v_head_dim,
        moe_inter=c.moe_intermediate_size, n_routed=c.n_routed_experts,
        top_k=c.num_experts_per_tok, rms_eps=c.rms_norm_eps,
        rope_theta=c.rope_theta, rope_factor=rs["factor"],
        rope_original_max=rs["original_max_position_embeddings"],
        beta_fast=rs["beta_fast"], beta_slow=rs["beta_slow"],
        mscale=rs["mscale"], mscale_all_dim=rs["mscale_all_dim"],
        routed_scaling=c.routed_scaling_factor,
        norm_topk_prob=c.norm_topk_prob,
    )


def np_of(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).numpy()


def compare(name: str, a: np.ndarray, b: np.ndarray,
            rel_tol: float, cos_tol: float) -> bool:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    denom = max(float(np.max(np.abs(b))), 1e-12)
    rel = float(np.max(np.abs(a - b))) / denom
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    ok = rel <= rel_tol and cos >= cos_tol
    print(f"  {name:<28} max rel {rel:9.2e}  cosine {cos:.8f}  "
          f"[{'PASS' if ok else 'FAIL'}]")
    return ok


# ---------------------------------------------------------------- boundaries

def test_rmsnorm(c) -> bool:
    mod = DeepseekV2RMSNorm(c.hidden_size, eps=c.rms_norm_eps).to(TORCH_DTYPE)
    torch.nn.init.uniform_(mod.weight, 0.5, 1.5)
    x = torch.randn(c.hidden_size, dtype=TORCH_DTYPE)
    got = ref.rms_norm(np_of(x), np_of(mod.weight), c.rms_norm_eps)
    want = np_of(mod(x))
    return compare("rmsnorm", got, want, 1e-6, 1 - 1e-9)


def test_rope(c) -> bool:
    """Our yarn_cos_sin + apply_rope must equal HF's rotary module."""
    from reference.modeling_deepseek import DeepseekV2YarnRotaryEmbedding
    rs = c.rope_scaling
    S, d = 24, c.qk_rope_head_dim
    mod = DeepseekV2YarnRotaryEmbedding(
        d, max_position_embeddings=c.max_position_embeddings,
        base=c.rope_theta, scaling_factor=rs["factor"],
        original_max_position_embeddings=rs["original_max_position_embeddings"],
        beta_fast=rs["beta_fast"], beta_slow=rs["beta_slow"],
        mscale=rs["mscale"], mscale_all_dim=rs["mscale_all_dim"])
    dummy = torch.zeros(1, 1, S, d, dtype=TORCH_DTYPE)
    cos_t, sin_t = mod(dummy, seq_len=S)

    rc = cfg_to_ref(c)
    cos, sin = ref.yarn_cos_sin(d, S, rc.rope_theta, rc.rope_factor,
                                rc.rope_original_max, rc.beta_fast, rc.beta_slow,
                                rc.mscale, rc.mscale_all_dim)
    ok_tab = compare("rope cos/sin table", cos, np_of(cos_t), 1e-6, 1 - 1e-9)

    # And the rotation itself, including the interleave step.
    pos = 7
    q = torch.randn(1, 2, 1, d, dtype=TORCH_DTYPE)
    k = torch.randn(1, 1, 1, d, dtype=TORCH_DTYPE)
    from reference.modeling_deepseek import apply_rotary_pos_emb
    q_hf, k_hf = apply_rotary_pos_emb(q, k, cos_t, sin_t,
                                      torch.tensor([[pos]]))
    q_np = ref.apply_rope(np_of(q)[0, :, 0, :], cos[pos][None, :], sin[pos][None, :])
    ok_rot = compare("rope applied", q_np, np_of(q_hf)[0, :, 0, :], 1e-5, 1 - 1e-9)
    return ok_tab and ok_rot


def _attn_weights(mod: DeepseekV2Attention) -> ref.LayerWeights:
    return ref.LayerWeights(
        input_layernorm=np.ones(mod.hidden_size, dtype=np.float32),
        q_proj=np_of(mod.q_proj.weight),
        kv_a_proj_with_mqa=np_of(mod.kv_a_proj_with_mqa.weight),
        kv_a_layernorm=np_of(mod.kv_a_layernorm.weight),
        kv_b_proj=np_of(mod.kv_b_proj.weight),
        o_proj=np_of(mod.o_proj.weight),
        post_attention_layernorm=np.ones(mod.hidden_size, dtype=np.float32),
    )


def test_attention(c) -> bool:
    """Absorbed decode at position S-1 vs HF's materialised path over S tokens.

    The numpy side walks the sequence one token at a time, appending to the
    compressed cache exactly as the kernel will; HF sees all S tokens at once.
    Comparing the last position therefore checks cache accumulation too.
    """
    torch.manual_seed(0)
    mod = DeepseekV2Attention(c, layer_idx=0).to(TORCH_DTYPE).eval()
    for p in mod.parameters():
        torch.nn.init.normal_(p, std=0.05)
    torch.nn.init.uniform_(mod.kv_a_layernorm.weight, 0.8, 1.2)

    S = 12
    x = torch.randn(1, S, c.hidden_size, dtype=TORCH_DTYPE) * 0.5
    mask = torch.full((1, 1, S, S), float("-inf"), dtype=TORCH_DTYPE).triu(1)
    pos_ids = torch.arange(S)[None, :]

    # HF's attention module starts at q_proj: input_layernorm lives in the
    # DecoderLayer. Our task folds that norm into the q/kv_a prologue (§3), so
    # the equivalent input for HF is the already-normed vector. Feeding raw x to
    # both sides would compare "normed" against "not normed" — a 1.8e-1 mismatch
    # that looks like a kernel bug but is a harness bug.
    x_normed = torch.from_numpy(
        ref.rms_norm(np_of(x), np.ones(c.hidden_size, dtype=np.float32),
                     c.rms_norm_eps)).to(TORCH_DTYPE)
    with torch.no_grad():
        hf_out, _, _ = mod(hidden_states=x_normed, attention_mask=mask,
                           position_ids=pos_ids)
    want = np_of(hf_out)[0, -1]

    rc = cfg_to_ref(c)
    w = _attn_weights(mod)
    cos, sin = ref.yarn_cos_sin(rc.qk_rope, S, rc.rope_theta, rc.rope_factor,
                                rc.rope_original_max, rc.beta_fast, rc.beta_slow,
                                rc.mscale, rc.mscale_all_dim)
    cache = np.zeros((S, rc.kv_lora_rank + rc.qk_rope), dtype=np.float32)
    xn = np_of(x)[0]
    got = None
    for t in range(S):
        got = ref.attention_absorbed(rc, w, xn[t], cache, t, cos, sin)
    return compare("attention (absorbed)", got, want, 2e-4, 1 - 1e-7)


def test_moe(c) -> bool:
    torch.manual_seed(1)
    mod = DeepseekV2MoE(c).to(TORCH_DTYPE).eval()
    for p in mod.parameters():
        torch.nn.init.normal_(p, std=0.08)

    x = torch.randn(1, 1, c.hidden_size, dtype=TORCH_DTYPE) * 0.5
    with torch.no_grad():
        want = np_of(mod(x))[0, 0]

    rc = cfg_to_ref(c)
    n_exp, moe_i, hid = c.n_routed_experts, c.moe_intermediate_size, c.hidden_size
    w = ref.LayerWeights(
        input_layernorm=np.ones(hid, dtype=np.float32),
        q_proj=np.zeros((1, 1), dtype=np.float32),
        kv_a_proj_with_mqa=np.zeros((1, 1), dtype=np.float32),
        kv_a_layernorm=np.zeros(1, dtype=np.float32),
        kv_b_proj=np.zeros((1, 1), dtype=np.float32),
        o_proj=np.zeros((1, 1), dtype=np.float32),
        post_attention_layernorm=np.ones(hid, dtype=np.float32),
        gate_weight=np_of(mod.gate.weight),
        experts_gate=np.stack([np_of(e.gate_proj.weight) for e in mod.experts]),
        experts_up=np.stack([np_of(e.up_proj.weight) for e in mod.experts]),
        experts_down=np.stack([np_of(e.down_proj.weight) for e in mod.experts]),
        shared_gate=np_of(mod.shared_experts.gate_proj.weight),
        shared_up=np_of(mod.shared_experts.up_proj.weight),
        shared_down=np_of(mod.shared_experts.down_proj.weight),
    )
    got = ref.moe(rc, w, np_of(x)[0, 0])

    # Routing must match exactly: a different expert set is not a tolerance issue.
    logits = w.gate_weight @ np_of(x)[0, 0]
    scores = ref.softmax(logits)
    ours = set(np.argpartition(-scores, rc.top_k - 1)[: rc.top_k].tolist())
    with torch.no_grad():
        idx_hf, _, _ = mod.gate(x)
    theirs = set(idx_hf.flatten().tolist())
    ok_route = ours == theirs
    print(f"  {'moe top-k ids':<28} ours {sorted(ours)} vs hf {sorted(theirs)}  "
          f"[{'PASS' if ok_route else 'FAIL'}]")
    return ok_route and compare("moe output", got, want, 2e-4, 1 - 1e-7)


def main() -> int:
    c = tiny_config()
    print(f"tiny config: hidden={c.hidden_size} heads={c.num_attention_heads} "
          f"kv_lora={c.kv_lora_rank} experts={c.n_routed_experts}/top-{c.num_experts_per_tok}")
    print("boundary checks (numpy reference vs HuggingFace, fp32 CPU):")
    results = [
        test_rmsnorm(c),
        test_rope(c),
        test_attention(c),
        test_moe(c),
    ]
    print(f"\n{sum(results)}/{len(results)} boundaries passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
