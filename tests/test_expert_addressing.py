#!/usr/bin/env python3
"""The packed expert layout, the slot resolution and the 8-unit MoE phase,
executed on the CPU against the reference.

pack_weights.py flattens experts into `[gate;up interleaved | down]` blocks at
a fixed stride and the shared expert into one merged block; expert.h turns a
descriptor slot into pointers (routed: base + id * stride; shared: half h owns
gate/up row pairs [h*rows, (h+1)*rows) and columns [h*rows, (h+1)*rows) of
every down row) and the kernel sums the 8 unit outputs in a fixed order. Two of
those offsets were wrong once and nothing executed them. This test builds the
exact byte layout the packer writes, resolves the 8 units the way expert.h
does, runs the kernel's arithmetic (HF's bf16 rounding points included) and
compares with reference_decode.moe. A sabotage run with the old offsets must
fail, or the test proves nothing.

    python3 tests/test_expert_addressing.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "host"))

import reference_decode as ref  # noqa: E402

XCDS = 8
R = ref.bf16_round


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {label:<44} [{'PASS' if ok else 'FAIL'}]{('  ' + detail) if detail else ''}")
    return ok


def interleave(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    out = np.empty((2 * gate.shape[0], gate.shape[1]), dtype=np.float32)
    out[0::2], out[1::2] = gate, up
    return out


def pack(w: ref.LayerWeights, hidden: int, inter: int):
    """pack_weights.py's layout: experts block + stride, shared block."""
    blocks = []
    for e in range(w.experts_gate.shape[0]):
        gu = interleave(w.experts_gate[e], w.experts_up[e]).reshape(-1)
        blocks.append(np.concatenate([gu, w.experts_down[e].reshape(-1)]))
    stride = blocks[0].size
    experts = np.concatenate(blocks)
    shared = np.concatenate([interleave(w.shared_gate, w.shared_up).reshape(-1),
                             w.shared_down.reshape(-1)])
    return experts, stride, shared


def resolve_unit(slot: int, experts: np.ndarray, stride: int, shared: np.ndarray,
                 topk_ids, topk_w, hidden: int, moe_inter: int, shared_inter: int,
                 old_bug: bool = False):
    """Mirror of expert.h resolve_unit, returning (gate_up view, down view, ld,
    weight, inter). Views are 1-D slices of the flat blobs, exactly like the
    device pointers."""
    if slot >= 0:
        base = topk_ids[slot] * stride
        gate_up = experts[base: base + 2 * moe_inter * hidden]
        down = experts[base + 2 * moe_inter * hidden: base + stride]
        return gate_up, down, moe_inter, topk_w[slot], moe_inter
    half = -slot - 1
    rows = shared_inter // 2
    if old_bug:   # the offsets before the review: missing x2, ld = rows
        gu_off, ld = half * rows * hidden, rows
    else:
        gu_off, ld = half * 2 * rows * hidden, shared_inter
    gate_up = shared[gu_off: gu_off + 2 * rows * hidden]
    down = shared[2 * shared_inter * hidden + half * rows:]
    return gate_up, down, ld, 1.0, rows


def unit_forward(gate_up, down, ld, weight, inter, hidden, x_n):
    """expert_gate_up + expert_down as gemv.h computes them."""
    h = np.empty(inter, dtype=np.float32)
    for n in range(inter):
        g = gate_up[(2 * n) * hidden: (2 * n + 1) * hidden] @ x_n
        u = gate_up[(2 * n + 1) * hidden: (2 * n + 2) * hidden] @ x_n
        h[n] = R(R(ref.silu(R(g))) * R(u))
    out = np.empty(hidden, dtype=np.float32)
    for n in range(hidden):
        out[n] = R(weight * R(down[n * ld: n * ld + inter] @ h))   # EPI_BF16 with scale
    return out


def moe_phase(cfg, w, x, x_n, old_bug=False):
    """NORM_ROUTER (top-k) + 8 units + REDUCE, as the kernel runs them."""
    hidden, moe_inter = cfg.hidden, cfg.moe_inter
    experts, stride, shared = pack(w, hidden, moe_inter)
    scores = ref.softmax(w.gate_weight @ x_n)
    ids, ws, taken = [], [], set()
    for _ in range(cfg.top_k):                         # kernel's selection scan
        best = max((e for e in range(cfg.n_routed) if e not in taken),
                   key=lambda e: (scores[e], -e))
        taken.add(best); ids.append(best); ws.append(scores[best] * cfg.routed_scaling)
    partials = []
    for xcd in range(XCDS):
        slot = xcd if xcd < cfg.top_k else -(xcd - cfg.top_k + 1)
        unit = resolve_unit(slot, experts, stride, shared, ids, ws, hidden,
                            moe_inter, 2 * moe_inter, old_bug)
        partials.append(unit_forward(*unit, hidden, x_n))
    acc = np.zeros(hidden, dtype=np.float32)
    for p in partials:                                  # fixed order, fp32
        acc += p
    return R(x + R(acc)), sorted(ids)


def main() -> int:
    rng = np.random.default_rng(7)
    hidden, moe_inter, n_routed, top_k = 64, 16, 8, 6      # 6 routed + 2 shared halves
    cfg = ref.ModelConfig(hidden=hidden, heads=4, kv_lora_rank=16, qk_nope=16,
                          qk_rope=8, v_head=16, moe_inter=moe_inter,
                          n_routed=n_routed, top_k=top_k, bf16=True)
    f = lambda *s: R(rng.normal(0, 0.08, size=s).astype(np.float32))  # noqa: E731
    w = ref.LayerWeights(
        input_layernorm=np.ones(hidden, np.float32), q_proj=np.zeros((1, 1), np.float32),
        kv_a_proj_with_mqa=np.zeros((1, 1), np.float32), kv_a_layernorm=np.zeros(1, np.float32),
        kv_b_proj=np.zeros((1, 1), np.float32), o_proj=np.zeros((1, 1), np.float32),
        post_attention_layernorm=f(hidden) + 1.0,
        gate_weight=f(n_routed, hidden),
        experts_gate=f(n_routed, moe_inter, hidden), experts_up=f(n_routed, moe_inter, hidden),
        experts_down=f(n_routed, hidden, moe_inter),
        shared_gate=f(2 * moe_inter, hidden), shared_up=f(2 * moe_inter, hidden),
        shared_down=f(hidden, 2 * moe_inter),
    )
    results = []

    # resolve_unit() above mirrors expert.h. Pin the four addresses that the
    # review got wrong once already, so the mirror cannot drift into agreeing
    # with itself.
    EH = re.sub(r"\s+", " ", (ROOT / "src" / "kernels" / "expert.h").read_text())
    for label, frag in (
            ("shared half keeps the x2 for interleaved rows", "half * 2 * rows * hidden"),
            ("shared down is strided by the full shared_inter", "u.down_ld = shared_inter;"),
            ("a routed expert is base + id * stride", "(int64_t)id * expert_stride"),
            ("routed down follows the gate/up block", "base + (int64_t)2 * moe_inter * hidden")):
        results.append(check(label, frag in EH, "" if frag in EH else frag))
    for trial in range(3):
        x = R(rng.normal(0, 0.5, size=hidden).astype(np.float32))
        x_n = ref.rms_norm(x, w.post_attention_layernorm, cfg.rms_eps, bf16=True)
        want = R(x + ref.moe(cfg, w, x_n))
        got, ids = moe_phase(cfg, w, x, x_n)
        scores = ref.softmax(w.gate_weight @ x_n)
        ref_ids = sorted(np.argpartition(-scores, top_k - 1)[:top_k].tolist())
        rel = float(np.max(np.abs(got - want)) / max(np.max(np.abs(want)), 1e-12))
        results.append(check(f"trial {trial}: routed ids match reference", ids == ref_ids,
                             str(ids)))
        # The kernel rounds two shared halves separately where HF rounds one
        # sum; that is a bf16-ulp-level gap, nothing larger.
        results.append(check(f"trial {trial}: 8-unit MoE phase == reference",
                             rel <= 1e-2, f"max rel {rel:.2e}"))
        if trial == 0:
            bad, _ = moe_phase(cfg, w, x, x_n, old_bug=True)
            rel_bad = float(np.max(np.abs(bad - want)) / max(np.max(np.abs(want)), 1e-12))
            results.append(check("old shared-half offsets are detected", rel_bad > 1e-2,
                                 f"max rel {rel_bad:.2e}"))
    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
