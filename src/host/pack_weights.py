#!/usr/bin/env python3
"""Pack the HF checkpoint into the flat layout the persistent kernel reads.

The kernel does pointer arithmetic, not tensor lookup: an expert is
`base + id * stride`, a layer's weights are contiguous, and gate/up rows are
interleaved so a wave computing SiLU(gate)*up touches one cache line pair
instead of two distant ones (docs/design.md §5, §12).

Doing this offline matters for two reasons. It keeps the task queue immutable —
the kernel resolves `topk_ids[k]` into an address without any host round trip —
and it moves the repack cost out of the decode path entirely, where the task
spec requires one-time conversions to be excluded from measured latency.

    python3 src/host/pack_weights.py --model ~/models/dsv2-lite-base \
        --out build/weights.bin
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

# bf16 stays bf16: the kernel widens in registers, so packing costs no traffic.
DTYPE = np.dtype(np.uint16)


class Packer:
    """Appends tensors to one flat file and records their byte offsets."""

    def __init__(self, path: Path):
        self.f = path.open("wb")
        self.offsets: dict[str, dict] = {}
        self.pos = 0

    def _write(self, name: str, arr: np.ndarray) -> int:
        assert arr.dtype == DTYPE, f"{name}: expected bf16-as-uint16, got {arr.dtype}"
        # 256-byte align every tensor: dwordx4 loads want the row base aligned,
        # and a misaligned expert base would cost a split transaction per row.
        pad = (-self.pos) % 256
        if pad:
            self.f.write(b"\0" * pad)
            self.pos += pad
        off = self.pos
        self.f.write(arr.tobytes(order="C"))
        self.pos += arr.nbytes
        self.offsets[name] = {"offset": off, "shape": list(arr.shape),
                              "bytes": int(arr.nbytes)}
        return off

    def write(self, name: str, t) -> int:
        return self._write(name, to_bf16_u16(t))

    def close(self) -> None:
        self.f.close()


def to_bf16_u16(t) -> np.ndarray:
    """View a torch bf16 tensor as uint16 without converting through fp32."""
    import torch
    if t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
    return t.detach().contiguous().view(torch.uint16).cpu().numpy()


def interleave_gate_up(gate, up):
    """Rows [g0, u0, g1, u1, ...] so a lane holds adjacent gate/up pairs.

    §12: with this layout SiLU(gate)*up is computed in registers — no
    cross-lane shuffle, no LDS round trip — and the epilogue writes h directly.
    """
    import torch
    assert gate.shape == up.shape, f"{gate.shape} vs {up.shape}"
    inter, hidden = gate.shape
    out = torch.empty((2 * inter, hidden), dtype=gate.dtype)
    out[0::2] = gate
    out[1::2] = up
    return out


def pack(model_dir: Path, out: Path) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    print(f"loading {model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    cfg = model.config
    p = Packer(out)

    p.write("embed", model.model.embed_tokens.weight)
    p.write("final_norm", model.model.norm.weight)
    p.write("lm_head", model.lm_head.weight)

    expert_strides = []
    for i, layer in enumerate(model.model.layers):
        a = layer.self_attn
        p.write(f"L{i}.norm_in", layer.input_layernorm.weight)
        p.write(f"L{i}.norm_post", layer.post_attention_layernorm.weight)
        p.write(f"L{i}.q_proj", a.q_proj.weight)
        p.write(f"L{i}.kv_a", a.kv_a_proj_with_mqa.weight)
        p.write(f"L{i}.kv_a_norm", a.kv_a_layernorm.weight)
        p.write(f"L{i}.kv_b", a.kv_b_proj.weight)
        p.write(f"L{i}.o_proj", a.o_proj.weight)

        mlp = layer.mlp
        if hasattr(mlp, "experts"):
            p.write(f"L{i}.router", mlp.gate.weight)
            # One contiguous block of 64 experts so `base + id*stride` works.
            blocks, stride = [], None
            for e in mlp.experts:
                gu = interleave_gate_up(e.gate_proj.weight, e.up_proj.weight)
                blk = torch.cat([gu.reshape(-1),
                                 e.down_proj.weight.reshape(-1)])
                if stride is None:
                    stride = blk.numel()
                assert blk.numel() == stride, "experts must be uniform"
                blocks.append(blk)
            p.write(f"L{i}.experts", torch.cat(blocks))
            expert_strides.append(int(stride))

            s = mlp.shared_experts
            gu = interleave_gate_up(s.gate_proj.weight, s.up_proj.weight)
            p.write(f"L{i}.shared",
                    torch.cat([gu.reshape(-1), s.down_proj.weight.reshape(-1)]))
        else:
            gu = interleave_gate_up(mlp.gate_proj.weight, mlp.up_proj.weight)
            p.write(f"L{i}.dense",
                    torch.cat([gu.reshape(-1), mlp.down_proj.weight.reshape(-1)]))
        print(f"  layer {i:2d} packed, {p.pos/1e9:.2f} GB")

    p.close()
    strides = set(expert_strides)
    assert len(strides) <= 1, f"expert stride differs across layers: {strides}"

    meta = {
        "bytes": p.pos,
        "dtype": "bfloat16",
        "expert_stride_elems": expert_strides[0] if expert_strides else 0,
        "hidden": cfg.hidden_size,
        "layers": cfg.num_hidden_layers,
        "first_k_dense": cfg.first_k_dense_replace,
        "moe_inter": cfg.moe_intermediate_size,
        "dense_inter": cfg.intermediate_size,
        "n_routed": cfg.n_routed_experts,
        "n_shared": cfg.n_shared_experts,
        "top_k": cfg.num_experts_per_tok,
        "heads": cfg.num_attention_heads,
        "kv_lora": cfg.kv_lora_rank,
        "qk_nope": cfg.qk_nope_head_dim,
        "qk_rope": cfg.qk_rope_head_dim,
        "v_head": cfg.v_head_dim,
        "vocab": cfg.vocab_size,
        "rms_eps": cfg.rms_norm_eps,
        "gate_up_layout": "interleaved [g0,u0,g1,u1,...]",
        "tensors": p.offsets,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {out} ({p.pos/1e9:.2f} GB) and {out.with_suffix('.json')}")
    print(f"  expert stride: {meta['expert_stride_elems']} elems "
          f"({meta['expert_stride_elems']*2/1e6:.1f} MB) — matches design.md §1's "
          f"17.3 MB per routed expert")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("build/weights.bin"))
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    pack(a.model, a.out)


if __name__ == "__main__":
    main()
