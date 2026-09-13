#!/usr/bin/env python3
"""HuggingFace reference run: the golden states every Fleet boundary is judged against.

Runs the stock DeepseekV2 implementation on the 1,024-token prompt and decodes
32 greedy tokens, saving:

  * per-layer hidden states for the first decode step  -> layer-boundary checks
  * the compressed KV rows captured at kv_a_proj_with_mqa -> Fleet cache
  * the 32 generated token ids and their top-2 logit margins -> end-to-end check

and, next to the .npz, the two files the C++ launcher consumes directly:
`fleet_cache.bin` (the converted cache, bf16, see kv_convert.py) and
`golden_tokens.txt` (the greedy token ids, one per line; the first is the
prefill's output and is the first decode input).

Prefill is explicitly out of scope for optimisation (task spec), so this is a
stock eager run; its latency is not a baseline and is not reported as one.

    python3 src/host/reference_run.py --model ~/models/dsv2-lite-base \
        --out build/golden.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kv_convert import CacheLayout, build_fleet_cache, write_fleet_cache_bin  # noqa: E402


def build_prompt(tokenizer, n_tokens: int) -> "torch.Tensor":
    """A fixed, reproducible prompt of exactly n_tokens."""
    import torch

    text = ("def solve(n):\n    # compute the answer\n" * 400)
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    if ids.numel() < n_tokens:
        reps = n_tokens // ids.numel() + 1
        ids = ids.repeat(reps)
    return ids[:n_tokens][None, :]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--out", type=Path, default=Path("build/golden.npz"))
    ap.add_argument("--device", default="cuda")  # ROCm reports as cuda
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    print(f"loading {a.model} (bf16)")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager").to(a.device).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    ids = build_prompt(tok, a.context).to(a.device)
    print(f"prompt: {ids.shape[1]} tokens")

    # ---- capture per-layer hidden states and the compressed KV rows
    hidden: dict[int, np.ndarray] = {}
    compressed: dict[int, np.ndarray] = {}
    handles = []

    def layer_hook(i):
        def h(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            hidden[i] = t.detach()[0, -1].float().cpu().numpy()
        return h

    def kv_hook(i):
        def h(_m, _inp, out):
            compressed[i] = out.detach()[0].float().cpu().numpy()
        return h

    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(layer_hook(i)))
        handles.append(layer.self_attn.kv_a_proj_with_mqa.register_forward_hook(kv_hook(i)))

    print("prefill + first decode step")
    with torch.no_grad():
        out = model(ids, use_cache=True)
    for h in handles:
        h.remove()

    first_logits = out.logits[0, -1].float().cpu().numpy()
    order = np.argsort(-first_logits)
    print(f"  first token {int(order[0])}, top-2 logit margin "
          f"{first_logits[order[0]] - first_logits[order[1]]:.4f}")

    # ---- greedy decode, no sampling, so the token sequence is a hard check
    print(f"decoding {a.decode} tokens greedily")
    gen_ids, margins = [], []
    past, cur = out.past_key_values, ids
    next_id = torch.tensor([[int(order[0])]], device=a.device)
    gen_ids.append(int(order[0]))
    margins.append(float(first_logits[order[0]] - first_logits[order[1]]))

    with torch.no_grad():
        for _ in range(a.decode - 1):
            o = model(next_id, past_key_values=past, use_cache=True)
            past = o.past_key_values
            lg = o.logits[0, -1].float()
            top2 = torch.topk(lg, 2)
            next_id = top2.indices[:1][None, :]
            gen_ids.append(int(top2.indices[0]))
            margins.append(float(top2.values[0] - top2.values[1]))

    print("  " + repr(tok.decode(gen_ids))[:160])

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out,
        hidden=np.stack([hidden[i] for i in range(n_layers)]),
        compressed_kv=np.stack([compressed[i] for i in range(n_layers)]),
        prompt_ids=ids[0].cpu().numpy(),
        token_ids=np.array(gen_ids, dtype=np.int64),
        margins=np.array(margins, dtype=np.float32),
        first_logits=first_logits,
    )
    meta = {
        "model": str(a.model), "context": a.context, "decode": a.decode,
        "layers": n_layers, "dtype": "bfloat16", "attn": "eager",
        "tokens": gen_ids, "min_margin": float(min(margins)),
    }
    a.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {a.out} and {a.out.with_suffix('.json')}")

    # Launcher inputs: the converted cache from *this* prefill, and the tokens.
    layout = CacheLayout(layers=n_layers, max_pos=a.context + a.decode,
                         kv_lora_rank=cfg.kv_lora_rank, qk_rope=cfg.qk_rope_head_dim)
    raw = torch.from_numpy(np.stack([compressed[i] for i in range(n_layers)]))
    cache = build_fleet_cache(model, raw, layout)
    write_fleet_cache_bin(cache, layout, a.out.parent / "fleet_cache.bin")
    (a.out.parent / "golden_tokens.txt").write_text(
        "".join(f"{t}\n" for t in gen_ids))
    print(f"wrote {a.out.parent / 'golden_tokens.txt'} ({len(gen_ids)} tokens)")
    print(f"  smallest top-2 margin over the run: {min(margins):.4f} "
          f"(a small margin here is where bf16 drift would flip a token)")


if __name__ == "__main__":
    main()
