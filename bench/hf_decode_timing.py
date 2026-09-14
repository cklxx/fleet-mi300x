#!/usr/bin/env python3
"""HuggingFace eager decode timing: the stock-framework baseline on the same box.

Same prompt (1,024 tokens), same model, same bf16 greedy decode as
src/host/reference_run.py, but timed: prefill once, then `--tokens` decode
steps through the KV cache, each step bracketed by torch.cuda.synchronize.
Reports median / mean ms per token and writes results/baseline_hf.json.

This is a *framework* baseline (eager PyTorch, one kernel launch per op), not
a SOTA number; it bounds what "no custom kernel" costs on this hardware.

    python3 bench/hf_decode_timing.py --model ~/models/dsv2-lite-base \
        --golden build/golden_tokens.txt --json results/baseline_hf.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "host"))
from reference_run import build_prompt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=1, help="untimed full runs first")
    ap.add_argument("--golden", type=Path, default=None,
                    help="golden_tokens.txt to compare the greedy tokens against")
    ap.add_argument("--json", type=Path, default=Path("results/baseline_hf.json"))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    print(f"loading {a.model} (bf16, eager attention)")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager").to(a.device).eval()
    print(f"  loaded in {time.time() - t0:.1f}s")
    ids = build_prompt(tok, a.context).to(a.device)

    def mask(n):
        return torch.ones(1, n, dtype=torch.long, device=a.device)

    def run(timed: bool):
        times, gen = [], []
        with torch.no_grad():
            torch.cuda.synchronize()
            tp = time.perf_counter()
            out = model(ids, attention_mask=mask(ids.shape[1]), use_cache=True)
            torch.cuda.synchronize()
            prefill_ms = (time.perf_counter() - tp) * 1e3
            past = out.past_key_values
            t_id = int(torch.argmax(out.logits[0, -1].float()))
            gen.append(t_id)
            next_id = torch.tensor([[t_id]], device=a.device)
            for step in range(a.tokens):
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                o = model(next_id, attention_mask=mask(ids.shape[1] + step + 1),
                          past_key_values=past, use_cache=True)
                t_id = int(torch.argmax(o.logits[0, -1].float()))
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t1) * 1e3)
                past = o.past_key_values
                next_id = torch.tensor([[t_id]], device=a.device)
                gen.append(t_id)
        return prefill_ms, times, gen

    for i in range(a.warmup):
        print(f"warm-up run {i + 1}")
        run(False)
    print(f"timed run: prefill {a.context} tokens, then {a.tokens} decode steps")
    prefill_ms, times, gen = run(True)
    med, mean = statistics.median(times), statistics.fmean(times)
    print(f"  prefill {prefill_ms:.1f} ms")
    print(f"  decode: median {med:.3f} ms/token  mean {mean:.3f} ms/token  "
          f"({1000.0 / mean:.1f} tok/s)  min {min(times):.3f}  max {max(times):.3f}")

    match = None
    if a.golden and a.golden.exists():
        golden = [int(x) for x in a.golden.read_text().split()]
        n = min(len(golden), len(gen))
        match = sum(1 for i in range(n) if golden[i] == gen[i])
        print(f"  tokens vs {a.golden}: {match}/{n} equal"
              + ("" if match == n else "  (eager HF is itself the reference; a mismatch here means a nondeterministic op)"))

    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps({
        "framework": "transformers-eager",
        "version": f"transformers {transformers.__version__}, torch {torch.__version__}",
        "device": torch.cuda.get_device_name(0),
        "dtype": "bfloat16",
        "ms_per_token_median": round(med, 3),
        "ms_per_token_mean": round(mean, 3),
        "tok_s": round(1000.0 / mean, 2),
        "prefill_ms": round(prefill_ms, 1),
        "context": a.context,
        "tokens": a.tokens,
        "golden_match": match,
        "tokens_out": gen,
    }, indent=2) + "\n")
    print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
