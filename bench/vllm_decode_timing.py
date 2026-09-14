#!/usr/bin/env python3
"""vLLM batch-1 greedy decode timing on the same prompt as reference_run.py.

Run inside the rocm/vllm container (see results/baseline_vllm.json for the
exact command). Decode cost is isolated as (time for `--tokens` new tokens
minus time for 1 new token) / (tokens - 1), each measured on the same
1,024-token prompt after a warm-up, so prefill and sampling of the first
token cancel out. vLLM's own per-request metrics are recorded when present.

    python3 bench/vllm_decode_timing.py --model /models/dsv2-lite-base \
        --json /work/results/baseline_vllm.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def build_prompt_ids(tokenizer, n_tokens: int) -> list[int]:
    """Same construction as src/host/reference_run.py::build_prompt."""
    text = ("def solve(n):\n    # compute the answer\n" * 400)
    ids = tokenizer(text).input_ids
    if len(ids) < n_tokens:
        ids = ids * (n_tokens // len(ids) + 1)
    return ids[:n_tokens]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--golden", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=Path("results/baseline_vllm.json"))
    a = ap.parse_args()

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(model=a.model, dtype="bfloat16", trust_remote_code=True,
              max_model_len=a.context + a.tokens + 8, gpu_memory_utilization=0.85,
              enforce_eager=False)
    tok = llm.get_tokenizer()
    ids = build_prompt_ids(tok, a.context)
    prompt = TokensPrompt(prompt_token_ids=ids)

    def gen(n_new: int):
        sp = SamplingParams(temperature=0.0, max_tokens=n_new, ignore_eos=True)
        t0 = time.perf_counter()
        out = llm.generate([prompt], sp, use_tqdm=False)[0]
        dt = (time.perf_counter() - t0) * 1e3
        return dt, list(out.outputs[0].token_ids), getattr(out, "metrics", None)

    gen(a.tokens)                                   # warm-up (graph capture etc.)
    gen(1)
    full, one, toks, metrics = [], [], None, None
    for _ in range(a.repeats):
        dt, toks, metrics = gen(a.tokens)
        full.append(dt)
        d1, _, _ = gen(1)
        one.append(d1)
    per_tok = [(f - o) / (a.tokens - 1) for f, o in zip(full, one)]
    med, mean = statistics.median(per_tok), statistics.fmean(per_tok)
    print(f"full({a.tokens} tokens) ms: {[round(x, 1) for x in full]}")
    print(f"one(1 token)  ms: {[round(x, 1) for x in one]}")
    print(f"decode: median {med:.3f} ms/token  mean {mean:.3f} ms/token  ({1000 / mean:.1f} tok/s)")

    metrics_out = None
    if metrics is not None:
        try:
            first = metrics.first_token_time - metrics.arrival_time
            total = metrics.finished_time - metrics.arrival_time
            metrics_out = {"first_token_s": first, "finished_s": total,
                           "decode_ms_per_token_from_metrics": (total - first) * 1e3 / (a.tokens - 1)}
            print(f"vLLM metrics: decode {metrics_out['decode_ms_per_token_from_metrics']:.3f} ms/token")
        except Exception as e:  # noqa: BLE001
            metrics_out = {"error": str(e)}

    match = None
    if a.golden and a.golden.exists():
        golden = [int(x) for x in a.golden.read_text().split()]
        # golden[0] is the prefill's output token = vLLM's first generated token
        n = min(len(golden), len(toks))
        match = sum(1 for i in range(n) if golden[i] == toks[i])
        print(f"tokens vs golden: {match}/{n} equal")

    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps({
        "framework": "vllm",
        "version": vllm.__version__,
        "dtype": "bfloat16",
        "ms_per_token_median": round(med, 3),
        "ms_per_token_mean": round(mean, 3),
        "tok_s": round(1000.0 / mean, 2),
        "method": "(t(32 tokens) - t(1 token)) / 31, batch 1, greedy, ignore_eos",
        "full_ms": full, "one_ms": one,
        "vllm_metrics": metrics_out,
        "context": a.context,
        "tokens": a.tokens,
        "golden_match": match,
        "tokens_out": toks,
    }, indent=2) + "\n")
    print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
