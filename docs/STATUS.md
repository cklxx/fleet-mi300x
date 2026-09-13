# Status — Fleet-style batch-1 decode, DeepSeek-Coder-V2-Lite-Base on MI300X

Design: [design.md](design.md) · Task: one MI300X, bf16, bs=1, 1024-token
context, 32 greedy tokens.

This file tracks what is done, what is verified, and what is known to be
missing — the task asks for the milestone reached and the remaining limitations
to be stated plainly, not just for code.

## Where this is

All work so far is **D0: local, no GPU**. Nothing has run on an MI300X yet;
every GPU-dependent number in design.md §9 is still an estimate and is labelled
as such. The GPU budget is ~24–30 hours, so the plan is to arrive with the code
written, the arithmetic already validated against HuggingFace on CPU, and the
build scripted.

## Done and verified locally

| Component | What it does | Verified by |
|---|---|---|
| `src/host/model_analysis.py` | Per-token HBM byte accounting from `config.json` | Reproduces design.md §1 exactly: attention 27.53 MB/layer, one routed expert 17.30 MB, MoE layer 166.20 MB, lm_head 419.43 MB, **4.935 GB/token**, 31.41 GB resident, floor 0.93 ms @ 5.3 TB/s |
| `src/host/taskgraph.py` | Builds + validates the Fleet task DAG, assigns AOT worker queues, emits packed descriptors | 66 tasks/MoE layer (kv_chunks=1), 114 (kv_chunks=4), 3087 total, 6 global events per layer, DAG validation passes, XCD imbalance 0.8% |
| `src/host/reference_decode.py` | NumPy absorbed-MLA decode: the arithmetic each HIP task must reproduce | Boundary tests below |
| `src/host/kv_convert.py` | prefill → decode cache interface, `[27][1056][576]` = 32.8 MB | Layout arithmetic matches §5; reconstruction check needs the model |
| `tests/test_absorbed_equivalence.py` | Absorbed MLA ≡ materialised K/V; RoPE interleave; softmax scale | 4/4 pass, max rel 4.6e-7 (fp32 accumulation floor) |
| `tests/test_reference_vs_hf.py` | NumPy reference vs HuggingFace on a tiny random config | 4/4: RMSNorm 0.0, RoPE table 6.0e-8, RoPE applied 3.5e-8, **absorbed attention 2.4e-7**, MoE top-k exact, MoE output 7.3e-8 |
| `src/runtime/fleet_runtime.h` | AOT queues, hierarchical events, XCD discovery, behind `fetch_task` / `signal_event` / `xcd_barrier` | Descriptor layout verified against the host packer, 5/5 |
| `tests/test_descriptor_layout.py` | Host `struct.pack` vs device C struct, field by field | 5/5: 64 B both sides, names and order identical, values round-trip, `-1` stays signed |
| `src/kernels/fleet_kernel.hip` | Persistent kernel: scheduler/worker split, AOT queue walk, dispatch for all 13 task kinds | Interface-checked statically, 8/8 |
| `src/kernels/gemv.h` | Register-streamed GEMV with fused epilogues (residual, SiLU·up); depth-8 dwordx4 in flight per §12's Little's-law estimate | Symbols and call sites checked; depth follows D1 (c) once measured |
| `src/kernels/attention.h` | Absorbed MLA: kv-post + q-absorb prologues, flash-decoding chunks, merge + W_UV | Same; the interleaved-RoPE hazard is covered by the numpy tests |
| `src/kernels/expert.h` | MoE units with indirect expert index, XCD-local gate_up→down, dense MLP path | Same |
| `src/runtime/fleet_types.h` | `ModelDims`/`Weights`/`Activations`/`KVCache` shared by kernel and launcher | Extracted after finding the launcher forward-declared them and passed them by value — an incomplete type cannot be passed by value, so that would not have compiled |
| `src/host/fleet_launch.hip` | CSR per-worker queues, event buffers, YaRN tables, occupancy check before launch | Unrun |
| `src/host/pack_weights.py` | Flat weight blob, 256-B aligned, gate/up rows interleaved per §12 | Syntax and CLI only; needs the checkpoint |
| `tests/test_kernel_interface.py` | Kernel vs runtime header, without a compiler | 5/5: 7 runtime symbols declared, all 13 task kinds handled, 6 descriptor fields exist, grid constants consistent, no float atomics |
| `README.md` | Build and run instructions (task spec deliverable) | Every local command listed is executed and passes |
| `bench/microbench.hip` | D1 microbenchmarks (a) cross-XCD event round-trip, (b) XCD-local flag, (c) streamed GEMV bandwidth | Written; **never compiled** (no ROCm locally, and the local syntax-check harness is still blocked — see below) |
| `scripts/setup_env.sh` | One-shot environment build on the MI300X | Written; unrun |

### Why the task graph numbers matter

design.md §3 claims four fusions cut global events from 10 to 6 per MoE layer.
`taskgraph.py --report` is what makes that falsifiable rather than asserted: it
counts the events the generated graph actually contains and fails if any event
has a producer count the kernel would deadlock on.

One real fix came out of it: routing every single-task op (router, reduce,
embed, argmax) to XCD 0 produced a 25.3% task imbalance across chiplets. Those
tasks all signal global events, so their placement is free — rotating them by
layer index brought the imbalance to 0.8%.

## Known limitations, open risks

1. **Nothing has run on the target hardware.** Every latency and bandwidth
   figure is a prediction. design.md §9 marks the synchronisation-overhead row
   as low confidence; the D1 microbenchmarks exist specifically to replace it
   with a measurement, and that has not happened yet.
2. **The HIP sources have never been compiled — not even parsed.** There is no
   ROCm toolchain on the development machine, and the local fallback
   (`scripts/hip_syntax_check.sh`: host compiler + stub headers) is itself
   blocked: this machine's clang cannot resolve `<cstddef>` even for a minimal
   `.cpp` with no stubs and no flags, so the C++ standard library search path is
   broken independently of anything in this repo (CommandLineTools present,
   Xcode absent). The first real parse of `microbench.hip`, `fleet_kernel.hip`
   and `fleet_runtime.h` will therefore happen on the MI300X.
   Mitigation, since GPU wall-clock is the scarce resource: interface
   consistency between the kernel and the runtime header is checked statically
   from Python instead (`tests/test_descriptor_layout.py` for the wire format,
   plus a symbol/field cross-check), and `setup_env.sh` compiles the
   self-contained microbenchmarks first so a failure there is diagnosed as a
   toolchain problem rather than a kernel problem.
   Still unverifiable locally either way: `__builtin_amdgcn_*` semantics,
   inline asm (`s_getreg_b32 HW_REG_XCC_ID`), register/LDS pressure, and
   whether `hipExtMallocWithFlags(hipDeviceMallocUncached)` exists in the
   installed ROCm.
3. **`transformers` 5.x vs the vendored modeling file.** `modeling_deepseek.py`
   from the model repo imports `is_torch_fx_available` and friends, removed in
   transformers 5. Locally this is worked around with import stubs; on the GPU
   box `scripts/setup_env.sh` pins `transformers>=4.39,<5` instead, which is the
   configuration the results should be reproduced under.
4. **The tiny-config comparison is structural, not numerical-at-scale.** It uses
   4 heads and 8 experts to exercise the same shape relationships in a second.
   It cannot catch anything that only appears at 16 heads / 64 experts / 1024
   context — bf16 accumulation drift in particular.
5. **Prefill is out of scope** per the task, but the conversion in
   `kv_convert.py` is untested against a real checkpoint; the reconstruction
   check (K,V rebuilt from the compressed cache vs HF's own cache, ≤ 1e-2) runs
   only once the model is downloaded.

## Next, in order

1. `bash scripts/setup_env.sh` on the MI300X — environment, model, first real
   `hipcc` compile, task graph emitted. The model download dominates.
2. `bench/microbench.hip` — replaces the §9 synchronisation and bandwidth
   estimates with measurements, and decides between the two event schemes in §4.
3. HF reference run → golden hidden states and 32 greedy tokens.
4. Persistent-kernel smoke test: cooperative launch at grid=304, XCD discovery,
   one global event round-trip through the real scheduler/worker structure.
5. Attention block of layer 1 validated against the golden states, then the
   full MoE layer — **the milestone the task requires** (layer index ≥ 1, so
   expert routing and expert execution are both exercised).
