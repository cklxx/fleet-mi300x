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
written, the arithmetic validated against HuggingFace on CPU, every HIP source
parsed by a real clang front end in HIP mode, the synchronisation protocol
executed end to end in a simulator, and the build scripted.

## Review pass before GPU time (2026-09-13)

A full read of the implementation before booking the GPU found that the first
version could not have run at all, and would have burned hours looking like a
hang. Everything below is fixed and covered by a local check that fails if it
comes back. Numbered for reference from the commit message.

| # | Defect | Where | Now |
|---|---|---|---|
| 1 | The launcher never launched the kernel; `hipLaunchCooperativeKernel` existed only in comments, and `./fleet_decode` printed "ready" and exited 0 | `fleet_launch.hip` | Full launcher: loads weights/cache/golden, YaRN tables, per-token cooperative launch with timing and HF comparison; `test_kernel_interface.py` requires the launch call to be real code |
| 2 | `rt.epoch` was never assigned, so every wait target was `0 × producers = 0` and no wait ever waited | `fleet_launch.hip` | Set per token; test requires every `RuntimeState` field the kernel reads to be assigned by the launcher |
| 3 | The kernel waited with the *signal-side* scope and count of its own descriptor; the descriptor had no wait-side fields at all | `fleet_kernel.hip`, `taskgraph.py` | Descriptor carries `wait_scope` / `wait_count`, filled from the event table; layout test checks offsets; `test_queue_simulation.py` runs the protocol and proves the old bug is detected |
| 4 | "Chiplet-task ×8" was one descriptor per XCD while every body strided by 37 workers — 36/37 of every GEMV output was never written | `taskgraph.py`, all bodies | A Chiplet-task is emitted as one descriptor per worker per XCD (37/XCD); the row split is a pure function of `(xcd, worker, wave)`; events have 296 producers |
| 5 | `q_pe` was never rotated by RoPE (the reference rotates it, `reference_decode.py:205`) | `fleet_kernel.hip` | `q_absorb` applies the interleaved RoPE to `q_pe` in LDS |
| 6 | `act.x_norm` was used as concurrent scratch by all 16 attention tasks and all 16 merge tasks | `fleet_kernel.hip` | Every per-task scratch (normed x, cache row, `q_c`, `q_pe`, partials, merge vector) is in this workgroup's LDS; no global scratch is shared |
| 7 | Only head 0 wrote the new KV row while every head on every XCD read it back from the cache with no event | `attention.h` | Every task scores the current position from its own LDS copy of the row; the cache write is only for later tokens |
| 8 | `__threadfence_block()` used as a release for cross-CU counters | `fleet_runtime.h` | Agent-scope release/acquire fences and agent-scope atomics everywhere, per the LLVM gfx942 memory model; consumers acquire (the original had no acquire at all) |
| 9 | Scheduler/worker roles came from `blockIdx`, XCD identity from `HW_REG_XCC_ID`; an XCD with zero schedulers hangs forever | `fleet_runtime.h` | Roles are XCD-local tickets; a grid barrier checks each XCD got exactly 38 blocks and aborts with a decoded reason otherwise |
| 10 | `done_flag` fired when each XCD's worker 0 drained its own queue, and was never zeroed | `fleet_kernel.hip` | Schedulers stop when the graph's final event reaches this epoch |
| 11 | Shared-expert half offset dropped the ×2 for interleaved gate/up rows | `expert.h` | `half * 2 * rows * hidden` |
| 12 | Shared-expert `down` rows were strided by 1408, the true leading dimension is 2816 | `expert.h` | `ExpertUnit.down_ld` |
| 13 | `struct.pack` used `H` for `layer`, and embed/lm_head/argmax have `layer = -1` — `--emit` raised on the very first task | `taskgraph.py` | `layer` is `int16`; `index` is `int32` (33k descriptors); layout test packs both |
| 14 | `__builtin_amdgcn_buffer_wbl2` and `__builtin_amdgcn_s_setprio` are not clang builtins; the microbench and the scheduler would not have compiled | `microbench.hip`, `fleet_runtime.h` | Release fence (which *is* `buffer_wbl2` on gfx942) and inline `s_setprio`; caught by the local HIP-mode parse |
| 15 | The GEMV put all 256 threads on one row: with K = 2048 that is one load per lane and a block reduction per row, so the "depth-8 streaming" loop never executed | `gemv.h` | One wave per row pair, 8 loads in flight per lane, shuffle reductions, no barrier in the row loop; attention likewise went from a block reduction per position to one wave per position |
| 16 | `hipMalloc(&kv.data, …)` cannot deduce through a `__restrict__` member; `kPartialStride` was invisible to the launcher | `fleet_launch.hip` | Both caught by the local parse |

Also fixed on the way: `expert_h` was indexed with stride `2 × moe_inter` but
allocated for nothing (no allocation existed); `q_proj` and `kv_a` are now
packed as one fused tensor rather than relying on the alignment padding
happening to be zero; the routing weight is multiplied by
`routed_scaling_factor` as in the reference.

## Done and verified locally

| Component | What it does | Verified by |
|---|---|---|
| `src/host/model_analysis.py` | Per-token HBM byte accounting from `config.json` | Reproduces design.md §1 exactly: attention 27.53 MB/layer, one routed expert 17.30 MB, MoE layer 166.20 MB, lm_head 419.43 MB, **4.935 GB/token**, 31.41 GB resident, floor 0.93 ms @ 5.3 TB/s |
| `src/host/taskgraph.py` | Builds + validates the Fleet task DAG, fans Chiplet-tasks out to per-worker descriptors, emits them with an event-label sidecar | 66 logical tasks / 1,218 descriptors per MoE layer (kv_chunks=1), 114 / 1,266 with split-KV; 1,791 logical / 33,183 descriptors per token (3,087 / 34,479 with split-KV); 6 global + 24 XCD-local events per layer; per-worker queue length 112–113; XCD imbalance 0.1% |
| `tests/test_queue_simulation.py` | Executes the emitted queues under the kernel's protocol: 296 queues in order, monotonic counters, targets `epoch × wait_count`, XCD-local visibility, 3 epochs, forward/reverse/random worker order | 10/10 for both graphs; the sabotage case (old signal-side count) is reported as a hang |
| `src/host/reference_decode.py` | NumPy absorbed-MLA decode: the arithmetic each HIP task must reproduce | Boundary tests below |
| `src/host/kv_convert.py`, `reference_run.py` | prefill → decode cache `[27][1056][576]` bf16 (32.8 MB) and greedy tokens, written from the *same* HF prefill the launcher is compared against | Layout arithmetic matches §5; reconstruction check needs the model |
| `tests/test_absorbed_equivalence.py` | Absorbed MLA ≡ materialised K/V; RoPE interleave; softmax scale | 4/4 pass, max rel 4.6e-7 |
| `tests/test_reference_vs_hf.py` | NumPy reference vs HuggingFace on a tiny random config | 4/4: RMSNorm, RoPE table 6.0e-8, RoPE applied 2.5e-8, **absorbed attention 2.4e-7**, MoE top-k exact, MoE output 7.3e-8 |
| `src/runtime/fleet_runtime.h` | AOT queues, events with wait-side fields, XCD-local role tickets, grid-distribution check, wait timeouts → abort codes | Layout test 6/6; interface test 14/14; queue simulation 10/10 |
| `src/kernels/fleet_kernel.hip`, `gemv.h`, `attention.h`, `expert.h` | Persistent kernel and task bodies; LDS-only scratch; wave-per-row GEMV; wave-per-position attention | **Parsed in HIP language mode, device and host passes, by clang 14 with the AMDGPU backend** (`scripts/hip_syntax_check.sh`): 0 errors, 0 warnings at `-Wall -Wextra`. The compiler's own `__hip_atomic_*`, `__builtin_amdgcn_fence`, attributes and inline asm are exercised; only the runtime API is stubbed |
| `src/host/fleet_launch.hip` | Queues, event buffers, weights/cache/golden loading, YaRN tables, per-token cooperative launch, HF comparison, abort decoding, `--smoke` / `--teacher-force` / `--json` | Same parse, both passes, 0 errors |
| `bench/microbench.hip` | D1 (a) cross-XCD event cost with the runtime's exact protocol, cached vs uncached counters; (b) same-XCD cost, with an XCC_ID check that the pair really shares a chiplet; (c) streamed read bandwidth vs depth | Same parse, 0 errors; never run |
| `src/host/pack_weights.py` | Flat bf16 blob, 256-B aligned, fused q‖kv_a, interleaved gate/up, `.manifest` the launcher parses | Syntax and CLI only; needs the checkpoint |
| `scripts/setup_env.sh` | One-shot environment build on the MI300X; runs every local check first, then the protocol smoke test before any weights are needed | Written; unrun |

## Decisions the design text did not make, made here

- **Every cross-workgroup handshake is agent-scope.** MI300X L2 is per XCD.
  Producers release (`buffer_wbl2`), counters are agent-scope atomics, pollers
  use agent-scope atomic loads, consumers acquire (`buffer_inv`). This is what
  the LLVM AMDGPU memory model guarantees; the "fence-free XCD-local counter"
  and "L2-resident mirror" in design.md §4 are *optimisations to be measured*
  on D1, not assumptions the correctness of the kernel rests on. If D1 (b)
  shows the same-XCD hop is no cheaper than the cross-XCD one under this
  protocol, the scheduler mirror buys nothing and should be removed.
- **Chiplet-tasks are per-worker descriptors.** Nothing on the device
  broadcasts a task; the descriptor count grows to 33k (2 MB), which is read
  sequentially per worker and is irrelevant next to 4.9 GB of weights.
- **Failure is loud.** A wrong XCD distribution, a wait that never completes,
  or a grid barrier that does not fill all end the launch with a reason code
  the launcher prints with the event's label, instead of a hung GPU.
- **Split-KV (D4) is wired but the merge still runs after all chunks**; the
  tile-granular producer/consumer of §12 row 1 is not implemented.

## Known limitations, open risks

1. **Nothing has run on the target hardware.** Every latency and bandwidth
   figure is a prediction. design.md §9 marks the synchronisation-overhead row
   as low confidence; `--smoke` mode measures it directly (every event, no
   task bodies) before any weights are downloaded.
2. **The HIP sources have been parsed, not compiled.** The local check runs
   clang's front end for gfx90a (the newest gfx9 that clang 14 knows), so
   `#if defined(__gfx942__)` branches, register/LDS pressure, occupancy, and
   the semantics of the builtins and of `s_getreg_b32 HW_REG_XCC_ID` are
   still unverified. `setup_env.sh` compiles the microbench first so a failure
   there is diagnosed as a toolchain problem, and prints
   `-Rpass-analysis=kernel-resource-usage` for the kernel.
3. **Attention lane mapping is specialised** to `kv_lora = 512`, `qk_rope = 64`
   (one 16-byte load per lane per position). The launcher refuses other shapes.
4. **`transformers` 5.x vs the vendored modeling file.** `modeling_deepseek.py`
   imports `is_torch_fx_available`, removed in transformers 5. Locally this is
   stubbed; `setup_env.sh` pins `transformers>=4.39,<5`.
5. **The tiny-config comparison is structural, not numerical-at-scale.** It
   cannot catch bf16 accumulation drift at 16 heads / 64 experts / 1024 context;
   the golden-token comparison on device is what does.
6. **Prefill is out of scope** per the task; the cache conversion is untested
   against a real checkpoint until `kv_convert.py --verify` runs there.

## Next, in order

1. `bash scripts/setup_env.sh` on the MI300X — environment, model, first real
   `hipcc` compile, task graphs, and the **protocol smoke test** (`--smoke`):
   cooperative launch at grid 304, XCD role discovery, all 805 events per
   token, timed. This needs no weights and is the first number worth having.
2. `bench/microbench.hip` — replaces the §9 synchronisation and bandwidth
   estimates with measurements and decides the event scheme.
3. HF reference run → golden tokens and the converted cache.
4. `fleet_decode --teacher-force` — every step fed HF's token, so the first
   mismatching step names the layer to look at; then free-running decode.
5. If tokens diverge: per-layer hidden-state comparison against `golden.npz`
   (the states are captured; the per-layer dump path in the launcher is the
   next thing to write). **The required milestone is one MoE layer (index ≥ 1)
   matching HF through the Fleet path.**
