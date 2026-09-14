# Status — Fleet-style batch-1 decode, DeepSeek-Coder-V2-Lite-Base on MI300X

Design: [design.md](design.md) · Task: one MI300X, bf16, bs=1, 1024-token
context, 32 greedy tokens.

This file tracks what is done, what is verified, and what is known to be
missing — the task asks for the milestone reached and the remaining limitations
to be stated plainly, not just for code.

## Where this is (2026-09-14, first day on the MI300X)

**End to end works and matches HuggingFace.** On a Hot Aisle 1×MI300X VM
(ROCm 7.2, gfx942), the persistent kernel decodes 32 greedy tokens over the
1,024-token context, free-running (its own tokens fed back) and teacher-forced,
and every token equals HF's; the smallest top-2 logit margin in the run is
6.6, so this is not a lucky tie. On the first decode step all 27 layers are
inside the §6 gate against HF's per-layer states (max rel ≤ 1.5e-2, cosine
≥ 0.9995; layer 1, the first MoE layer, is at 3.1e-3 / 0.999993). The
required milestone — one MoE layer through the Fleet path — is therefore met
for every MoE layer, and the stretch goal (full model, e2e) as well.
Raw numbers: [`results/`](../results/).

| Measured | Value |
|---|---|
| Tokens matching HF greedy, free-running / teacher-forced | 32/32 and 32/32 |
| Layers inside the §6 gate on step 0 | 27 of 27 |
| Per-token latency, median / p95 (1 launch per token), split-KV ×8 graph | **5.46 ms / 5.51 ms (171 tok/s)** free-running; 5.41 ms teacher-forced; the first version of the day was 22.05 ms |
| Protocol only (`--smoke`, all 805 events, no task bodies) | 1.17 ms per token |
| Cross-XCD event, idle / under a 1.46 TB/s stream | 1.44 µs / 6.0 µs (§9 was 2–4 µs) |
| Cross-XCD payload visibility (4 waves store, thread 0 releases) | 0 stale words in 16.4 M |
| Streamed read bandwidth, best depth | 4.25 TB/s at depth 8 (80% of peak) |
| Kernel resources | 253 VGPRs, 0 AGPRs, 24 KB LDS, 1 wave/SIMD (the design point), 169 SGPR spills, 80 B/lane scratch |
| KV-cache conversion vs HF's own cache | K_nope, K_rope, V all exactly 0 error, 27 layers |

**From 22 ms to 6.5 ms in four steps, each attributed by the per-task trace**
(`results/trace_d2_summary.txt` before, `results/trace_d4_summary.txt` after):

| Step | Per token | What the trace said and what changed |
|---|---|---|
| first run | 22.05 ms | MoE layer critical path ~810 µs: attention 296 µs/task, router 371 µs |
| router top-k mask in registers | 13.7 ms | `bool taken[64]` was in scratch memory and thread 0 paid a memory round trip per element; a 64-bit mask made it 68 µs |
| q-absorb streamed by rows; GEMV depth 8 | 11.6 ms | the attention prologue walked 128 rows per output column with one dependent load each; streaming W_UK rows made attention 203 µs. Depth 8 changed nothing: with a dozen rows per worker the GEMV tasks are latency-bound, not bandwidth-bound |
| split-KV graph (4 chunks/head, 64 tasks/layer) | 7.65 ms | attention 61 µs/task; the `taskgraph_d4.bin` from D0, no kernel change |
| prologues unrolled; router softmax/top-k as wave reductions | 6.50 ms | `stage_vector` / RMSNorm loads all in flight; router 31 µs; MoE layer critical path ~227 µs |
| GEMV tails batched, 8 chunks/head split-KV | 5.35 ms | with depth 8 and K = 2048 every row had been going through a one-load-at-a-time tail loop (so "depth 8" never happened); now every partial batch is issued in full. Attention 37 µs/task on `taskgraph_d8.bin` |
| 16 loads in flight per lane in every GEMV shape; direct polling A/B | **5.46 ms** | no gain: the expert GEMVs already stream at ~2.4 TB/s aggregate and the small ones are round-trip bound. Workers polling the global counters directly instead of the scheduler mirror is *slower* (5.8 ms): 296 fabric pollers cost more than the mirror hop, which settles §4's open question in favour of the scheduler |

Where the remaining 5.4 ms goes (per MoE layer, ~190 µs; `results/trace_d8_mirror_summary.txt`):
gate_up 39, attention 38, router 20, down 20, merge 20, q/kv_a 16, o_proj 12,
reduce 5, and ~6 global events at ~6 µs each. The layer's bytes at the
measured 4.25 TB/s are 39 µs, so the layer runs at ~20% of the bandwidth
ceiling; per token that is 5.4 ms against a 1.2 ms byte floor and a 1.2 ms
protocol cost. The levers left are §12's cross-task prefetch (the small
GEMVs pay 2–4 round trips each that could overlap the preceding event
wait), fusing merge into attention, and fewer, larger tasks for q/kv_a and
o_proj so each worker streams more than a dozen rows. The design's
2.5–3.5 ms target was not reached in the day on the machine.

## Review pass before GPU time (2026-09-13)

A full read of the implementation before booking the GPU found that the first
version could not have run at all, and would have burned hours looking like a
hang. Everything below is fixed; the "Now" column says which local check
would catch a regression, and where none can (device-only behaviour), says so.
A second, independent review pass (Codex) over the fixed tree then found the
items in the section after this one. Numbered for reference from the commit
message.

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
| 11 | Shared-expert half offset dropped the ×2 for interleaved gate/up rows | `expert.h` | `half * 2 * rows * hidden`; `test_expert_addressing.py` executes the packed layout and fails with the old offsets |
| 12 | Shared-expert `down` rows were strided by 1408, the true leading dimension is 2816 | `expert.h` | `ExpertUnit.down_ld`; same test |
| 13 | `struct.pack` used `H` for `layer`, and embed/lm_head/argmax have `layer = -1` — `--emit` raised on the very first task | `taskgraph.py` | `layer` is `int16`; `index` is `int32` (33k descriptors); layout test packs both |
| 14 | `__builtin_amdgcn_buffer_wbl2` and `__builtin_amdgcn_s_setprio` are not clang builtins; the microbench and the scheduler would not have compiled | `microbench.hip`, `fleet_runtime.h` | Release fence (which *is* `buffer_wbl2` on gfx942) and inline `s_setprio`; caught by the local HIP-mode parse |
| 15 | The GEMV put all 256 threads on one row: with K = 2048 that is one load per lane and a block reduction per row, so the "depth-8 streaming" loop never executed | `gemv.h` | One wave per row pair, 8 loads in flight per lane, shuffle reductions, no barrier in the row loop; attention likewise went from a block reduction per position to one wave per position |
| 16 | `hipMalloc(&kv.data, …)` cannot deduce through a `__restrict__` member; `kPartialStride` was invisible to the launcher | `fleet_launch.hip` | Both caught by the local parse |

Also fixed on the way: `expert_h` was indexed with stride `2 × moe_inter` but
allocated for nothing (no allocation existed); `q_proj` and `kv_a` are now
packed as one fused tensor rather than relying on the alignment padding
happening to be zero; the routing weight is multiplied by
`routed_scaling_factor` as in the reference.

### Second review pass (Codex, independent), what it found and what was done

| Finding | Done |
|---|---|
| The scheduler relayed counters with relaxed loads and stores, so the producer's release never formally synchronised with the worker's acquire on a *different* atomic | Scheduler acquires after reading a changed counter and releases before storing the mirror: a complete fence-to-fence chain |
| The kernel kept fp32 where HF rounds to bf16: RMSNorm output (before and after γ), every linear output, the residual add, gate/up/SiLU products, logits before argmax. Routing and argmax ties could land on the other side | All of those roundings implemented (`EPI_BF16`, `EPI_RESIDUAL = bf16(x + bf16(sum))`, `bf16_round` in the norms) and mirrored in `reference_decode.py` (`ModelConfig.bf16`); `test_reference_vs_hf.py` now runs a second pass against HF in bf16 |
| `golden.npz` "first decode step" hidden states were the last *prefill* position | Captured from the forward that consumes the prefill's token |
| Golden tokens picked with unstable `argsort`/`topk`; HF's greedy is `argmax` (lowest index on ties) | `torch.argmax`; top-k only for margins |
| 32 launches but 31 reference outputs, and exit 0 regardless | Reference writes `decode + 1` tokens; the launcher exits 0 only if every launch was compared and matched |
| Cache verification skipped all 64 RoPE columns and checked the fp32 tensor, not what the launcher loads; `FAIL` exited 0 | Verifies `K_nope`, `K_rope` and `V` from the bf16-rounded cache; non-zero exit |
| RoPE in the converter ran in fp32 where HF runs bf16 | Runs in the model's dtype through HF's own module; prefill rows are bit-identical to HF's |
| Cross-XCD microbench never checked its two blocks were on different XCDs | Both blocks record `HW_REG_XCC_ID`; the host flags a same-XCD pair |
| Shared-expert addressing had no execution test; the simulation let workers read global counters directly | `test_expert_addressing.py` (packed layout → slot resolution → 8-unit phase vs reference, sabotage checked); the simulation now reads a per-pass stale mirror |
| Seven statements in `design.md` were stale or described unimplemented mechanisms (W_UK redundancy 6.3 MB not 0.5 MB; q-column ownership; scheduler protocol; last-worker flush and cache modifiers; memory table; §10 tooling) | Corrected in place; unimplemented items are marked *planned* / *not implemented* |
| `setup_env.sh` downloaded the model before the checks and skipped the HF comparison | Reordered: checks, build, graphs, smoke test, *then* the download |
| Not adopted: "the event microbench should model fan-in, relay and payload" | `fleet_decode --smoke` runs the real protocol (296 producers, relay, all 805 events) per token; the microbench measures one hop on purpose |

### Third pass (Hermes), four items, all adopted

| Finding | Done |
|---|---|
| The one semantic the design rests on was untested: whether four waves' stores are published by *thread 0's* release fence after `__syncthreads()`. (a) only counts events | microbench **(d)**: producer's 4 waves write 32 KB, barrier, thread 0 releases and signals; consumer on another XCD acquires (thread 0) and verifies every word, 2,000 rounds. Runs first; a single stale word fails the whole benchmark |
| The scheduler polled all 805 event slots, of which 640 are XCD-local and never touch `global_events`: 80% of its agent-scope reads were wasted fabric traffic competing with the weight stream | `RuntimeState.global_event_ids` (165 ids, built by the launcher from the descriptors); the scheduler loops over that list only |
| (a) and (b) measured an idle fabric: 302 workgroups exit immediately, the real kernel has 296 workers streaming HBM | microbench **(e)**: the same ping-pong while the other 302 workgroups stream a 2 GiB buffer; this is the number for design.md §9 |
| `hip_syntax_check.sh` never parsed the `#if defined(__gfx942__)` branch (the XCC_ID asm); placement was only checked after a paid cooperative launch | `-D__gfx942__=1` in the parse; a pre-flight probe launch histograms `kGrid` workgroups by `HW_REG_XCC_ID` and refuses to continue unless it is 8 × 38 |

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
| `bench/microbench.hip` | D1 (d) cross-XCD payload visibility under the kernel's exact fence placement; (a) cross-XCD event cost, cached vs uncached counters; (e) the same under a 302-workgroup HBM stream; (b) same-XCD cost; (c) streamed read bandwidth vs depth; XCC_ID checks on every pair | Same parse, 0 errors; never run |
| `src/host/pack_weights.py` | Flat bf16 blob, 256-B aligned, fused q‖kv_a, interleaved gate/up, `.manifest` the launcher parses | Syntax and CLI only; needs the checkpoint |
| `scripts/setup_env.sh` | One-shot environment build on the MI300X: the seven local tests, hipcc, task graphs, the protocol smoke test, and only then the 31 GB download and the packing | Written; unrun |
| `tests/test_expert_addressing.py` | Builds the packer's byte layout, resolves the 8 units as `expert.h` does, runs the kernel's arithmetic with HF's rounding points against `reference_decode.moe` | 7/7: routing exact, phase output within 4.1e-3 of the reference (bf16-ulp level), and the pre-review offsets produce a 37% error that the test reports |

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
4. `fleet_decode --teacher-force` — every step fed HF's token, so a mismatch
   is isolated to the step it appears in; then free-running decode.
5. The per-layer comparison runs on every non-smoke decode automatically: on
   token 0 the kernel copies each layer's output out, and the launcher prints
   max rel and cosine per layer against HF's first-decode-step states plus
   the count of consecutive layers inside the §6 gate. **The required
   milestone is one MoE layer (index ≥ 1) matching HF through the Fleet
   path** — that count is the evidence, with the token comparison on top.
