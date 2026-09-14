# Status — Fleet-style batch-1 decode, DeepSeek-Coder-V2-Lite-Base on MI300X

Design: [design.md](design.md) · Task: one MI300X, bf16, bs=1, 1024-token
context, 32 greedy tokens.

This file tracks what is done, what is verified, and what is known to be
missing — the task asks for the milestone reached and the remaining limitations
to be stated plainly, not just for code.

## Where this is (2026-09-15, after two days on the MI300X)

**End to end works, matches HuggingFace, and runs at 4.08 ms per token.** On a
Hot Aisle 1×MI300X VM (ROCm 7.2, gfx942), one cooperative launch decodes all
32 greedy tokens over the 1,024-token context; the argmax task of each token
feeds the next token's embed on the device. Free-running and teacher-forced,
every token equals HF's (smallest top-2 margin in the run 6.6), and on the
first decode step all 27 layers are inside the §6 gate against HF's
per-layer states (layer 1: max rel 3.1e-3, cosine 0.999993). The required
milestone (one MoE layer through the Fleet path) holds for every MoE layer
and the stretch goal (full model, e2e, single launch) as well.
Raw numbers: [`results/`](../results/).

| Measured | Value |
|---|---|
| Tokens matching HF greedy, free-running / teacher-forced | 32/32 and 32/32 |
| Layers inside the §6 gate on step 0 | 27 of 27 |
| Per-token latency, median / p95, **one launch for 32 tokens** | **4.08 ms / 4.11 ms (245 tok/s)**; 4.09 ms wall per token including the launch |
| Same, one launch per token (v1, for comparison) | 4.07 ms on the device, 4.11 ms wall per token with 32 launches vs 4.09 with one: the launch itself is cheap; the single launch is the design's v2 delivered, not a speed-up |
| Where the first version of the day stood | 22.05 ms |
| Protocol only (`--smoke`, 1157 events, no task bodies), one token per launch | 1.06 ms per token |
| Cross-XCD event, idle / under load (last full run, `results/microbench_summary.txt`) | 1.44 µs / 6.76 µs at 1.64 TB/s of streaming load; across the session's runs 5.9–6.8 µs at 1.5–2.2 TB/s (`microbench.json` is an earlier run: 5.88 µs at 2.20 TB/s) |
| Payload visibility: 37 producers, drained stores, one last-arriver flush; same-XCD consumer with an L1-only acquire, cross-XCD consumer | 0 stale words in 151 M, each |
| The kernel's expert GEMVs in isolation (`microbench (f)`) | 3.9 TB/s gate_up, 3.8 TB/s with down; with the per-producer flush protocol 2.8 TB/s, with the last-arriver flush 3.3 TB/s |
| Streamed read bandwidth, best depth | 4.2 TB/s (79% of peak) |
| Kernel resources (`results/kernel_resource_usage.txt`) | 253 VGPRs + 16 AGPRs (VGPR pressure spilling into AGPRs), 274 SGPR spills, 0 scratch, 24 KB LDS, 1 wave/SIMD |
| KV-cache conversion vs HF's own cache | K_nope, K_rope, V all exactly 0 error, 27 layers |

**From 22 ms to 4.1 ms**, each step attributed by the per-task trace
(`results/trace_d2_summary.txt` … `trace_d8_summary.txt`):

| Step | Per token | What the trace said and what changed |
|---|---|---|
| first run | 22.05 ms | attention 296 µs/task, router 371 µs |
| router top-k out of scratch memory, then as wave reductions | 13.7 → 6.5 ms (with the next two) | `bool taken[64]` lived in scratch; 371 → 68 → 31 µs |
| q-absorb streamed by W_UK rows | | the per-column walk with a dependent load per row was the 296 µs, not the position loop |
| split-KV ×4, then ×8 | | attention 296 → 61 → 37 µs/task |
| GEMV tails batched; prologue loads unrolled | 5.35 ms | with depth 8 and K = 2048 every row had gone through a one-load-at-a-time tail; "depth 8" had never happened |
| three global events per layer instead of six; in-kernel token loop | 5.33 ms | head-aligned q/kv_a, per-XCD router, folded reduce; the protocol cost fell (1.17 → 1.06 ms/token) but q/kv_a grew with more rows and the fold; one launch for 32 tokens |
| batched attention score reduction | 5.04 ms | 8 interleaved shuffle trees instead of 8 dependent ones: 39 → 25 µs/task |
| last-arriver L2 flush (the Fleet scheme) | 4.71 ms | per-producer `buffer_wbl2` measured at 13 µs/layer on the expert phase alone; one flush per XCD per event, validated by (d′) |
| router and merge spread over the XCD's 37 workers | 4.31 ms | one CU cannot stream 256 KB (router) or 128 KB (W_UV) fast enough: 22 → 9 µs and 12 → 8 µs |
| merge loads unrolled; L1-only acquire for XCD-local waits | **4.08 ms** | partial-merge loads in flight together (12.7 → 7.6 µs); a same-L2 consumer invalidates only its L1 instead of the L2 that 36 workers stream through |

Tried and rejected by measurement: two workgroups per CU (6.5 ms: per-CU
parallelism does not bound the expert GEMVs); workers polling the global
counters directly instead of the scheduler mirror (slower); 16-chunk-per-lane
GEMV batches (register cap, scratch spills).

Where the remaining 4.1 ms goes (per MoE layer, ~142 µs): gate_up 36,
down 23, attention 22, q/kv_a 21 (26 rows/worker plus the 64 KB partial fold),
router 9.5, o_proj 8.5, merge 7.6, and ~15 µs of events. The expert GEMVs
still run 1.5× slower inside the kernel than the same code does in
isolation (36 vs 23–27 µs); the protocol around them has been taken apart
piece by piece (flush, acquire, barrier) and accounts for roughly half of
that gap. The design's 2.5–3.5 ms target is not reached; 4.08 ms is 28% of
the measured 4.2 TB/s ceiling (the byte floor is 1.17 ms) against Fleet's
own 44% on a dense model.

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

1. **The design target (2.5–3.5 ms) is not reached: 4.08 ms.** The trace
   shows why, and it is not what the design assumed. Per MoE layer, 126 of
   the 142 µs are the seven phases' slowest tasks added up (q/kv_a 21,
   attention 21, merge 7.5, o_proj 8.4, router 9.3, gate_up 36, down 23);
   events are ~15 µs. The token moves 4.94 GB in 4.08 ms = **1.21 TB/s**
   against a measured 4.2 TB/s ceiling, so the loss is neither bandwidth nor
   the protocol: the phases are fully serial and each task carries a fixed
   cost that its few rows cannot amortise. §12's tile-granular
   gate_up→down overlap, still unimplemented, is the single largest lever.
2. **Bytes per token are not the 4.935 GB of §1.** Two redundancies added
   with the three-event graph: every XCD recomputes all 576 kv_a rows
   (8 × 2.36 MB/layer = 0.51 GB/token of HBM reads) and every q/kv_a and
   lm_head worker reads the 8 expert partials itself (296 × 64 KB/layer,
   mostly MALL/L2 hits, but 0.5 GB/token of traffic plus a redundant
   RMSNorm per worker). `rocprofv3 --pmc FETCH_SIZE` has not been run yet;
   until it is, the bandwidth-utilisation figures above use §1's byte count.
3. **The expert GEMVs run 1.45× slower inside the kernel than the same code
   in isolation** (36 vs 25 µs, `microbench (f)`), and (f) also shows the
   handshake explains only a small part. The isolated benchmark *is* the
   expert code compiled on its own; the difference is the union kernel's
   register environment (253 VGPRs, 16 AGPRs, 274 SGPR spills). Not yet
   attributed: sub-phase timestamps and the ISA are the next step.
4. Attention's lane mapping is specialised to `kv_lora = 512`,
   `qk_rope = 64`; the launcher refuses other shapes.
5. `transformers` 5.x vs the vendored modeling file: pinned `<5` on the box;
   the model's own modeling file also declares `import flash_attn` under a
   guard that transformers' import check ignores — a stub package is
   installed (`scripts/hotaisle_bootstrap.sh`), eager attention stays in use.
6. Prefill is out of scope per the task; the cache conversion is verified
   against HF's own cache (exact) on the real checkpoint.

## Next, in order

1. `rocprofv3 --pmc FETCH_SIZE` on one token: the real byte count, so
   every bandwidth claim rests on a measurement.
2. Fold the expert partials once per layer: the last-arriving down worker
   (the mechanism exists) folds the 8 partials and publishes the 8 KB
   residual; q/kv_a and lm_head workers read it instead of each reading
   64 KB and re-doing the sum. Expected 0.2–0.4 ms/token.
3. §12 tile-granular gate_up→down: down accumulates over K-chunks of h as
   they land (fixed chunk order, still deterministic) instead of waiting
   for all 37 gate_up workers. Expected 0.3–0.4 ms/token.
4. Attribute the in-kernel expert GEMV slowdown: timestamps around the
   prologue vs the stream, then the ISA of the union kernel vs (f)'s;
   decide between register pressure fixes and splitting the body out.
5. Then the structural items: activations in uncached memory (no L2
   writeback or invalidate anywhere), o_proj folded into merge (two global
   events per layer), non-temporal weight streams so the KV cache and the
   small tensors stay in the Infinity Cache, prefetch of routing-independent
   weights by idle workers during the latency-bound phases.
