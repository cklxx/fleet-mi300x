# Status — Fleet-style batch-1 decode, DeepSeek-Coder-V2-Lite-Base on MI300X

Design: [design.md](design.md) · Task: one MI300X, bf16, bs=1, 1024-token
context, 32 greedy tokens.

This file tracks what is done, what is verified, and what is known to be
missing — the task asks for the milestone reached and the remaining limitations
to be stated plainly, not just for code.

## Where this is (2026-09-15, after three sessions on the MI300X)

**End to end works, matches HuggingFace, and runs at 3.60 ms per token —
20% faster than vLLM on the same box (4.52 ms).** On a Hot Aisle 1×MI300X VM
(ROCm 7.2, gfx942), one cooperative launch decodes all 32 greedy tokens over
the 1,024-token context; the argmax task of each token feeds the next
token's embed on the device. Every token equals HF's greedy choice, and on
the first decode step all 27 layers are inside the §6 gate against HF's
per-layer states. The required milestone (one MoE layer through the Fleet
path) holds for every MoE layer and the stretch goal (full model, e2e,
single launch) as well. Raw numbers: [`results/`](../results/).

| Measured (v0.16: `fleet_decode_nt`, `taskgraph_d16.bin`; session 6 on a fresh VM) | Value |
|---|---|
| Tokens matching HF greedy, free-running / teacher-forced | 32/32 and 32/32 (`results/decode_final_r*.json`, `decode_final_teacher.json`) |
| Layers inside the §6 gate on step 0 | 27 of 27 |
| Per-token latency, median / p95, **one launch for 32 tokens** | **3.661–3.665 / 3.670–3.673 ms (273 tok/s)** over three runs; teacher-forced 3.654 ms. (Session 5 measured 3.624–3.628 for the same graph on a different VM instance: the ~1% is between-VM, not a regression) |
| Same session, same binary, the v0.15 graph before this session's changes | 3.673–3.682 ms; between VMs/sessions the same binary has varied by ~5% (3.73–3.93), within a session <1% |
| Top-k published once per XCD by the router's last arriver (`--topk-published`) | 3.670–3.678 ms over three runs: **slower**. The gate_up prologue falls 5.7 → 1.5 µs, but the router phase grows 12 → 18 µs because one worker does the 36-shuffle reduction while 36 wait. Off |
| Cross-task weight prefetch across the event wait (`--prefetch-next`) | 3.812–3.815 vs 3.673–3.682 ms: **4% slower**. 8 extra 16 B loads per thread held across the wait cost more than the head start is worth. Off |
| One launch per token (v1, v0.14 binary) | 3.733 ms: the single launch is the design's v2 delivered, not a speed-up |
| `--coherent-acts` (atomic cross-XCD activations, no consumer L2 invalidate) | 3.718–3.719 vs 3.709–3.715 fenced **on the same kv_a-replicated graph** (it cannot run on the shared-kv_a default, which the launcher refuses): not faster; off |
| Variants (v0.14 binary): plain loads / 8 KV chunks / K-chunk tiling / prefetch / kv_a replicated (v0.15) | 3.941 / 3.818 / 4.020 / 4.121 / 3.640–3.653 ms (`results/decode_*.json`) |
| Expert phase on split worker groups (18 gate_up + 19 down per XCD) with / without tiling; 22 + 15 with tiling | 4.784 / 4.150 / 4.834 ms: an XCD's bandwidth needs all 37 CUs, 18 pull 44 µs of gate_up instead of 33 |
| **vLLM 0.11.2 on the same VM** (rocm/vllm docker, V1 engine, AITER MLA backend, full CUDA graphs, bf16, batch 1, same prompt) | 4.522 ms median / 4.553 mean (220 tok/s), 32/32 tokens equal to the golden ones; `results/baseline_vllm.json`, `bench/vllm_decode_timing.py` |
| HF transformers 4.44 eager (torch 2.10 rocm7.0), same prompt | 54.4 ms (18 tok/s); `results/baseline_hf.json` |
| Where the first correct version stood | 22.05 ms |
| Protocol only (`--smoke`, 1346 events, no task bodies) | 2.84 ms per token with the fences, 2.20 ms without (smoke has no bodies, so the fences are its whole cost) |
| Global events per MoE layer / per token | 3 / 85 (q/kv_a, o_proj, down; v0.10 had merge instead of q/kv_a) |
| Cross-XCD event, idle / under load (`results/microbench_summary.txt`) | 1.36 µs / 5.88 µs at 1.59 TB/s of streaming load |
| Payload visibility (`microbench (d)(d')(d''')`) | fenced protocol 0 stale in 16.4 M; 37 producers + last-arriver flush 0 stale in 151 M each; agent-scope atomic payload without fences 0 stale in 16.4 M. MTYPE-UC memory without fences (d''): 12.3 M of 16.4 M stale — not usable on this VM |
| Streamed read bandwidth, best depth | 4.2 TB/s (79% of peak) |
| **Bytes per token, measured** (`rocprofv3 --pmc FETCH_SIZE`, 4 teacher-forced tokens, **the shipped graph**, `results/rocprofv3_fetch_size_d16_shared.csv`) | 20.80 GB for 4 tokens = **5.20 GB/token**. Two profilers agree to five digits (20,799,909 KB and 20,805,896 KB). The byte model's 4.90 GB of HBM traffic is 5.8% under it, which is the reconciliation the phase-rate tool reports. Earlier runs measured 5.67 GB on the graph that still replicated kv_a; the 5.20 figure was an estimate until this session and is now measured |
| Protocol only (`--smoke`, 1157 events, no task bodies) | 3.01 ms per token: with no bodies every wait is back-to-back, so this is the protocol's worst case, not its share of the 3.63 |
| Kernel resources (`results/kernel_resource_usage.txt`, `results/isa_summary.txt`) | 256 VGPRs + 72 AGPRs (VGPR spill space), 472 SGPR spills, 112 B of stack (4 scratch instructions, none in a GEMV loop), 25 KB LDS, 1 wave/SIMD. The weight-streaming basic blocks (16–18 `global_load_dwordx4` each) carry 0 scratch and 2 AGPR moves: **the in-kernel vs isolated GEMV gap is not register pressure** |
| KV-cache conversion vs HF's own cache | K_nope, K_rope, V all exactly 0 error, 27 layers |

`rocprofv3` (and `rocprof` v1) crash at process exit on this VM (ROCm
7.2.4, SR-IOV VF), but with `--kernel-trace` left out the counter row for
the decode kernel is written before the crash; both tools agree
(22,660,092 vs 22,665,918 KB).

### The optimisation path, each step attributed by the per-task trace

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
| merge loads unrolled; L1-only acquire for XCD-local waits | 4.08 ms | partial-merge loads in flight together (12.7 → 7.6 µs); a same-L2 consumer invalidates only its L1 instead of the L2 that 36 workers stream through |
| **v0.11** — fold once by the last-arriving down worker, gate_up→down tiled in 512-row K-chunks, o_proj K-split per XCD (merge→o_proj XCD-local), 2 global events/layer | 4.80 ms (**slower**) | the per-layer timeline (`scripts/trace_timeline.py`) showed why: the one-writer fold serialises ~4 µs at each of the two fold points; the tiled down cannot start before gate_up ends because both run on the *same* 37 workers (down's first ready at 120.6 µs, gate_up's last done at 122.1); the K=256 o_proj GEMV left half of every wave's lanes idle (8.4 → 11 µs) |
| v0.11 without the tiling | 4.38 ms | tiling alone cost 0.42 ms; kept as `--k-chunk`, off by default |
| **v0.12** — every consumer folds the 8 partials in its own prologue (o_proj partials in the router, expert partials in the next q/kv_a), worker (0,0) publishes to the other `x` buffer; half-wave GEMV for K ≤ 256 | 4.06 ms | the parallel fold costs ~1 µs on the prologue instead of ~4 µs serial; o_proj 11 → 6.8 µs |
| non-temporal weight loads (`-DFLEET_NT_WEIGHTS=1`, second binary) | 3.90 ms | 4% for one flag: the weight stream stops evicting the KV cache, kv_b, router and norms |
| **v0.13** — 16 KV chunks (256 attention tasks/layer, still ≤ 37 per XCD); argmax from 296 per-worker candidates written by the lm_head epilogue | **3.73 ms** | attention 20.3 → 16.7 µs; argmax was 109 µs on one CU (a dependent load per compare), now 4 µs |
| idle-worker prefetch of routing-independent weights into the Infinity Cache (`--prefetch`) | no gain (4.84 vs 4.80; 4.77 vs 4.70 with NT) | kept as a graph option; the per-CU streaming rate (~30 GB/s) limits what 21 idle workers can pull in 20 µs to ~60 MB/layer, and the phases that would benefit are latency-bound, not byte-bound |
| fold with all 72 partial loads in flight (one round trip instead of two) | +0.1 ms (**slower**) | 48 B/lane of scratch: the register file, again |
| **v0.14** — 16-byte loads for every staged vector and for the fold (18 loads per thread in flight instead of 36 scalar ones in two batches); merge partials in two batches of 8 | 3.75 ms, and the run-to-run spread within a session fell below 1% | the prologues did *not* get shorter (7.5 / 8.3 / 7.2 / 5.6 µs), so the round-trip count was not what bounded them; the scratch stayed at 48 B/lane, so the merge was not its source either |
| a second stamp splits each prologue into staging vs the rest | — | q/kv_a 7.4 = 4.9 fold + 2.5 norm; router 7.4 = 4.7 + 2.7; attention 8.0 = 3.4 kv_post + 4.6 q_absorb; **gate_up 5.5 = 4.9 routing (64 logits from L2 under full load, softmax, top-k) + 0.6 staging; down 5.7, the same routing again** |
| **v0.15** — down reuses the routing its own worker's gate_up left in LDS (`ROUTING_CACHED`, the validator checks the queue order); kv_a computed once, split over the 8 XCDs behind one global event instead of replicated (`--kva-replicated` restores the old graph) | **3.63 ms** | down's prologue 5.7 → 1.6 µs (−2.2%); q/kv_a 23.2 → 18.6 µs but attention's kv_post now reads kv_a from another XCD's write (+2 µs), net −0.6%, and 0.51 GB/token fewer bytes |
| expert phase on two worker groups so the K-chunk tiling can overlap gate_up and down | 4.15–4.83 ms (**slower**) | 18 CUs stream gate_up in 44 µs where 37 take 33: an XCD's share of HBM needs every CU issuing; the tiling on top makes it worse again |
| **v0.16** — three more candidates, all measured, all rejected: publish the top-k once per XCD instead of per task; prefetch the next task's first rows across its event wait; re-test `--coherent-acts` on an equal graph | 3.661–3.665 ms, i.e. unchanged | the publication trades 4 µs of gate_up prologue for 6 µs of serialised router tail — the same shape as the v0.11 one-writer fold; the prefetch's 8 held loads cost 4%; the coherent protocol is within noise of the fenced one. The kernel keeps all three behind flags and ships none |
| **v0.17** — the expert partials stored as bf16 | **3.643 ms** vs 3.652–3.659 over three alternating pairs, about 0.4% | the fold reads 72 KB per worker in 4.8 µs, which is 15 GB/s against a 14.2 GB/s per-worker ceiling: bandwidth-bound, and that is why vectorising it to float4 earlier changed nothing. The partials are bf16-valued already (EPI_BF16), so storing them narrower is bit-exact — 32/32 tokens and 27/27 layers unchanged. Predicted 1.5%, measured 0.4%; the rest of the fold's cost is not bytes |
| **v0.18** — each head's absorbed q_c published once by the workers that idle during attention, instead of all 16 KV chunks streaming W_UK for it | **3.593, 3.595, 3.596 ms** against 3.645, 3.654, 3.650 for the same binary on the old graph: 0.055 ms, 1.5%, intervals not overlapping | 32 of an XCD's 37 workers take an attention task, so five are free. Each publisher owns a row range and writes an fp32 partial; the readers add them in slice order and take q_pe straight from q. The readers wait *inside the body*, after kv_post, which is the discriminator the three failed publications lacked: their waiters had nothing to overlap. Now the default; `--qc-per-task` restores the old graph |
| exempting the router and kv_b from the non-temporal weight stream so they could stay in the Infinity Cache | **4.249 vs 3.597 ms, 18% slower** (reverted) | the (g) probe had already said why and I did not listen to it: the read-rate cliff is at the 4 MB per-XCD L2, so making 113 MB of kv_b cacheable pushes the working set past it and evicts the stream it was meant to share with |
| issuing the gate_up logit load before the 8 KB staging, so the load flies during work that does not need it | 3.595 and 3.588 against 3.591 and 3.592: noise (reverted) | the ceiling was 0.03 ms and it did not reach it. Worth recording alongside: that build spilled 729 SGPRs against the baseline's 570 and ran at baseline speed, while the 18% regression spilled 692. **Spill counts do not predict this kernel's performance** |

### Is 26% of peak bandwidth a defect? No, and here is the evidence

The obvious worry about this kernel is that a batch-1 decode is a pure
bandwidth problem and we only reach a quarter of peak. By the standard
metric, achieved bandwidth over peak with achieved taken as useful bytes
divided by time per token, we sit at 4.935 GB / 3.598 ms / 5.3 TB/s =
**25.9%** (27.2% if the denominator uses the 5.19 GB actually fetched, or
33% against the 4.2 TB/s this hardware can really stream).

Four independent checks say the streaming code is not what is losing:

* the same GEMV in isolation reaches 3.9 TB/s, 93% of the 4.2 TB/s
  achievable, so the inner loop is close to the metal;
* Little's law wants HBM latency times per-CU bandwidth in flight, about
  14 KB per CU here; the kernel keeps roughly 32 KB in flight, over twice
  what is needed;
* AMD's own occupancy guidance says one wave per SIMD can saturate HBM when
  vector width and outstanding-load count are right, and warns that maximum
  occupancy often *starves* bandwidth. Our shape, deep per-wave instruction
  level parallelism at one wave per SIMD, is the recommended one, not an
  accident;
* the closest production comparison, vLLM 0.11.2 with the AITER MLA backend
  and full CUDA graphs, on the same GPU, model, prompt and token count,
  runs at 4.52 ms, which is **20.6% MBU**. It is 22% slower than this
  kernel on the same hardware.

For scale, Databricks report batch-1 MBU of about 50% for MPT-7B on one
A100 and 55 to 60% for Llama-2-70B across two to four GPUs. Those are
**dense** models: one contiguous multi-gigabyte weight set per token, where
a single phase streams uninterrupted for hundreds of microseconds. A MoE
decode at batch 1 is the opposite shape, and the literature names the same
cause we measure: small per-expert GEMV shapes with low arithmetic
intensity. Here each XCD streams one 17.3 MB expert for 36 microseconds,
and the layer then passes through seven serial phases of which roughly 40%
of the time moves almost no bytes at all.

That last sentence is the whole gap, and it is structural rather than a
coding defect. Closing it does not need a better inner loop; it needs
either fewer phase boundaries, which is worth tens of microseconds per
token, or more tokens per weight read, which is what speculative decoding
buys and which would change the task's terms.

Occupancy was then tested directly rather than argued about, because the
one phase where it could plausibly matter is the expert stream. The
isolated benchmark compiles to 154 VGPRs and reaches 3 waves per SIMD,
while the same code inside the megakernel is pinned to one. A second copy
of the identical body was given 120 extra live VGPRs to take that occupancy
away, and the compiler's own resource report confirms the manipulation:
256 VGPRs plus 20 AGPRs, occupancy 1.

| gate_up + down, no handshake | 3 waves/SIMD | 1 wave/SIMD |
|---|---|---|
| first run | 34.5 us/layer | 34.1 us/layer |
| second run | 38.8 us/layer | 38.8 us/layer |

**Occupancy makes no difference to this phase, and the register union the
megakernel forces costs no throughput.** That closes a theory this file
floated twice, on our own hardware rather than on vendor guidance.

The same table retires something else. The two runs of the *identical*
3-wave configuration differ by 12%, which is larger than the roughly 6% gap
between in-kernel and isolated that had been carried as the largest
unattributed loss. It was inside the benchmark's own noise.

Occupancy, for the record: the kernel uses 256 VGPRs and 87 AGPRs. Those
share one 512-register file per SIMD, so 343 registers allow exactly one
wave. That is also the real reason two workgroups per CU measured slower
long ago: they could not be co-resident, so they serialised and each took
half the rows. The attribution in the rejected-experiments line above was
wrong for two sessions.

Sources: [ROCm occupancy math on CDNA](https://rocm.blogs.amd.com/software-tools-optimization/occupancy-math-mi355x/README.html),
[ROCm MI300X workload optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html),
[Databricks, LLM inference performance engineering](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices),
[MoE-Inference-Bench](https://arxiv.org/pdf/2508.17467).

Raw numbers for the occupancy test are in
[`results/microbench_f_occupancy.txt`](../results/microbench_f_occupancy.txt).

### One probe, two optimisations closed

`microbench (g)` streams a single buffer from 1, 37, 148 and 296 workgroups
at once, over four working sets. It costs nothing to run and it answered two
open questions that would otherwise have cost a kernel rewrite each.

| working set | 1 | 37 | 148 | 296 | concurrent workgroups |
|---|---|---|---|---|---|
| 128 KB | 50.3 | 48.5 | 48.2 | 47.1 | per-worker GB/s |
| 1 MB | 49.1 | 49.6 | 49.4 | 48.3 | |
| 8 MB | 27.8 | 28.2 | 27.6 | 25.9 | |
| 64 MB | 27.4 | 28.0 | 27.6 | 25.8 | |

Every row is flat: going from one reader to 296 readers of the same bytes
costs at most 6%. **Sharing is free, so merging two attention heads into one
task cannot pay.** That idea only earns anything if an XCD's L2 is the
bottleneck under concurrency, and it is not; it would also cost per-lane
registers on a kernel already at one wave per SIMD.

Every column falls once, between 1 MB and 8 MB, and then stops falling. That
is the per-XCD L2 capacity boundary (4 MB), not paging: a TLB effect would
keep degrading from 8 MB to 64 MB and it does not. **So the page-size A/B
has nothing to measure**, and the expert GEMV's 77% of the per-worker ceiling
is a capacity and locality effect rather than a paging one.

Note what the probe does not say: at 296 readers of 64 MB the aggregate is
7.6 TB/s, above the 4.2 TB/s HBM ceiling, so even the largest working set
here is being served by the Infinity Cache. This measures cache bandwidth,
not HBM.

Estimates for this kernel have run about twice high, three times now: the
bf16 partials were predicted at 1.5% and measured 0.4%, and q_c was
predicted at 3.4% and measured 1.5%. The staging and publication costs are
not proportional to the bytes they move, which is also why vectorising the
fold changed nothing. Halve any byte-based estimate before spending a day
on it.

Tried and rejected by measurement: two workgroups per CU (6.5 ms, and the
reason is now understood: see the occupancy section below); workers polling
the global counters directly instead of the scheduler mirror (slower);
16-chunk-per-lane GEMV batches (register cap, scratch spills); the one-writer
fold; K-chunk tiling of gate_up→down on one worker group (nothing overlaps)
and on two (too few CUs per group); idle-worker prefetch; MTYPE-UC
activations without fences (wrong: bench (d'')); fence-free coherent
activations (no gain). Not tried, by arithmetic: a per-head q_absorb task
— 16 chunk tasks do it in parallel today, one task would do it serially
for the same wall time plus an event. The pattern behind three of these
rejections: at batch 1 every phase already has 296 workers doing the same
small thing in parallel, so moving work onto one of them to remove
redundancy trades parallel microseconds for serial ones and loses.

Where the 3.66 ms goes (`results/timeline_v16_default_L5.txt`, layer 5, 129 µs;
"prologue" is everything before the weight stream, "staging" its first part):

| phase | starts | ends | avg task busy | prologue | of it: staging |
|---|---|---|---|---|---|
| q/kv_a | 0 | 19.3 | 16.9 | 7.6 | 4.8 (fold) |
| attention (256 tasks) | 20.2 | 40.8 | 19.2 | 10.5 | 5.7 (kv_post) |
| merge + W_UV (256) | 39.8 | 51.1 | 9.0 | — | — |
| o_proj K-split | 49.7 | 59.7 | 6.9 | 0.7 | — |
| norm + router | 60.4 | 73.2 | 11.2 | 7.6 | 4.8 (fold) |
| expert gate_up | 72.4 | 108.9 | 34.0 | 5.7 | 4.7 (routing) |
| expert down | 106.6 | 129.3 | 18.7 | 1.6 | — |

Events cost under 1 µs each now (2 global + 48 XCD-local per layer); the
gaps between phases are 0.4–1 µs. What remains is the phases themselves,
and inside them the prologues: every one of the 296 workers stages 8 KB of
x (and 64 KB of partials at the two fold points) from HBM/Infinity Cache
under full streaming load. The split stamps say what it is: a 64 KB fold
or a 64-logit routing read costs ~5 µs under load whether it is 18 loads
or 72, and a 2048-float RMSNorm ~2.5 µs. The 27 layers × ~32 µs of
prologue ≈ 0.9 ms of the 3.63. The lm_head streams 420 MB at 3.1 TB/s
(136 µs) and its prologue is 9 µs.

### Third session: the "far goal" route and what survived contact

The plan was eight structural steps to ~2.5 ms. Measured one by one:

| Step | Result |
|---|---|
| 1. `rocprofv3 --pmc FETCH_SIZE` | obtained on the third try (without `--kernel-trace`): 5.67 GB/token, 15% over §1 |
| 2. Fold once by the globally last worker, single `x` | slower (serial ~4 µs per fold point); replaced by the parallel prologue fold with a ping-pong `x` and a separate o_proj partial buffer |
| 2b. kv_a once instead of 8 copies | kept (v0.15): −0.6%, −0.51 GB/token |
| 2c. Routing once per worker instead of twice | kept (v0.15): −2.2% |
| 3. Tile-granular gate_up→down | slower on one worker group (nothing overlaps) and slower again on two (`--split-workers`: 18 CUs cannot pull an XCD's share of HBM). The machinery stays behind `--k-chunk` / `--split-workers` |
| 4. o_proj K-split per XCD | kept: merge→o_proj is XCD-local, 2 global events per layer; needed the half-wave GEMV to pay off |
| 5. Activations without L2 fences | MTYPE-UC allocation does not deliver it here (bench d''); agent-scope atomic payload accesses do (bench d'''): `--coherent-acts`, see below |
| 6. Non-temporal weight streams | kept, −4% |
| 7. Idle-worker prefetch | no gain; kept as an option |
| 8. Attention split ×16; argmax off the critical path | kept, −4% |

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
| `src/host/taskgraph.py` | Builds + validates the Fleet task DAG, fans Chiplet-tasks out to per-worker descriptors, emits them with an event-label sidecar | v0.13 graph (`--kv-chunks 16`): 552 logical tasks / 1,992 descriptors per MoE layer, 2 global + 48 XCD-local events per layer; 14,914 logical / 54,082 descriptors and 1,346 events (58 global) per token; `--report` prints the counts and the validator's verdict |
| `tests/test_queue_simulation.py` | Executes the emitted queues under the kernel's protocol: 296 queues in order, monotonic counters, targets `epoch × wait_count`, XCD-local visibility, per-wave chunk arrivals, 3 epochs, forward/reverse/random worker order | 30/30 over five graph variants; the sabotage cases (old signal-side count, missing last-arrival rule) are reported |
| `src/host/reference_decode.py` | NumPy absorbed-MLA decode: the arithmetic each HIP task must reproduce | Boundary tests below |
| `src/host/kv_convert.py`, `reference_run.py` | prefill → decode cache `[27][1056][576]` bf16 (32.8 MB) and greedy tokens, written from the *same* HF prefill the launcher is compared against | Layout arithmetic matches §5; reconstruction check needs the model |
| `tests/test_absorbed_equivalence.py` | Absorbed MLA ≡ materialised K/V; RoPE interleave; softmax scale | 4/4 pass, max rel 4.6e-7 |
| `tests/test_reference_vs_hf.py` | NumPy reference vs HuggingFace on a tiny random config | 4/4: RMSNorm, RoPE table 6.0e-8, RoPE applied 2.5e-8, **absorbed attention 2.4e-7**, MoE top-k exact, MoE output 7.3e-8 |
| `src/runtime/fleet_runtime.h` | AOT queues, events with wait-side fields, XCD-local role tickets, grid-distribution check, wait timeouts → abort codes | Layout test 6/6; interface test 14/14; queue simulation 10/10 |
| `src/kernels/fleet_kernel.hip`, `gemv.h`, `attention.h`, `expert.h` | Persistent kernel and task bodies; LDS-only scratch; wave-per-row GEMV; wave-per-position attention | **Parsed in HIP language mode, device and host passes, by clang 14 with the AMDGPU backend** (`scripts/hip_syntax_check.sh`): 0 errors, 0 warnings at `-Wall -Wextra`. The compiler's own `__hip_atomic_*`, `__builtin_amdgcn_fence`, attributes and inline asm are exercised; only the runtime API is stubbed |
| `src/host/fleet_launch.hip` | Queues, event buffers, weights/cache/golden loading, YaRN tables, per-token cooperative launch, HF comparison, abort decoding, `--smoke` / `--teacher-force` / `--json` | Same parse, both passes, 0 errors |
| `bench/microbench.hip` | (d) cross-XCD payload visibility under the kernel's fence placement, (d') 37 producers with one last-arriver flush, (d'') MTYPE-UC memory without fences, (d''') agent-scope atomic payload without fences; (a) cross-XCD event cost, cached vs uncached counters; (e) the same under a 302-workgroup HBM stream; (b) same-XCD cost; (c) streamed read bandwidth vs depth; (f) the expert phase in isolation; XCC_ID checks on every pair | Run on every session; `results/microbench_summary.txt` |
| `src/host/pack_weights.py` | Flat bf16 blob, 256-B aligned, fused q‖kv_a, interleaved gate/up, `.manifest` the launcher parses | Syntax and CLI only; needs the checkpoint |
| `scripts/hotaisle_bootstrap.sh` (`setup_env.sh` is the generic form) | Fresh Ubuntu + ROCm box to results, unattended: venv, torch/transformers, model download, the seven local tests, both binaries, graphs, microbench, smoke, reference run, packing, the headline decodes, the variants, a best-effort rocprof byte count | Run from scratch on every new VM (three times) |
| `tests/test_expert_addressing.py` | Builds the packer's byte layout, resolves the 8 units as `expert.h` does, runs the kernel's arithmetic with HF's rounding points against `reference_decode.moe` | 7/7: routing exact, phase output within 4.1e-3 of the reference (bf16-ulp level), and the pre-review offsets produce a 37% error that the test reports |

### The test suite, and what an audit of it found

`python3 tests/run_all.py` is the single entry point: the unit tests (found
by glob, so a new file is picked up without editing a list), the graph
validator on all seven flag combinations the builder supports, and the HIP
parse of the three translation units in both passes and in the non-temporal
build. `--mutate` adds the gate below. Eighteen gates, all green.

The suite was then audited by injecting realistic regressions into the code
each test covers and re-running it. Two suites caught nothing at all:
`test_row_partition` and `test_expert_addressing` re-implemented the
production formula in Python and compared the mirror with itself, so no
mutation was detectable. Both are now pinned to their sources: the row split
reads the constants out of the headers and asserts that `xcd_rows` and the
interleaved worker stride still read the way the mirror models them, and the
expert layout asserts the four addresses in `expert.h` that the first review
got wrong once already.

The highest-risk gap was the memory model. Deleting the producer-side
release or the XCD-local acquire passed every test and the parse gate, on
the one mechanism the design rests on. `test_kernel_interface` now asserts
both fences, that the XCD-local acquire is the L1-only `buffer_inv sc0`
inside `fence_acquire_local` rather than anywhere in the file, and that the
producer writeback is unconditional, which is the invariant measured above.

`taskgraph.validate()` was 130 lines that no test ever called.
`test_validator` builds all six graph variants and sabotages each the way a
real edit would. Writing it exposed a hole in the validator itself: it kept
one per-XCD share per (event, XCD), so a single wrong descriptor was
overwritten by the 36 that followed it and never reported.

| file | what it is | count |
|---|---|---|
| `tests/run_all.py` | every gate that runs without a GPU, one exit code | 18 gates |
| `tests/mutate.py` | breaks the code on purpose, requires the suite to notice | 12 mutations |
| `tests/run_gpu.py` | every gate that needs 304 resident workgroups | 18 gates + 1 informational |
| `tests/test_queue_simulation.py` | the event protocol executed on the CPU over five graph variants | 30 |
| `tests/test_kernel_interface.py` | kernel, runtime header and launcher cross-checks, including the fences | 19 |
| `tests/test_validator.py` | the graph validator against nine sabotages | 16 |
| `tests/test_row_partition.py` | every GEMV row owned once, pinned to the headers | 17 |
| `tests/test_expert_addressing.py` | packed expert layout and slot resolution, pinned to `expert.h` | 11 |
| `tests/test_reference_vs_hf.py` | the NumPy reference against HuggingFace | 8 boundaries |
| `tests/test_descriptor_layout.py` | host packer and device struct, byte for byte | 6 |
| `tests/test_absorbed_equivalence.py` | absorbed MLA equals the materialised path | 4 |

`tests/run_gpu.py` is the other half, for what only 304 resident
workgroups can exercise: the payload-visibility proofs, the 8x38 placement
probe, a full 32-token decode on every graph the builder emits, teacher
forcing, the plain-loads build, bitwise determinism (the same decode twice,
layer dumps compared byte for byte), and the abort path. The sweep earned
itself on its first real run: storing the expert partials as bf16 left a
null destination on the EPI_BF16_ACC path, which only the tiled graphs take,
so the default decode and every local gate stayed green while two variant
decodes died. Before that run the sweep had been scoring those variants
PASS with the detail "not emitted", which is why a missing input is now a
skip that counts against the exit code.

The abort gate was wrong in concept at first: it lowered the spin limit and
expected a timeout, but lowering the limit does not create a stuck wait, and
the wait loop only evaluates the limit every 1024 polls. It now writes a
graph with a wait count no number of producers can reach; the launcher
reports the event it timed out on and exits non-zero, and a normal decode
straight afterwards returns 3.646 ms and 32/32.

Two mutation rows are marked redundant on purpose: the validator carries two
overlapping XCD-locality checks, and deleting either alone is invisible
while deleting both is caught. That is recorded rather than papered over.

## Decisions the design text did not make, made here

- **Every cross-XCD handshake is agent-scope; XCD-local ones are not.**
  MI300X L2 is per XCD. For a global event the XCD's last arriver releases
  once (`buffer_wbl2`), counters are agent-scope atomics, pollers use
  agent-scope atomic loads, consumers acquire (`buffer_inv sc1`). For an
  XCD-local event producers only drain their stores and consumers invalidate
  their L1. Both are what the LLVM AMDGPU memory model guarantees and both are
  checked word for word by the microbench. The scheduler mirror stays: direct
  polling of the global counters by 296 workers measured slower.
- **Chiplet-tasks are per-worker descriptors.** Nothing on the device
  broadcasts a task; the descriptor count grows to 33k (2 MB), which is read
  sequentially per worker and is irrelevant next to 4.9 GB of weights.
- **Failure is loud.** A wrong XCD distribution, a wait that never completes,
  or a grid barrier that does not fill all end the launch with a reason code
  the launcher prints with the event's label, instead of a hung GPU.
- **Split-KV runs at 16 chunks per head; the merge runs after all chunks.**
  The tile-granular producer/consumer of §12 row 1 is implemented
  (`--k-chunk`) and measured slower, so it is off.

## Known limitations, open risks

1. **The design target (2.5–3.5 ms) is not reached: 3.66 ms**, ~1.4 TB/s
   of a 4.2 TB/s ceiling at ~5.2 GB/token (byte floor ~1.25 ms), against
   Fleet's own 44% of peak on a dense model. The token is 27 serial layers of seven serial
   phases; events are no longer the cost (< 1 µs each), the per-task
   prologues and the latency-bound phases (q/kv_a with 28 rows per worker,
   attention at 1K context, router) are. It beats vLLM on the same box by
   19% (4.52 ms) with bit-exact HF greedy tokens.
2. **Bytes per token were 5.67 GB, not the 4.935 GB of §1** (measured,
   `FETCH_SIZE`, v0.14). Of the excess, 0.45 GB was the replicated kv_a
   (7 redundant copies of 2.36 MB over 27 layers, not 8 — an earlier
   version of this file said 0.51 GB)
   (gone in v0.15, not re-measured: the profiler crashes at exit and the
   run was not repeated); every worker still folds the 8 partials itself
   (64 KB per worker per fold point) and the rest is L2 misses.
3. **`--coherent-acts` is validated by measurement, not only by the memory
   model.** Cross-XCD activations go through agent-scope atomic loads and
   stores, and global waits skip the L2 invalidate (except the token
   boundary and the dense layer). The fully fence-free variant — producers
   skipping the `buffer_wbl2` too, which bench (d''') passes — produced wrong
   tokens in 10 of 18 launches; with the writeback kept it was correct in
   18/18 launches plus every run since (free-running and teacher-forced,
   27/27 layers). It was 1.5–2% faster than the fenced protocol before the
   prologue loads were vectorised and is not measurably faster after
   (3.748–3.763 vs 3.746–3.756 ms). Why the write-through atomic store is
   not enough without the writeback is not understood. Off by default; the
   headline number uses the fenced protocol.
4. The expert GEMVs still run slower inside the kernel (33.5 µs gate_up)
   than in isolation (25 µs, `microbench (f)`). The ISA rules out register
   pressure (the streaming loops have no spill traffic); the sub-phase
   stamps attribute 5 µs to the routing read; the remaining ~4 µs is the
   stream itself under 296-CU contention plus the per-wave chunk arrival
   at the end.
5. Attention's lane mapping is specialised to `kv_lora = 512`,
   `qk_rope = 64`; the launcher refuses other shapes.
6. `transformers` 5.x vs the vendored modeling file: pinned `<5` on the box;
   the model's own modeling file also declares `import flash_attn` under a
   guard that transformers' import check ignores — a stub package is
   installed (`scripts/hotaisle_bootstrap.sh`), eager attention stays in use.
7. Prefill is out of scope per the task; the cache conversion is verified
   against HF's own cache (exact) on the real checkpoint.

## Next, in order

1. ~~Publish the top-k from the router's last arriver~~ — done and
   measured slower (above); the routing stays per task.
2. The two folds (4.9 µs each per layer, ~0.26 ms/token): the bytes are
   not the cost (18 vs 72 loads made no difference), the round trip under
   load is; the only way out is fewer of them, e.g. the router's fold
   result published by one worker per XCD and re-read from L2.
3. The attention prologue (10 µs of 18.8 with the shared kv_a): kv_post
   reads kv_a from another XCD's memory now; a per-XCD copy written by the
   q/kv_a workers of that XCD alongside the shared one would keep the
   byte saving and the local read.
4. Bytes in flight per CU: the GEMV streams ~48 KB per CU; a third buffer
   in AGPRs (loads can target AGPRs on gfx942) or LDS-direct loads would
   raise it without touching occupancy.
5. Understand why `--coherent-acts` needs the producer writeback; the
   protocol's worst case (3.0 ms in the body-less smoke run) says the
   fences are not free even though dropping the acquires bought nothing.
