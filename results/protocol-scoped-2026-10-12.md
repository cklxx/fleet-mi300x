# Scoped audit of the shipped event/fence protocol — 2026-10-12

**Conclusion [R]: no complete global event, last-arriver producer writeback, or payload acquire is proved removable from the shipped decoder by this tree. Two narrow changes preserve its existing visibility contract: cache the scheduler's own mirror value, and combine the single-chunk gate-up's four wave arrivals into one workgroup arrival. Their latency savings are unmeasured. The demonstrated additional removable time is therefore 0.000 ms, meaning “no established gain,” not “proved zero cost.” The 0.640 ms smoke difference is a gross historical fence-associated pool, not an available speedup.** The source and measurement evidence for these conclusions follows.

Notation: **[M]** = recorded measurement; **[L]** = read-only local check performed for this audit; **[R]** = reasoning or arithmetic; **[U]** = not verified. Citations use physical source-file line numbers. This analysis fixes the shipped d16 configuration: shared kv_a, published q_c, one gate-up chunk, fenced activations, scheduler mirrors, and the experimental local-assembly switches off. These defaults are in `src/host/taskgraph.py:901–923`, `src/host/taskgraph.py:936–946`, and `src/host/fleet_launch.hip:373–394`; the shipped result is authoritative at `docs/STATUS.md:24–25` and `docs/STATUS.md:80`. No computational, precision, task-shape, worker-count, or weight-traffic change is proposed.

## 1. Inventory of events, fences, and their time

### What has actually been timed

| Evidence | What it establishes | What it does not establish |
|---|---|---|
| [M] 3.593 / 3.595 / 3.596 ms/token, 32/32 tokens and 27/27 layers (`docs/STATUS.md:24–26`, `docs/STATUS.md:80`) | Shipped end-to-end baseline. | A decomposition of that time into protocol instructions. |
| [M] Bodyless smoke: 2.84 ms fenced, 2.20 ms unfenced, labelled 1,346 events (`docs/STATUS.md:38`). Another bodyless run is 3.01 ms with 1,157 events (`docs/STATUS.md:44`). | Whole-protocol worst-case measurements under back-to-back waits. [R] Paired difference: 0.640 ms. | Neither the protocol's share of 3.595 ms nor individual fence/event costs. The 2.20 ms residual means “fences are its whole cost” cannot literally mean all smoke time is fence time. |
| [M] Cross-XCD event: 1.36 µs idle, 5.88 µs at 1.59 TB/s load (`docs/STATUS.md:40`) | Authoritative isolated-handshake scale, sensitive to contention. | Cost of any one of the 85 shipped global events, or a sum over them. The benchmark performs a ping-pong with an acknowledgement (`bench/microbench.hip:132–145`), not the decoder's fan-in/mirror/fan-out path. |
| [M] Fenced payload: zero stale words in 16,384,000; last-arriver scheme: zero in 151,552,000 for each local and remote consumer (`results/microbench_summary.txt:4–5`, `results/microbench_summary.txt:13–14`) | Empirical support for the existing barrier/drain/release/acquire placement. | A duration for any one instruction or proof of a different fence-free decoder. |
| [M] Historical per-producer writeback cost and replacement by one last-arriver flush (`docs/STATUS.md:64`, `src/runtime/fleet_runtime.h:369–377`) | The major redundant producer writeback scheme was already removed. | New savings from removing it again. |
| [M] Old phase gaps of 0.4–1 µs, “events under 1 µs” (`docs/STATUS.md:280–281`) | Historical observed gaps. | A shipped per-event budget: the same document explicitly says no shipped per-layer timeline exists (`docs/STATUS.md:264–267`). |

**Revision reconciliation [L/R].** Building the exact d16 defaults in memory yields **54,946 descriptors and 1,589 events = 85 global + 1,504 local**, with validator success. The builder allocates the extra q_c publication events at `src/host/taskgraph.py:280–289`. Building the older combinations gives 1,346 events/58 globals for replicated kv_a without q_c publication, and 1,157 events/85 globals for shared kv_a without q_c publication. This explains the counts structurally; it does not authenticate the historic smoke binaries. `docs/design.md:219` also distinguishes historical smoke from the current graph. The brief's smoke numbers remain useful gross protocol budgets, but they are not paired v0.18 attribution.

**Measurement provenance [M/U].** The current raw summary reports 1.40/6.16 µs at 1.73 TB/s (`results/microbench_summary.txt:16–25`), unlike STATUS's 1.36/5.88 at 1.59. I retain STATUS's authoritative values and do not treat those two recordings as one experiment. Likewise, the raw expert handshake rows give 46.8−36.9 = 9.9 µs for per-worker release and 38.4−36.9 = 1.5 µs for last-worker release (`results/microbench_summary.txt:38–40`), rather than the older 13/6 µs source comment. Neither is a current isolated fence price.

**Timing windows [R].** The decoder's token timestamp starts inside embed, after the prior-done wait, and ends inside argmax, before its final event publication (`src/kernels/fleet_kernel.hip:70–71`, `src/kernels/fleet_kernel.hip:583`, `src/kernels/fleet_kernel.hip:978–982`, `src/kernels/fleet_kernel.hip:1021–1025`). The launcher computes the reported token interval from those timestamps (`src/host/fleet_launch.hip:791`), whereas smoke divides launch wall time by tokens (`src/host/fleet_launch.hip:766–781`). It reports both measures separately (`src/host/fleet_launch.hip:854–871`). Startup, final publication, and between-token waits are therefore real protocol work without a direct one-for-one subtraction from the headline token interval. Contention can still affect that interval; its effect is unmeasured.

### Every one of the 85 global events

The table of purposes below supplies the static production/consumption sites for **each event in the subsequent ID table**. IDs are generated by the current builder, which assigns them monotonically (`src/host/taskgraph.py:193–197`). Every listed event uses the common protocol sites inventoried after the tables. **Per-event elapsed cost for every entry is [U]; the measurements above are aggregate or different experiments.**

| Event family | Producers → consumers; ordering purpose | Graph/static payload sites |
|---|---|---|
| `embed` | One producer → all 296 L0 qkv descriptors. Publish the token's new residual before any L0 reader. | `src/host/taskgraph.py:479–485`; `src/kernels/fleet_kernel.hip:601`, `src/kernels/fleet_kernel.hip:707–719`. |
| `L*.qkv` | 296 producers, 37/XCD → 256 attention + 32 q_c publishers per layer. Shared kv_a needs all XCD shards; publishers consume their head's q. | Event/waits: `src/host/taskgraph.py:243–247`, `src/host/taskgraph.py:285–303`; writes/reads: `src/kernels/fleet_kernel.hip:647–652`, `src/kernels/fleet_kernel.hip:724–764`. |
| `L*.o_proj` | 296 producers → 296 norm/router descriptors. Publish all eight partials before their consumers fold them. | `src/host/taskgraph.py:358–363`, `src/host/taskgraph.py:394–396`, `src/host/taskgraph.py:459–462`; `src/kernels/fleet_kernel.hip:819–821`. |
| `L0.dense_gate_up` | 296 producers → 296 dense-down descriptors. Consumers need the full cross-XCD intermediate. | `src/host/taskgraph.py:444–473`; `src/kernels/fleet_kernel.hip:891–901`. |
| `L0.dense_down` | 296 producers → L1's 296 qkv descriptors. Publish the updated residual. | `src/host/taskgraph.py:471–488`; `src/kernels/fleet_kernel.hip:898–901`, `src/kernels/fleet_kernel.hip:707–719`. |
| `L1..L26.down` | 296 producers → next layer's 296 qkv descriptors, or the final 296 lm_head descriptors. Publish all expert partials before the next fold. | `src/host/taskgraph.py:423–440`, `src/host/taskgraph.py:483–494`; `src/kernels/fleet_kernel.hip:869–886`. |
| `lm_head` | 296 producers → one argmax. All per-worker candidates must exist before their reduction. | `src/host/taskgraph.py:491–496`; `src/kernels/fleet_kernel.hip:546–563`, `src/kernels/fleet_kernel.hip:921–922`. |
| `argmax` / done | One producer → next-token embed, and all schedulers' final termination test. Publish token_in and prevent premature completion. | `src/host/taskgraph.py:495–496`; `src/kernels/fleet_kernel.hip:580–583`, `src/kernels/fleet_kernel.hip:978–982`; `src/runtime/fleet_runtime.h:446`, `src/runtime/fleet_runtime.h:472–476`. |

All layer-table entries inherit the purpose, sites, and **[U] per-event cost** from their named family above.

| Layer | qkv event ID | o_proj event ID | down event ID |
|---|---:|---:|---:|
| L0 | 1 | 42 | 52 (dense_down); dense_gate_up is 51 |
| L1 | 53 | 94 | 111 |
| L2 | 112 | 153 | 170 |
| L3 | 171 | 212 | 229 |
| L4 | 230 | 271 | 288 |
| L5 | 289 | 330 | 347 |
| L6 | 348 | 389 | 406 |
| L7 | 407 | 448 | 465 |
| L8 | 466 | 507 | 524 |
| L9 | 525 | 566 | 583 |
| L10 | 584 | 625 | 642 |
| L11 | 643 | 684 | 701 |
| L12 | 702 | 743 | 760 |
| L13 | 761 | 802 | 819 |
| L14 | 820 | 861 | 878 |
| L15 | 879 | 920 | 937 |
| L16 | 938 | 979 | 996 |
| L17 | 997 | 1038 | 1055 |
| L18 | 1056 | 1097 | 1114 |
| L19 | 1115 | 1156 | 1173 |
| L20 | 1174 | 1215 | 1232 |
| L21 | 1233 | 1274 | 1291 |
| L22 | 1292 | 1333 | 1350 |
| L23 | 1351 | 1392 | 1409 |
| L24 | 1410 | 1451 | 1468 |
| L25 | 1469 | 1510 | 1527 |
| L26 | 1528 | 1569 | 1586 |

The remaining IDs are **embed = 0; lm_head = 1587; argmax = 1588**. Thus the enumeration is 27 qkv + 27 o_proj + 26 MoE down + 2 dense events + 3 endpoints = **85**. [L/R, from the cited builder sites.]

### Local events must also be charged to the protocol

For layer L, let B = 1 if L=0, otherwise B = 53 + 59(L−1). The following indexed sets enumerate all 1,504 local events, including their purpose. Their individual times are **[U]**.

| Event IDs / labels | Count/token; producers per event | Required handoff and static site |
|---|---|---|
| B+1+x, `merge.x{x}`, x=0..7 | 216; 32 producers | Completed head-output slices → same-XCD o_proj. `src/host/taskgraph.py:273–274`, `src/host/taskgraph.py:304–309`, `src/host/taskgraph.py:361–362`. |
| B+9+h, `qabs.h{h}`, h=0..15 | 432; 2 producers | Published q_c slices → 16 readers/head, waiting inside attention after kv_post. `src/host/taskgraph.py:283–303`; `src/kernels/fleet_kernel.hip:772–786`. |
| B+25+h, `attn.h{h}`, h=0..15 | 432; 16 producers | All attention partials of a head → that head's merge descriptors. `src/host/taskgraph.py:291–309`; `src/kernels/fleet_kernel.hip:694–699`. |
| B+42+x, `norm.x{x}`, x=0..7 | 216; 37 producers | Local normalized vector/router output → gate-up. Includes the dense layer. `src/host/taskgraph.py:388–396`, `src/host/taskgraph.py:428–431`, `src/host/taskgraph.py:456–469`. |
| B+50+x, `gate_up.x{x}.k0`, L=1..26, x=0..7 | 208; 148 wave arrivals | All local intermediate rows → down. This is one chunk despite retaining the wave-level publication machinery. `src/host/taskgraph.py:411–440`; `src/kernels/gemv.h:496–507`. |

[R] Count: 27×(8+16+16+8) + 26×8 = **1,504**. In particular, “85 events” means **85 global events**, not 85 executions of a fence and not the entire protocol.

### Every active fence/drain/acquire/release site on these paths

The counts below are **[L/R] dynamic source-level invocations on a successful token**, derived from the generated descriptors and the cited control flow. They are not instruction counts or critical-path time. Compiler-generated waits are [U].

| Primitive / site | Purpose and shipped multiplicity | Cost evidence |
|---|---|---|
| Role tickets/grid readiness: `src/runtime/fleet_runtime.h:304–326` | Two block barriers plus ticket/grid atomics and polling per workgroup **once per launch**, to establish roles and reject bad placement. | [U] Separate duration; do not charge as a per-event fence. Wrong-placement failure is recorded at `docs/STATUS.md:327`. |
| Wait-to-body `__syncthreads`: `src/kernels/fleet_kernel.hip:975–990` | 54,946/token. Makes thread 0's wait/acquire and ok flag visible before the other waves execute the task. | [U] Isolated duration. Existing placement is tested by the payload benchmark (`docs/STATUS.md:363`). |
| Body-to-signal `__syncthreads`: `src/kernels/fleet_kernel.hip:1021–1025` | 54,946/token. Joins the waves before thread 0 publishes. Completion across waves cannot be replaced by a counter increment alone. | [U] Isolated duration; aggregate smoke and payload evidence only. |
| Generic producer drain: `src/runtime/fleet_runtime.h:343–348`, `src/runtime/fleet_runtime.h:385–389` | 47,250/token = 24,570 global-event producer calls + 22,680 local-event producer calls. Complete producer stores before the local arrival counter. Gate-up has signal_event<0, so its generic call returns before draining. | [U] Distinct machine-code cost; [M] the barrier + thread-0 drain pattern is exercised at `bench/microbench.hip:258–275`. |
| Per-wave gate-up drain + relaxed agent arrival: `src/kernels/gemv.h:496–507` | 26×8×37×4 = **30,784/token**, one per wave for the single chunk. Needed today because each wave independently advertises its completed rows. | [U] Separate cost. This operation is inside a task body, so bodyless smoke does not reproduce it instruction-for-instruction: smoke substitutes a thread-0 add of four at `src/kernels/fleet_kernel.hip:1003–1010`. |
| Last-arriver global release: `src/runtime/fleet_runtime.h:225–227`, `src/runtime/fleet_runtime.h:391–395` | **666/token = 83×8 + 2**. The 83 distributed events flush once on each XCD; embed and argmax flush once each. All earlier drained producers of that XCD precede publication of its global share. | [M] Last-arriver payload test and historical optimization. [U] Per-release latency. Release is unconditional at `src/runtime/fleet_runtime.h:262–264`. |
| Scheduler global load → acquire → release → mirror store: `src/runtime/fleet_runtime.h:459–468` | Relay the producer's publication through a different atomic object. Acquire/release execute on a lane observing a changed value, not once per graph event overall. With 85 IDs/256 lanes each active lane owns one ID per scheduler. | [U] Number of polling passes, observed partial increments, and duration. The required relay fix is recorded at `docs/STATUS.md:346`. |
| Scheduler mirror read-back: `src/runtime/fleet_runtime.h:462`; scheduler barriers/backoff/stop: `src/runtime/fleet_runtime.h:449–476` | Compare against its own previously stored value; block barrier synchronizes the stop flag; polling/backoff controls progress. | [U] Individual costs. The self-read is the first removable candidate below; relay fences and stop barriers remain. |
| Global worker acquire: `src/runtime/fleet_runtime.h:229–231`, `src/runtime/fleet_runtime.h:419–430` | **24,353 descriptor waits/token**, plus one prior-done wait at each of 31 internal boundaries in a 32-token launch. Invalidates cached cross-XCD payload after the observed publication. | [M] Fenced payload visibility; [U] acquire-only time. Total global worker acquires/launch = 24,353×32 + 31. |
| Local worker acquire: `src/runtime/fleet_runtime.h:239–244`, `src/runtime/fleet_runtime.h:429` | **37,504/token = 30,592 descriptor waits + 6,912 q_c body waits**. L1-only invalidation plus the helper's explicit waits; the producer's L2 is shared. | [M] Same-XCD payload test (`results/microbench_summary.txt:13–14`); historical L1-only improvement (`docs/STATUS.md:66`). [U] Isolated helper cost. |
| q_c body wait and block join: `src/kernels/fleet_kernel.hip:776–781` | 6,912/token; included in the local-acquire count above, with one extra block barrier each. Ensures both publisher slices are ready. | [U] Shipped wait cost. Bodyless smoke skips this body wait (`src/kernels/fleet_kernel.hip:1001–1003`). |
| lm_head write→read block join: `src/kernels/fleet_kernel.hip:921–922` | 296/token before candidates read the just-produced logits (`src/kernels/fleet_kernel.hip:517–525`). | [U] Barrier cost. This is a real payload read-back, not scheduler bookkeeping. |

[R] Global descriptor-wait arithmetic: embed 296 + qkv 27×288 + o_proj 27×296 + dense_gate_up 296 + dense_down 296 + MoE down 26×296 + lm_head 1 = **24,353**. Wait loops use relaxed agent loads and `s_sleep` at `src/runtime/fleet_runtime.h:215–222`, `src/runtime/fleet_runtime.h:419–427`; arrival and publication atomics are relaxed agent operations at `src/runtime/fleet_runtime.h:388–394`. None is an implicit C++ acquire/release atomic to which the explicit fence can casually be declared a duplicate.

**Inactive sites [L/R].** The shipped graph has no SIGNAL_LAST or TOPK_PUBLISH descriptors, so `arrive_last`'s drain (`src/runtime/fleet_runtime.h:354–360`), router last-arriver acquire (`src/kernels/fleet_kernel.hip:443–451`), and smoke SIGNAL_LAST path (`src/kernels/fleet_kernel.hip:1012–1019`) contribute zero. One chunk means the later-chunk body waits at `src/kernels/fleet_kernel.hip:875–881` execute zero times. coherent_acts=false makes the extra cache-row release at `src/kernels/fleet_kernel.hip:766–770` inactive. Experimental local assembly is off (`src/kernels/fleet_kernel.hip:605–641`). Deleting inactive code earns zero shipped runtime savings.

Other task-internal LDS barriers support arithmetic scratch/reductions, not event publication; they are not proposed for removal. They remain outside this synchronization-boundary audit. [R, scope; examples `src/kernels/gemv.h:169–175`, `src/kernels/attention.h:49–56`.]

## 2. Which synchronization is provably redundant?

### Required visibility and ordering

| Candidate | Exact ordering property / decision |
|---|---|
| **Last-arriver L2 flush** | **Keep.** Drained stores are in the producer XCD's L2; the last-arriver release precedes the global share increment (`src/runtime/fleet_runtime.h:369–395`). Local arrival alone does not publish them to another XCD. Most decisively, removing producer writeback produced wrong tokens in 10/18 full launches even though the atomic-payload microbenchmark passed (`docs/STATUS.md:491–502`). Nothing in the tree supplies a replacement proof. |
| **Per-producer buffer_wbl2** | **Already removed; zero additional removal.** Every producer drains, but only the last local arrival flushes (`src/runtime/fleet_runtime.h:385–395`). Do not confuse per-producer drain with per-producer L2 writeback. This is the optimization already recorded at `docs/STATUS.md:64`. |
| **All-wave store completion** | **Keep the property.** An arrival must not advertise stores still outstanding in another wave. Current chunk publication enforces it with each wave's own wait; generic publication uses the block join then thread-0 drain (`src/kernels/gemv.h:501–504`; `src/kernels/fleet_kernel.hip:1021–1025`). A last-arriver flush cannot repair a producer counted before its stores arrive. [R] Four distinct arrivals are redundant when the only observable condition is “this whole one-chunk workgroup finished”; completion itself is not redundant. |
| **Thread-0 drain after a block barrier; waitcnts inside acquire helpers** | **No source-only deletion proof.** A compiler may emit overlapping waits, but source barriers are not an ISA listing. Inspect exact target code before calling an individual wait redundant. The existing placement has payload evidence; changing it does not inherit a proof just from a DAG (`docs/STATUS.md:363`, `docs/STATUS.md:405–410`; `src/runtime/fleet_runtime.h:239–244`, `src/runtime/fleet_runtime.h:343–348`). |
| **Scheduler acquire/release pair** | **Keep both.** It connects the producer's release on global_events to the worker's acquire on xcd_flags. Acquire and release around a relay do not cancel; this missing link was a recorded defect (`docs/STATUS.md:346`; `src/runtime/fleet_runtime.h:451–468`). |
| **Consumer global acquires** | **Keep for cross-XCD ordinary payloads.** Observing a fresh atomic counter alone does not invalidate stale payload cache lines; payload readers follow the acquire and the block join (`src/runtime/fleet_runtime.h:400–430`; `src/kernels/fleet_kernel.hip:984–990`). coherent_acts was not faster and cannot run with shared kv_a (`docs/STATUS.md:32`; `src/host/fleet_launch.hip:501–503`). |
| **Consumer local acquires / self-invalidation** | **Keep where peer payload is read.** Being a producer of one shard does not make a consumer the producer of every shard it reads. L1 may retain old data even on a shared L2; the existing local helper is already L1-only (`src/runtime/fleet_runtime.h:233–244`). Dropping this acquire formerly passed the CPU suite, which is why source checks were added (`docs/STATUS.md:405–410`). |
| **All 85 complete global events** | **No whole-event deletion proved.** Each has a remaining payload or termination consumer in §1. Same-worker queue order cannot replace a multi-producer readiness condition. [R, graph sites in §1.] |
| **Token-boundary wait / done** | **Keep the dependency.** [L] embed is worker (0,0), argmax is worker (0,24), not the same queue. Assignment is round-robin at `src/host/taskgraph.py:741–749`; the endpoint tasks are created at `src/host/taskgraph.py:479–480`, `src/host/taskgraph.py:495–496`. Done also protects scheduler termination; premature completion was a real failure (`docs/STATUS.md:328`). |
| **Current-row cache read-back** | **Already absent; zero additional removal.** The current position uses each task's LDS row. Reintroducing an unsynchronized read of another XCD's new cache write recreates the failure at `docs/STATUS.md:325`; the repair is described at `src/kernels/attention.h:20–23`. |
| **lm_head's own output read-back** | It feeds candidate generation after the block join; it is not a redundant event acknowledgement (`src/kernels/fleet_kernel.hip:517–549`, `src/kernels/fleet_kernel.hip:921–922`). Removing it would require changing computation, outside this task. [R] |

### Two redundant pieces within the existing contract

**A — Repeated scheduler read of its own mirror [R].** For a fixed XCD and event, one scheduler lane is the sole writer. The host constructs a unique global-ID list (`src/host/fleet_launch.hip:449–450`), and the scheduler assigns indices by lane (`src/runtime/fleet_runtime.h:459–468`). A relaxed re-read of the value that lane last stored enforces **no additional ordering property**. Retaining that value locally can make the same change/no-change decision, with exactly the same global read, relay fences, and atomic mirror store. Initial mirror state must be read once per launch: counters persist across launches and repeats (`src/host/fleet_launch.hip:732`, `src/host/fleet_launch.hip:749–769`).

**B — Four arrival signals for an indivisible single-chunk producer [R].** In the shipped graph there is one intermediate-readiness event per XCD/layer, counted at 37×4. Every down descriptor waits for all 148 arrivals before reading any of that intermediate; there is no second chunk to release early (`src/host/taskgraph.py:411–440`; `src/kernels/fleet_kernel.hip:865–883`). One arrival from each workgroup **after its existing all-wave join and drain** enforces the same complete-payload condition with count 37. The extra three arrivals per workgroup enforce nothing additional. This changes publication accounting, not payload arithmetic, task placement, or the required down acquire.

### Two narrower scope reductions are logical candidates, not approved fence deletions

**C — q_c publishers need only their XCD's qkv share [R].** A publisher reads its own head's q, not shared kv_a (`src/kernels/fleet_kernel.hip:647–652`; `src/kernels/attention.h:111–138`). Those q rows are written on its XCD (`src/kernels/fleet_kernel.hip:724–728`; ownership `src/host/taskgraph.py:283–289`). Thus waiting for seven remote qkv shares adds no input-readiness property for this consumer. Attention still needs the full global qkv event. Replacing these 864 global waits with a local-share wait and L1 acquire is a plausible scope contraction, but the descriptor format's current interpretation requires every wait to use the event's global scope/count (`src/host/taskgraph.py:504–509`, `src/host/taskgraph.py:682–688`). A deliberate, verified exception is necessary; simply changing the event scope is wrong.

**D — next-token embed could consume the local done share [R].** Argmax and embed are on different workers of XCD0; token_in is their carried payload (`src/kernels/fleet_kernel.hip:73`, `src/kernels/fleet_kernel.hip:582`). A local done counter reaches the prior epoch's target (epoch×1) after the generic producer drain but before its global writeback (`src/runtime/fleet_runtime.h:386–394`). The dependency therefore admits a local-readiness argument. However, the existing code explicitly treats done as a global acquire boundary for cached KV state (`src/runtime/fleet_runtime.h:247–266`). A scope change must establish that every later remote cache reader still passes an adequate retained global acquire and that early embed cannot overwrite prior-token data. Do not present this as a removable self-wait.

**Disposition [R]: A and B preserve the established publication mechanisms. C and D expose logically excess scope but change which acquire a consumer executes; they remain conditional pending the evidence in §3. No timing credit is assigned to any of them.**

## 3. Minimal edits and the cheapest local decision

No implementation edits were made. The regions below specify prospective edits; all checks performed here operated on existing sources or in-memory graph/model copies.

### A. Cache the scheduler's last mirrored value

- **Exact edit region:** `src/runtime/fleet_runtime.h:442–470`, replacing only the repeated `poll(&mirror[e])` comparison at line 462 with a remembered value for that lane's event. Seed it from the mirror once and update it after the existing store. With 85 events and kBlock=256 (`src/runtime/fleet_runtime.h:51`), a lane has at most one event. Preserve a generic fallback when n_global_events>blockDim.x, or use per-owned-event state; one scalar unconditionally reused for multiple events would be wrong. [R]
- **Correctness:** same sole writer, same observed global values, same changed-value fence/store sequence. Keep both relay fences, all producer writebacks, worker acquires, block stop barriers, and abort tests. This directly avoids both the missing-relay defect (`docs/STATUS.md:346`) and the fence-free decoder failure (`docs/STATUS.md:491–502`). [R]
- **Cheapest local decision:** assert ID uniqueness/one event per lane and compare original vs cached-state relay traces for unchanged values, partial shares, skipped intermediate values, and nonzero restart state. [L] Those in-memory checks passed; the modeled acquire/release/store sequences were identical. This decides the bookkeeping equivalence, not GPU speed. Counter lifetime evidence is `src/host/fleet_launch.hip:732–769`.
- **Remaining check:** target code inspection should confirm state does not spill and all required fence/store order remains. That is not a new payload memory-model assumption, but its performance is [U]. Compiler-stub results cannot establish production ROCm behavior (`tests/check_default_build.py:7–18`).

### B. Use the existing workgroup publication path for exactly one chunk

- **Exact edit region:** `src/host/taskgraph.py:411–440` only, conditional on n_chunks==1. Set the local event's producer count to n_gu; give gate-up that event as signal_event with LOCAL scope; clear CHUNK_SIGNAL; clear down's CHUNK_WAIT while preserving its wait_event and other flags. Finalize the wait-side counts as usual (`src/host/taskgraph.py:500–523`). Leave multi-chunk behavior intact. [R]
- **Why no payload-body edit is required:** clearing CHUNK_SIGNAL makes the existing call pass nullptr chunk counters (`src/kernels/fleet_kernel.hip:840–844`). Clearing CHUNK_WAIT leaves the existing one-iteration down path (`src/kernels/fleet_kernel.hip:865–886`). The unchanged workgroup barrier and generic signal call publish at `src/kernels/fleet_kernel.hip:1021–1025`, using the unchanged drain at `src/runtime/fleet_runtime.h:386`. [R]
- **Correctness:** every workgroup signals exactly once after all its payload stores; all 37 must arrive before down acquires. Readiness may be later than an early wave's arrival, never earlier than complete publication. All global event identities, global producer releases, and consumer L1/L2 acquires remain intact. The inherited generic publication sequence is exercised in `bench/microbench.hip:258–275`. This is not “remove all-wave draining”; it is “use the already-existing all-wave join before one arrival.” [R]
- **Cheapest local decision:** transform the shipped graph in memory, rerun finalize/validate, pack/unpack descriptors, and simulate exact counter targets over three epochs in forward/reverse/random order. [L] All passed with **208 events changed from 148 to 37 arrivals**, no new event, no deadlock, and every counter exact. The unchanged shipped graph also passed the same three-order checks. The model's scope and counter assertions are at `tests/test_queue_simulation.py:47–153`.
- **Failure-mode checks:** changing only the producer count or only the flags recreates early-release/hang bugs; count/scope consistency is checked at `src/host/taskgraph.py:648–688`, against the recorded wait-side-count failure at `docs/STATUS.md:321`. Preserve the original output rounding and arithmetic, avoiding the precision failures at `docs/STATUS.md:347`. [R]
- **Limit:** simulation explicitly cannot verify cache visibility (`tests/test_queue_simulation.py:13–15`). Before adopting a generated candidate on hardware, inspect exact-target barrier/drain code and require full payload/token/layer gates; zero stale words in a small benchmark alone is insufficient (`docs/STATUS.md:491–502`). No current candidate GPU result exists. [U]

**Qualification on the payload evidence [R].** In benchmark (d′), even the local consumer waits for published *after* the last-arriver writeback (`bench/microbench.hip:262–273`). Its passing local check supports the L1 acquire and barrier/drain placement; it is not an isolated test of a no-writeback local event. B inherits the generic LOCAL event path already used by the shipped decoder (`src/runtime/fleet_runtime.h:386–390`, `src/runtime/fleet_runtime.h:429`), rather than claiming (d′) alone proves a new path. Later publication after the workgroup join can also offset the saved arrival instructions; no speedup is guaranteed.

### C and D. Evidence needed before contracting acquire scope

**C exact regions:** `src/host/taskgraph.py:285–289` for the publisher's intended local-share wait; `src/host/taskgraph.py:500–509` and `src/host/taskgraph.py:682–688` for an explicit, narrowly validated wait-side exception. The actual consumer already dispatches wait_scope/wait_count through `src/kernels/fleet_kernel.hip:984–985`; `src/runtime/fleet_runtime.h:414–430` supports local-counter/L1 acquire. Derive the count from that qkv event's actual per-XCD producer share, not an unrelated signal-side field. Keep qkv's global event and every last-arriver release. [R]

C's cheapest local check is a source-backed ownership assertion: each publisher's head range belongs entirely to its XCD's qkv producers; it reads no shared kv_a; the local target equals the share; attention retains its global wait. A simulator must check that it may proceed after that share, before the other seven shares, without a lifetime violation. Existing validate rules reject the exception, so a clean default validator run is not this proof. [R; ownership sites in §2 and validator sites above.]

**D exact region:** only the special prior-done wait at `src/kernels/fleet_kernel.hip:978–982`, if a restricted shipped-config local-wait branch is justified. Keep argmax's global signal and all scheduler final-event logic unchanged. [R]

D's cheapest local check requires per-worker token cursors and separate local-arrival/global-publication steps, verifying token availability, residual lifetime, and eventual final termination. The present simulator advances epochs only after every queue finishes (`tests/test_queue_simulation.py:70–73`); it cannot decide this cross-token change. A same-XCD ownership assertion alone is insufficient. [R]

**Required new evidence for either scope contraction:** exact-target release/acquire code inspection; a complete payload/lifetime proof including subsequent cached-KV reads; and loaded producer/consumer diagnostics plus repeated full-decoder free-running and teacher-forced checks, 32/32 tokens and 27/27 layer gates, with deterministic outputs. Retain every producer writeback throughout. These address the known microbenchmark/full-decoder discrepancy rather than repeating the passed small test (`docs/STATUS.md:491–502`; full-device gate scope `docs/STATUS.md:432–442`). **No GPU-free check in this tree can resolve that hardware discrepancy or decide a speedup.** [R/U]

### Checks actually completed

[L] Read-only runs returned success: `tests/test_kernel_interface.py` **19/19**, `tests/test_queue_simulation.py` **35/35**, and `tests/test_validator.py` **18/18**, with bytecode generation disabled. The exact shipped d16 graph and candidate B each separately passed the in-memory three-epoch/three-order checks; candidate A passed the in-memory relay equivalence checks described above. No candidate C/D implementation or GPU test was run.

These are structural results. The interface gate checks fence presence in source (`tests/test_kernel_interface.py:153–167`); it does not measure or prove their semantics. HIP syntax checking likewise excludes device semantics (`scripts/hip_syntax_check.sh:96–101`). No repository code, graph binary, test, or other result file was changed.

## 4. Ceilings, shared pools, and subtraction

### The owner's gap and the smoke envelopes

[M/R] Use the measured shipped 5.20 GB/token and 4.2 TB/s streaming rate (`docs/STATUS.md:42–43`), with decimal units:

- Pure-bytes reference: **5.20 / 4.2 = 1.238095 ms/token**.
- Residual to baseline: **3.595 − 1.238095 = 2.356905 ms/token**, about **65.56%** of baseline.
- Historical smoke fence-associated difference: **2.840 − 2.200 = 0.640 ms/token**.
- Smoke residual after that subtraction: **2.840 − 0.640 = 2.200 ms/token**.

[R] The 2.356905 ms is the overall non-byte opportunity under that reference model, not protocol time. The 0.640 and 2.200 ms are parts of one historical bodyless measurement, not independent measured parts of the shipped 3.595 ms. Changing task-body execution changes contention and overlap; smoke also substitutes wave arrivals and omits q_c body waits (§1). Thus even these gross smoke pools are not rigorous candidate-specific ceilings on real-decoder savings.

[R] If, counterfactually, the entire 0.640 ms difference transferred into useful decoder savings, the subtraction would be **3.595−0.640 = 2.955 ms**, recovering only **0.640/2.356905 = 27.15%** of the owner's gap and leaving **1.716905 ms**. This is **not safe, not measured, and not a proposal**: required writebacks/acquires occupy that pool, and the historical fence-free boundary failed correctness (`docs/STATUS.md:491–502`).

[R] Assigning the entire 2.200 ms residual as a gain would give **3.595−2.200 = 1.395 ms**; there is no evidence any candidate can do this. Assigning both smoke pools would give **3.595−2.840 = 0.755 ms**, below the **1.238095 ms** byte reference. That contradiction within the assumed model demonstrates why gross protocol time cannot simply be subtracted from the decoder.

### Candidate-specific accounting

| Candidate | Pool and subtraction | Honest ceiling / result |
|---|---|---|
| A: cached scheduler mirror value | Before: 85 global-counter + 85 mirror loads/scheduler sweep; after: 85 global-counter loads, plus one mirror initialization per active lane/launch. [R] Remove **85 loads/scheduler sweep**, or **680 across one sweep by all eight schedulers**, from the recurring comparison work (`src/runtime/fleet_runtime.h:459–468`). Done/abort reads remain. If a scheduler makes P sweeps, its removed mirror reads are 85(P−1), not a fixed number/token. | Competes for an **unknown portion of the 2.200 ms non-fence smoke residual**, with possible contention interaction. Global polling, stores, relay fences, waits, and barriers still cost time. **No supported numerical latency gain or tight shipped ceiling.** |
| B: one workgroup arrival instead of four wave arrivals | Before **26×8×37×4 = 30,784** arrivals; after **26×8×37 = 7,696**; difference **23,088/token**, a **75% reduction in this arrival class**. Explicit source-level drain calls change by the same difference when wave drains are replaced by generic drains; this is not a 75% fence-time reduction. Events remain **1,589 total/85 global**, and global producer releases remain **666/token**. [R, §1 sites.] | Competes for arrival/polling work and store-completion overhead. The old smoke residual is only gross context because smoke replaces the real per-wave body path. No measured time for the removed 23,088 operations; no conversion of operation share into wall-time share. |
| C: publisher local-share wait, conditional | **27×32 = 864** global worker acquires/waits would become local ones; distributed qkv publication remains. [R] Generic global descriptor-acquire count would fall **24,353−864 = 23,489**, with 864 local acquires added. | Competes with a subset of acquire/relay waiting and contention costs; local wait/acquire still has cost. No isolated pool or measured gain; requires §3 evidence. |
| D: local prior-done wait, conditional | At most **31** internal boundaries per 32-token launch, or **31/32** substituted waits/token. Global done signal/release and scheduler termination remain. [R, `src/kernels/fleet_kernel.hip:978–982`.] | Only the token-boundary wait route and acquire scope can change, not all done-event work. Its direct waiting is outside the headline embed→argmax interval; wall-time and contention effects remain unmeasured (§1). Requires §3 evidence. |
| Per-producer writeback; current-row cache read-back; inactive branches | Already absent from the shipped path. | **0.000 ms incremental pool** for deleting these again. [R, §2 sites.] |

[R] The only defensible small numerical **screen**, rather than shipped ceiling, comes from the historical sub-1-µs phase gaps: erasing all 26 MoE gate-up→down handoff gaps at zero replacement cost would save **<26×1 µs = 0.026 ms**, giving **>3.569 ms** and **<0.724%** of baseline. That would consume **<1.104%** of the 2.356905 ms gap, leaving **>2.330905 ms**. The source is the pre-shipped timeline (`docs/STATUS.md:264–281`), so this does **not** bound B's in-body drain/arrival effects or establish current headroom. A nonzero replacement handoff further reduces that historical allowance.

[R] Do not assign **85×1.36 = 115.6 µs** or **85×5.88 = 499.8 µs** to the shipped graph. Those are merely products of unlike quantities: a round-trip benchmark scale times a count of fan-in events, excluding 1,504 local events and their multiple waiters. Likewise, **0.640/85** is not a measured fence price. None of the candidate rows may be summed with the gross smoke pools; their work overlaps those pools and each other.

## Ranked removable synchronization, pool, and cheapest check

Ranking is by proof strength and edit size, **not predicted speedup**. [R]

| Rank | Removal / status | Pool | Cheapest deciding check |
|---|---|---|---|
| **1** | **Repeated scheduler self-read**. Minimal bookkeeping change preserving all fences. | Unknown subpool of polling work; **85 recurring reads/scheduler sweep** removed, no measured ms. | Unique ID/lane ownership + trace equivalence including nonzero launch seed. **Passed locally**; inspect target code after an implementation. (§3 A.) |
| **2** | **Three extra arrivals per one-chunk gate-up workgroup**. Graph-only switch to the existing barrier/drain/local-publication path. | **23,088 arrivals/token** removed; at most the historical **<0.026 ms gap-only screen**, with in-body effects unknown. No whole global event removed. | In-memory descriptor transformation, finalize/validate, exact multi-epoch counter simulation. **Passed locally**; retain target-code/payload/full-decoder gates. (§3 B.) |
| **3, conditional** | q_c publisher's unnecessary wait for remote qkv shares / global acquire scope. | **864 global→local waits/token**, no priced subpool. | Source-backed ownership and deliberate local-share wait validation, then visibility evidence. **Not established for adoption.** (§3 C.) |
| **4, conditional** | Token-boundary relay/global acquire scope where the token consumer is local. Keep the wait and global done event. | **31 waits/32-token launch** substituted, no priced subpool. | Asynchronous cross-token lifetime model with split local/global publication, then visibility evidence. **Not established for adoption.** (§3 D.) |

[R] Plainly: the tree supports small, concrete reductions in bookkeeping and **zero demonstrated additional milliseconds of recovery**. It does not support a claim that removable synchronization accounts for the owner's approximately 2.36 ms. Even the broad 0.640 ms historical fence-associated pool includes mechanisms known to be necessary.

The closed alternatives stay closed: coherent_acts not faster, prefetch_next 4% slower, topk_published slower, and one launch/token no speedup (`docs/STATUS.md:29–32`). No precision change or renewed fence-free experiment is proposed.

## [U] What this tree does not verify

- **[U] Per-event/per-fence timings for v0.18.** The tree explicitly lacks a shipped per-layer timeline (`docs/STATUS.md:264–267`). Historical task busy/wait sums include bodies, dependencies, and concurrent workers; they cannot provide the requested individual elapsed costs.
- **[U] Raw pairing for the authoritative 3.593/3.595/3.596 ms result.** The currently named final-run JSONs instead record 3.62828/3.624/3.62804 ms medians (`results/decode_final_r1.json:4`, `results/decode_final_r2.json:4`, `results/decode_final_r3.json:4`). STATUS remains the authority for the shipped baseline (`docs/STATUS.md:24`, `docs/STATUS.md:80`); these raw files cannot establish its missing per-event attribution.
- **[U] Exact provenance of the historical smoke A/B binaries and disabled instructions.** Their recorded event counts differ from the current graph (`docs/STATUS.md:38`, `docs/STATUS.md:44`); current smoke skips q_c body waits and substitutes wave arrivals (`src/kernels/fleet_kernel.hip:1001–1010`). The audit therefore uses 0.640/2.200 ms as gross historical context only.
- **[U] Resolution of STATUS/raw microbenchmark differences.** STATUS's 1.36/5.88 µs and raw 1.40/6.16 µs are distinct records, not a single reproducible attribution (`docs/STATUS.md:40`; `results/microbench_summary.txt:16–25`).
- **[U] Scheduler sweep counts, number of partial increments observed, and exposed critical-path relay time.** The loop is data-dependent (`src/runtime/fleet_runtime.h:450–476`); neither 85 graph events nor 666 producer-XCD releases determines relay-fence executions.
- **[U] Exact production gfx942 codegen for redundant waitcnts, all-wave completion, and proposed scheduler state.** Source-level invocation counts do not prove machine instruction counts. Local compiler-stub artifacts explicitly do not establish production ROCm behavior (`tests/check_default_build.py:7–18`).
- **[U] Why removing producer writeback passes the atomic-payload microbenchmark but fails full decoding.** This remains unresolved in STATUS (`docs/STATUS.md:491–502`); CPU simulation and source spelling checks cannot supply missing hardware evidence.
- **[U] New-path GPU payload, full-token/layer correctness, determinism, and speed for A/B; stronger scope/lifetime proof and those same gates for C/D.** Structural checks completed here are not fresh GPU results (`tests/test_queue_simulation.py:13–15`; `docs/STATUS.md:432–442`).
- **[U] A tight positive numerical bound on all safely removable protocol time in the shipped decoder.** The operation reductions are established as described, but the tree contains no individual timing that converts them to milliseconds. Reporting a precise gain would invent evidence. [R]
