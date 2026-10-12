# DeepSeek decode: current-build baseline and per-phase microseconds (2026-10-12)

Session `f085448b233a45d5ba8ada5ad271265f`, one DigitalOcean MI300X VF (`gpu-mi300x1-192gb-devcloud`,
atl1), commit `ce32cd5`, model DeepSeek-Coder-V2-Lite, batch 1, context 1024, 32 greedy tokens.

## 1. The shipped build, re-measured on this machine

Six free-running runs of the shipped binary with the default 16-chunk graph, no F, no K, plus one
teacher-forced run:

| run | device median_ms | wall ms/token | matched | layers |
|---|---|---|---|---|
| prerequisite_gate det0 | 3.54388 | 3.95867 | 32/32 | 27/27 |
| prerequisite_gate det1 | 3.54164 | 3.94216 | 32/32 | 27/27 |
| prerequisite_gate det2 | 3.5444 | 3.95005 | 32/32 | 27/27 |
| final_B det0 | 3.54244 | 3.92884 | 32/32 | 27/27 |
| final_B det1 | 3.54708 | 3.96579 | 32/32 | 27/27 |
| final_B det2 | 3.54192 | 3.95161 | 32/32 | 27/27 |
| final_B teacher | 3.5466 | 3.97446 | 32/32 | 27/27 |

**Read it as: device 3.5416–3.5471 ms, wall 3.9288–3.9745 ms per token, 32/32 tokens and 27/27 layers.**
On the wall basis that is **251.6–254.5 tok/s**; on the device basis, **281.9–282.4 tok/s**.

Two bases, two answers, and they must not be mixed:

- `docs/STATUS.md` records the shipped v0.18 as **3.593 / 3.595 / 3.596 ms**. Today's build measures
  **~3.542 ms device**, 51 us faster, on a different machine and in a different session. Per this
  repository's own rule, absolute milliseconds are not compared across sessions, so this is a new
  record of the shipped configuration, not a claimed gain.
- The baseline numbers in `docs/STATUS.md` have **no raw pairing file left in the tree**; the JSONs named
  beside them are September v0.16 artefacts. These seven runs are that missing raw record for this build.

## 2. Per-phase microseconds, aggregated over the whole step

From `artifacts/prerequisite_trace.trace` with `scripts/trace_phase_table.py` (new). Traced run, so this is
an **attribution screen, not a timing verdict**: the stamps perturb the schedule.

| phase | n | worker_us | per_task_us | in_layer_us | stage_us | prologue_us | finishes_at_us |
|---|---|---|---|---|---|---|---|
| EMBED | 1 | 5.5 | 5.5 | 5.5 | 0.0 | 0.0 | 5.5 |
| DENSE_GATE_UP | 296 | 7678.6 | 25.9 | 28.8 | 0.0 | 0.0 | 104.9 |
| DENSE_DOWN | 296 | 5760.1 | 19.5 | 21.9 | 0.0 | 0.0 | 127.5 |
| QKV_FUSED | 7992 | 130700.3 | 16.4 | 18.6 | 35088.4 | 56369.2 | 3333.8 |
| Q_ABSORB | 864 | 5128.5 | 5.9 | 7.1 | 0.0 | 0.0 | 3341.8 |
| ATTENTION | 6912 | 125679.3 | 18.2 | 20.2 | 39783.5 | 54100.2 | 3354.4 |
| MERGE_UV | 6912 | 67108.2 | 9.7 | 13.0 | 0.0 | 0.0 | 3365.5 |
| O_PROJ | 7992 | 59006.5 | 7.4 | 11.8 | 0.0 | 7071.5 | 3374.7 |
| NORM_ROUTER | 7992 | 86185.6 | 10.8 | 12.2 | 36839.4 | 58596.7 | 3387.7 |
| EXPERT_GATE_UP | 7696 | 237466.7 | 30.9 | 33.7 | 38267.4 | 46036.1 | 3420.7 |
| EXPERT_DOWN | 7696 | 136091.1 | 17.7 | 21.6 | 0.0 | 12763.2 | 3440.3 |
| LM_HEAD | 296 | 37029.4 | 125.1 | 130.2 | 0.0 | 2413.6 | 3571.3 |
| ARGMAX | 1 | 4.4 | 4.4 | 4.4 | 0.0 | 0.0 | 3576.8 |

`worker_us` counts every worker, so it is occupancy, not wall time; it does not sum to the step.
`in_layer_us` is the mean span of that phase inside one layer. Busiest single task: LM_HEAD 129 us, then
EXPERT_GATE_UP 35 us, DENSE_GATE_UP 28 us.

Step span on the device clock: **3576.8 us**.

## 3. Where the step goes, by contiguous section

Reading `finishes_at` as the step clock at which each section completes:

- **prologue, 0 → 127.5 us**: embed and the single dense layer.
- **core loop, 127.5 → 3333.8 us (3206 us)**: the 27 MoE layers, QKV through the last layer's QKV_FUSED.
- **tail, 3333.8 → 3576.8 us (243 us)**: the final layer's attention through argmax.

The core loop is **89.6%** of the step and it is where the 27 layers pipeline. The tail is 6.8%.

## 4. The two arms tested in the same session

Both were built for this and both were **rejected by the pre-registered screen**, so neither reached the
paired verdict:

| arm | phase screen, default graph | with the arm | verdict |
|---|---|---|---|
| F (per-XCD local fold) | QKV stage sum 367.11 us | 374.59 us | rejected: complete reader-ready/kv_post interval not shorter |
| K (local kv_a assembly) | attention stage sum 210.55 us | 219.62 us | rejected: same, plus gather/publication/q_c wait not repaid |

F's and K's own payload-verification executables both passed the full gates (32/32 tokens, 27/27 layers,
free and teacher-forced), so the code is correct; it is slower. This matches the pre-registered reject
condition in `results/session-fk-2026-10-12.md` — the scheme adds an assembly, a publication, a wait and a
staging tail that do not pay for themselves at these sizes.

## 5. What this says about 300 tok/s

Target 300 tok/s is 3.3333 ms. Against this session:

- device basis: 3.542 → 3.3333 needs **0.209 ms** (5.9%).
- wall basis: 3.95 → 3.3333 needs **0.617 ms** (15.6%).

The byte floor is 1.238 ms (5.20 GB per token at the measured 4.2 TB/s), so bandwidth is not the binding
constraint: **2.30 ms of the step is not bytes**, and no phase in the table above has been shown to give up
0.209 ms. The two candidates that were built to try are both slower. Until the non-byte time is decomposed
per phase, 300 tok/s has no path, only a target.

## [U]

- Single-token traces do not give a timing verdict; the stamps approximate publication, not cache residency.
- The seven baseline runs are `--repeat 1`; they are consistent to 5 us across runs but they are not a
  five-pair interleaved comparison.
- `worker_us` counts occupancy and cannot be turned into wall time without a critical-path model.
