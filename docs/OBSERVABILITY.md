# Observability: making the optimisation space visible (2026-10-12)

Purpose: from one decode step's logs, a reader must see where the microseconds went, how much of each stage is
bytes, hops, machinery or waiting, which pool a candidate lever can touch, and whether a change moved any of it.
This is the contract for the logging we build. It serves both repositories (`fleet-mi300x` now; `etx` shares the
model).

The idea is not new from nothing. `etx` already has the strongest version: three stamps per task in
`etx/codegen/kernel.py` (`taken`, deps-ready, `done`), a byte spec (`docs/xcd-timeline-2026-10-11/bytes.json`)
and a decomposition (`docs/master-table-2026-10-11/master.py` -> `master.txt`) that splits every role into
`bytes [E] + hops [E] + machinery [R] + gap [M] = total`. This document generalises that model and fixes the
parts that have already cost us wrong answers.

## 1. Five blind spots we have already paid for

1. **Nothing inside a body is stamped.** A phase's body is one opaque interval. Any claim about its interior is
   source and ISA, not time, unless it is labelled `[M]`.
2. **The producer's arrive and the next queue fetch are invisible.** They happen after `done` and before the next
   `taken`, so they hide inside the next gap, and a queue or arrival cost is unattributable today.
3. **Stamps perturb the schedule.** A traced run is not a timed run. Verdicts come from unstamped runs; traces
   attribute only.
4. **An ambiguous basis has produced wrong numbers.** `device` vs `wall`, per-task vs per-step, mean vs
   critical path, `[M]` vs `[E]`. A number whose basis is not in the same record gets misread, by a human or by
   an agent. This has already happened twice on this project.
5. **Source-level reasoning does not prove the emitted order.** A reordered C++ load once produced a
   byte-identical instruction sequence. Only an ISA artefact catches that, so every build carries one.

## 2. What we record

### 2.1 One record per task, NDJSON, one line per task per step

| field | meaning |
|---|---|
| `step`, `layer`, `task`, `role` | identity, stable across runs |
| `xcd`, `worker`, `wave` | placement, so occupancy is derivable |
| `t_taken`, `t_deps_ready`, `t_done` | the three stamps, device clock, nanoseconds |
| `t_arrive`, `t_fetch` | **new**: producer arrival and the next queue fetch, closing blind spot 2 |
| `bytes_spec` | per-task bytes from the spec, split weight / activation / KV |
| `bytes_meas` | the counter row for the same task, where a counter run exists |
| `wait_class` | barrier, event, coherence-invalidate, queue, launch, or body |
| `stamped` | true if this run was traced; a stamped record may not decide a latency verdict |

### 2.2 One record per run, JSON, beside the log directory

The run record carries, in one object: conditions (GPU, rocm, batch, context, tokens), arm id, build identity
(source commit, kernel object hashes, graph hash, weight and golden hashes, and the **exact compiler flag list**),
the resource line (VGPR, AGPR, spill, LDS, occupancy), graph counters (`tasks`, `events`), the correctness
verdict (`matched`, `compared`, `layers_within_gate`, determinism), the log path, and whether the run was
stamped. An unknown field says `null`; it never guesses.

### 2.3 Phase summary, derived and regenerated

One row per role: `tasks | body [M] | bytes [E] | hops [E] | machinery [R] | gap [M] | total [M] | XCDs live |
per-layer body/gap`. One generator produces it from the records plus the byte spec, so the arithmetic is
reproducible, and the columns keep `etx`'s master-table names.

### 2.4 Two counts that describe the schedule's shape

- **Occupancy**: per-XCD busy fraction and the full-time-equivalent XCD count per role. This is what exposes a
  stage live on 3.67 of 8 XCDs while its body is trivial.
- **Idle time**: derived from the busy fraction, reported as a description of utilisation, never as removable
  latency.

### 2.5 One artefact per build

The compiler's resource report; an ISA order artefact for the stage's critical loads and waits (the load, the
wait and the barrier in emitted order, each with the source line that produced it); and the exact `-D` list the
build used.

## 3. What the reader gets

One command writes the summary; one command diffs two runs. The diff decides levers:

```
trace_diff control/ arm/ --by role --by xcd
  role         control_us   arm_us    delta   verdict
  expert_up        822.5     818.0     -4.5   below bar
  ...
  totals:            ...      ...     ...
  correctness: control 32/32 27/27 | arm 32/32 27/27 | determinism both
```

A lever is kept only when every paired delta is faster, the median clears **0.030 ms** and exceeds twice the
spread, and the correctness gates hold. The logs must make that verdict reproducible without the GPU.

## 4. Rules that make the logs trustworthy

1. A latency verdict comes from **unstamped** runs; a trace attributes only. Never mix the two bases.
2. Every number carries its basis and its label (`[M]`, `[E]`, `[R]`, `[U]`) in the same record.
3. The log directory is empty before an arm is compared. Stale logs have produced stale verdicts.
4. Full, unfiltered logs. Never grep errors away; a filtered log hides the failure that matters.
5. **Fetch before destroy.** Copy the session log directory into the tree before the machine is deleted. A
   destroyed box takes its numbers with it, and this project has already lost one session's raw evidence that way.
   The shipped baseline is already in that state: the only record of v0.18's 3.593 / 3.595 / 3.596 ms is
   `docs/STATUS.md:24,80`. The JSON files named beside it in `docs/STATUS.md:25` are September v0.16 artefacts
   (`results/decode_final_r*.json`, medians 3.628 ms), so the shipped number has no raw pairing file left. A run
   record in the tree would have prevented that.
6. The writer never invents: an unknown interval stays `[U]` and adds nothing to a total.
7. Traces attribute; the phase summary is regenerated by a generator, never edited by hand.
8. One log format for both repositories, so a fleet trace and an ETX trace are diffable by the same reader.

## 5. Build order

1. **Close blind spot 2** (`t_arrive`, `t_fetch`) and regenerate the phase summary on the shipped fleet build.
   That alone produces the missing current trace the F/K session needs as its prerequisite.
2. Write the run record and the phase-summary generator for fleet, seeded from the phase table in
   `docs/STATUS.md`.
3. Add the per-build artefacts: resource line, flag list, ISA order.
4. Write `trace_diff` and use it for F, K and FK.
5. Only then spend GPU time on new levers. The protocol pool and the byte floor are the two big ones, and both
   need the attribution this document defines.

## 6. What this does not do

- It does not make an unmeasured interval removable. `[U]` stays `[U]`.
- It does not replace the correctness gates; a fast wrong step is not a result.
- It does not license price inflation: a gross pool is still not a gain.
