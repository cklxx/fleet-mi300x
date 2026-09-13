# Fleet-style batch-1 decode — DeepSeek-Coder-V2-Lite-Base on one MI300X

One decode step executed as a single persistent HIP kernel: GPU workers stay
resident and coordinate through an on-device task graph, instead of launching
~800 separate kernels per token.

- **Design:** [docs/design.md](docs/design.md) — model analysis, task graph,
  synchronisation strategy, memory plan, correctness methodology, expected
  performance, risks.
- **Current state:** [docs/STATUS.md](docs/STATUS.md) — what is verified, what
  is untested, and what is known to be broken. Read this before the numbers.
- **Originals:** [docs/task/](docs/task) — the assignment as given
  (`CandidateTaskAMD.pdf`) and the design proposal this implements
  (`fleet_dsv2lite_mi300x_design.pdf`), kept verbatim so everything above can be
  checked against what was actually asked for.

Target: single MI300X (gfx942), bf16, batch 1, 1,024-token context, 32 greedy
tokens. No tensor parallelism, no continuous batching, no speculative decoding.

## Layout

```
src/host/       model_analysis.py   per-token HBM byte accounting from config.json
                taskgraph.py        builds/validates the task DAG, emits descriptors,
                                    reports Fleet-native vs fallback coverage
                reference_decode.py NumPy absorbed-MLA decode (the spec for each task)
                reference_run.py    HF golden states + 32 greedy tokens
                kv_convert.py       prefill cache -> absorbed [layer][pos][576]
                pack_weights.py     flat weight blob; gate/up rows interleaved (§12)
                fleet_launch.hip    per-worker queues, buffers, cooperative launch
src/runtime/    fleet_runtime.h     AOT queues, events, XCD discovery
                fleet_types.h       types shared by kernel and host launcher
src/kernels/    fleet_kernel.hip    the persistent kernel: scheduler/worker, dispatch
                gemv.h              register-streamed GEMV (q/kv_a, o_proj, experts, lm_head)
                attention.h         absorbed MLA: kv-post, q-absorb, flash-decoding, merge
                expert.h            MoE units, indirect expert index, dense MLP
bench/          microbench.hip      D1 (a) event round-trip (b) XCD flag (c) GEMV BW
tests/          boundary + layout + interface checks
scripts/        setup_env.sh        one-shot build on the GPU box
```

## Running it

### Locally (no GPU)

```bash
python3 -m venv .venv && .venv/bin/pip install numpy torch transformers
.venv/bin/python src/host/model_analysis.py          # byte accounting
.venv/bin/python src/host/taskgraph.py --report      # task graph + DAG validation
.venv/bin/python tests/test_absorbed_equivalence.py  # absorbed MLA == naive
.venv/bin/python tests/test_reference_vs_hf.py       # NumPy vs HuggingFace
.venv/bin/python tests/test_descriptor_layout.py     # host/device wire format
.venv/bin/python tests/test_kernel_interface.py      # kernel vs runtime header
```

All of these pass today and none of them needs a GPU or the 31 GB checkpoint.

### On the MI300X

```bash
bash scripts/setup_env.sh        # ROCm check, deps, model, hipcc, task graph, weights
./build/microbench --json results/microbench.json           # D1 (a)(b)(c)
python3 src/host/reference_run.py --model ~/models/dsv2-lite-base \
        --out build/golden.npz                              # golden states
python3 src/host/kv_convert.py --model ~/models/dsv2-lite-base --verify
./build/fleet_decode --graph build/taskgraph_d2.bin         # residency + queues
```

Order matters. The microbenchmarks come first because they need no weights and
they decide two things the kernel is built around: which of the two event
schemes in §4 is cheaper, and what load depth the GEMV should stream at. The
golden states come next because every later boundary is judged against them.

`setup_env.sh` pins `transformers>=4.39,<5`: the vendored `modeling_deepseek.py`
imports symbols that transformers 5 removed. Results should be reproduced under
that pin.

## Why the numbers in the design are checkable

Two scripts exist so the design document cannot quietly be wrong:

`model_analysis.py` recomputes every byte figure in §1 from `config.json` —
4.935 GB per token, 31.41 GB resident, a 0.93 ms floor at 5.3 TB/s. `taskgraph.py`
builds the graph §3 describes and fails if any event has a producer count that
would deadlock the kernel: 66 tasks per MoE layer, 114 with split-KV, 6 global
events per layer, 3,087 tasks per token.

Building it that way already caught one real problem: routing every
single-workgroup task to XCD 0 left a 25.3% imbalance across chiplets. Those
tasks signal global events, so their placement is free — rotating them by layer
index brought it to 0.8%.
