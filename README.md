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
                taskgraph.py        builds/validates the task DAG, fans Chiplet-tasks
                                    out to per-worker descriptors, emits them
                reference_decode.py NumPy absorbed-MLA decode (the spec for each task)
                reference_run.py    HF golden states + 32 greedy tokens + the cache
                                    and token files the launcher decodes against
                kv_convert.py       prefill cache -> absorbed [layer][pos][576]
                pack_weights.py     flat weight blob + manifest; fused q‖kv_a;
                                    gate/up rows interleaved (§12)
                fleet_launch.hip    queues, buffers, weights, cooperative launch per
                                    token, HF comparison, abort decoding
src/runtime/    fleet_runtime.h     AOT queues, events, XCD role tickets, timeouts
                fleet_types.h       types shared by kernel and host launcher
src/kernels/    fleet_kernel.hip    the persistent kernel: scheduler/worker, dispatch
                gemv.h              wave-per-row streamed GEMV (q/kv_a, o_proj, experts, lm_head)
                attention.h         absorbed MLA: kv-post, q-absorb, flash-decoding, merge
                expert.h            MoE units, indirect expert index, dense MLP
bench/          microbench.hip      D1 (a) event cost (b) same-XCD cost (c) read BW
tests/          numerics, wire format, interface, queue-protocol simulation
scripts/        setup_env.sh        one-shot build + smoke test on the GPU box
                hip_syntax_check.sh parse every .hip in HIP mode with a stock clang
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
.venv/bin/python tests/test_kernel_interface.py      # kernel vs runtime vs launcher
.venv/bin/python tests/test_queue_simulation.py      # the event protocol, 3 tokens
.venv/bin/python tests/test_row_partition.py         # every GEMV row owned exactly once
bash scripts/hip_syntax_check.sh                     # clang front end, HIP mode
```

All of these pass today and none of them needs a GPU or the 31 GB checkpoint.
The last one needs a clang built with the AMDGPU backend (`brew install llvm`);
it parses the kernel, the launcher and the microbench in both the device and
the host compilation pass with only the runtime API stubbed, so the compiler's
own builtins, host/device rules and inline asm are checked for real.

### On the MI300X

```bash
bash scripts/setup_env.sh        # checks, deps, model, hipcc, graphs, smoke test, weights
./build/microbench --json results/microbench.json           # D1 (a)(b)(c)
python3 src/host/reference_run.py --model ~/models/dsv2-lite-base \
        --out build/golden.npz            # golden tokens, fleet_cache.bin, golden_tokens.txt
python3 src/host/kv_convert.py --model ~/models/dsv2-lite-base --verify
./build/fleet_decode --graph build/taskgraph_d2.bin --teacher-force   # step-isolated check
./build/fleet_decode --graph build/taskgraph_d2.bin --json results/decode_d2.json
```

Order matters. `setup_env.sh` ends with `fleet_decode --smoke`, which runs the
whole per-token protocol — cooperative launch at grid 304, XCD role discovery,
all 805 events — with every task body skipped, so the synchronisation cost is
measured before a single weight is downloaded. The microbenchmarks then decide
which event scheme is cheaper and what depth the GEMV should stream at. The
golden run comes next because every later boundary is judged against it.

`setup_env.sh` pins `transformers>=4.39,<5`: the vendored `modeling_deepseek.py`
imports symbols that transformers 5 removed. Results should be reproduced under
that pin.

## Why the numbers in the design are checkable

Two scripts exist so the design document cannot quietly be wrong:

`model_analysis.py` recomputes every byte figure in §1 from `config.json` —
4.935 GB per token, 31.41 GB resident, a 0.93 ms floor at 5.3 TB/s. `taskgraph.py`
builds the graph §3 describes and fails if any event has a producer count that
would deadlock the kernel: 66 logical tasks per MoE layer (114 with split-KV),
6 global events per layer, 1,791 logical tasks per token (3,087 with split-KV).
A Chiplet-task reaches the device as one descriptor per worker, so those
become 33,183 descriptors (2 MB) in 296 per-worker queues of 112–113 each.
`tests/test_queue_simulation.py` then *executes* those queues under the
kernel's event protocol for three tokens in adversarial worker orders — the
wiring between descriptor fields and wait targets is where the first version's
two deadlocks came from, and it is now checked without a GPU.

Building it that way already caught real problems: routing every
single-workgroup task to XCD 0 left a 25% imbalance across chiplets (rotating
them by layer index brought it to 0.1%), and the pre-review implementation is
catalogued in [docs/STATUS.md](docs/STATUS.md) — sixteen defects, two of which
meant the kernel was never launched and never waited on anything.
