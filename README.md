# Batch-1 decode of DeepSeek-Coder-V2-Lite-Base on one MI300X

One token of decoding runs as **a single GPU kernel that never exits**.

Normally a decode step launches around 800 small kernels. Here 304 workgroups
stay resident on the GPU and hand work to each other through a task graph built
in advance on the host.

**It is 20% faster than vLLM on the same machine, and every token it produces
is identical to HuggingFace's.**

## Results

Measured on one MI300X, 1,024-token context, 32 greedy tokens, bf16, batch 1.

| | |
|---|---|
| **Time per token** | **3.60 ms** (278 tokens/second) |
| vLLM 0.11.2, same GPU, same model, same prompt | 4.52 ms (220 tokens/second) |
| HuggingFace eager, same again | 54.4 ms |
| Tokens matching HuggingFace exactly | **32 of 32** |
| Layer outputs matching on the first step | **27 of 27** |
| Kernel launches for all 32 tokens | **1** |

How that was reached, step by step, with the measurement behind each step:
[docs/STATUS.md](docs/STATUS.md).

### Is that fast?

It moves 5.20 GB of weights per token, measured, in 3.60 ms. That is 26% of the
GPU's peak memory bandwidth.

That sounds low. It is not. vLLM on the same machine reaches 21%. The published
figures near 50% are all **dense** models, which read one big contiguous block
of weights per token. This is a mixture-of-experts model: it reads many small
pieces, and about 40% of each token is spent in steps that move almost no data
at all. [docs/STATUS.md](docs/STATUS.md) has the evidence, including four
separate checks that the streaming code itself is near the hardware limit.

## The machine

Everything was run on a rented GPU box.

| | |
|---|---|
| GPU | 1 × AMD Instinct MI300X (gfx942) |
| Host | Intel Xeon 8470, 13 cores, 224 GiB RAM |
| Software | ROCm 7.2.4, Ubuntu 24.04 |
| Provider | [Hot Aisle](https://hotaisle.ai), $2.99/hour |
| Billing | **per minute**, minimum one minute |

Billing continues while the machine is powered off. Only deleting it stops the
charge.

You do not need a GPU to check most of this work. See below.

## How to run it

### Without a GPU

One command runs everything that does not need hardware.

```bash
python3 -m venv .venv && .venv/bin/pip install numpy torch transformers
.venv/bin/python tests/run_all.py
```

That is 19 checks: the maths against HuggingFace, the host/device wire format,
the task graph validator on all seven variants, the event protocol simulated
for three tokens, and the kernel parsed by a real compiler in both passes.

Two extra commands, both optional:

```bash
.venv/bin/python tests/run_all.py --mutate   # breaks the code on purpose, 12 ways
.venv/bin/python src/host/model_analysis.py  # where the 5 GB per token goes
```

The `--mutate` one matters. A test that passes proves nothing on its own, so
each of those 12 runs injects a realistic bug and requires the suite to catch
it. Two of our tests used to catch nothing at all; that is how we found out.

### With a GPU

One command sets the machine up from nothing and runs everything.

```bash
bash scripts/hotaisle_bootstrap.sh
```

It installs the dependencies, downloads the 31 GB model, builds both binaries,
generates the task graphs, runs the benchmarks, produces the HuggingFace
reference, packs the weights, and then runs the full GPU test suite. About 15
minutes.

To run pieces by hand afterwards:

```bash
# 19 GPU checks: memory model, placement, decode on every graph variant,
# determinism, and the abort path
python3 tests/run_gpu.py

# one decode, with a per-task timing trace
./build/fleet_decode_nt --graph build/taskgraph_d16.bin --json results/decode.json \
                        --trace results/trace.bin

# read that trace as one layer, phase by phase
python3 scripts/trace_timeline.py results/trace.bin build/taskgraph_d16.bin --layer 5
```

`fleet_decode_nt` is the shipped binary. `fleet_decode` is the same code without
non-temporal weight loads, which is 4% slower. The bootstrap builds both.

Pin `transformers>=4.39,<5`. The model ships its own modeling file, and it uses
symbols that transformers 5 removed.

## What is in the repo

```
src/host/       taskgraph.py        builds and validates the task graph
                pack_weights.py     one flat weight file the kernel reads directly
                reference_run.py    HuggingFace reference: tokens, layers, KV cache
                reference_decode.py the same maths in NumPy, task by task
                fleet_launch.hip    loads everything, launches the kernel, checks it
src/kernels/    fleet_kernel.hip    the kernel that never exits
                gemv.h              the streaming matrix-vector code
                attention.h         absorbed MLA attention
                expert.h            mixture-of-experts
src/runtime/    fleet_runtime.h     task queues, events, memory ordering
tests/          run_all.py          every check that needs no GPU
                run_gpu.py          every check that needs one
                mutate.py           breaks the code to prove the tests work
bench/          microbench.hip      memory-model proofs, bandwidth, occupancy
scripts/        hotaisle_bootstrap.sh   bare machine to results, unattended
results/        every measurement, with an index
```

## Reading further

- [docs/STATUS.md](docs/STATUS.md) — what is verified, what was tried and
  reverted, and what is still missing. **Read this before trusting any number.**
- [docs/design.md](docs/design.md) — why the kernel is shaped this way.
- [docs/task/](docs/task) — the original assignment and design proposal, kept
  unchanged so everything here can be checked against what was asked for.
- [results/README.md](results/README.md) — a map of the raw measurements.

## One thing worth knowing

Most ideas in this project were measured and thrown away. Publishing shared
values, tiling the expert phase, prefetching weights, cache hints, compile-time
specialisation, raising occupancy: all tried, all reverted, each with its number
recorded. Two of the estimates that looked best were wrong by a factor of two in
the same direction.

That record is in [docs/STATUS.md](docs/STATUS.md), and it is the most useful
part of this repository.
