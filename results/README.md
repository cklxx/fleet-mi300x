# What is in here

Raw output from every machine session. Nothing here is edited by hand.

**Start with these.**

| file | what it is |
|---|---|
| `decode_final_r1..r3.json` | the shipped configuration, three runs |
| `decode_final_teacher.json` | same, teacher-forced |
| `baseline_vllm.json` | vLLM on the same GPU, model and prompt |
| `baseline_hf.json` | HuggingFace eager, same again |
| `rocprofv3_fetch_size_d16_shared.csv` | bytes actually fetched per token |
| `microbench_summary.txt` | memory-model proofs and bandwidth probes |
| `timeline_final_L5.txt` | one layer, phase by phase |
| `kernel_resource_usage.txt` | registers, spills, LDS, occupancy |

**Everything else is the audit trail.** Each optimisation was measured against
the version before it, and both were kept. `decode_v11_*` through
`decode_v16_*` are those pairs, in order. `trace_*` and `timeline_*` are the
per-task timings behind them. Several record changes that were **reverted**,
which is the point: `docs/STATUS.md` says which, and why.
