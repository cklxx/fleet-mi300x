# Device-compile gate for F/K, local ROCm clang (2026-10-12)

Ran before provisioning, as `AGENTS.md` section 4 asks. Compiler: ROCm 7.2.4's own clang
(LLVM 22, `AMD clang version 22.0.0git roc-7.2.4`), the compiler the VM's `hipcc` runs on,
in the local `etx-rocm22` x86 container. Script: `scripts/device_gate_rocm.sh`.

**Scope, plainly:** this compiles the DEVICE code of `src/kernels/fleet_kernel.hip` for
`gfx942` and prints the resource report. It does **not** link the host executable, so it is
not a successful `hipcc` build of the full binary, and it is not a hardware test. The full
linked build remains the first step of the session's own prerequisite stage.

## Result: all five configurations compile

| config | defines | VGPRs | AGPRs | SGPR spill | scratch/lane | occupancy | LDS/block |
|---|---|---|---|---|---|---|---|
| basis | `FLEET_NT_WEIGHTS` | 256 | 86 | 730 | 48 | 1 | 25488 |
| F | + `FLEET_LOCAL_FOLD` | 256 | 89 | 630 | 48 | 1 | 25488 |
| K | + `FLEET_LOCAL_KVA` | 256 | 88 | 692 | 48 | 1 | 25488 |
| FK | + both | 256 | 91 | 740 | 48 | 1 | 25488 |
| F_verify | `FLEET_LOCAL_FOLD` + `FLEET_VERIFY_LOCAL_ASSEMBLY` | 256 | 92 | 650 | 48 | 1 | 25488 |

No register or LDS regression: VGPR, scratch, occupancy and LDS are identical across all five.
The SGPR spill count moves (730 basis, 630 F, 692 K, 740 FK) but the repository's own record says
spill counts do not predict this kernel's performance (`docs/STATUS.md:82`), so no gain or loss is
read from it here.

The emitted code does change with the flags: `s_waitcnt`/`ds_read` counts are 1636 basis, 1939 F,
1664 K, 1873 FK.This is a code-change check, not a timing claim.

## What this does not establish

- The host link and the production `hipcc` invocation.
- Any GPU visibility, correctness or performance of F and K.
- That the VM's ROCm runtime behaves identically; only that the device code compiles under the
  same compiler version the VM uses.
