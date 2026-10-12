#!/usr/bin/env bash
# Device-compile gate for the F/K arms with ROCm 7.2.4's own clang (LLVM 22), the compiler
# the VM's hipcc runs on. Same stand-in the ETX project uses
# (examples/vllm_llm/rocm_build_check.sh:49-56): ROCm clang plus the ROCm headers tree.
#
# Scope, stated plainly: this compiles the DEVICE code of src/kernels/fleet_kernel.hip for
# gfx942 and prints the resource usage. It does not link the host executable, so a successful
# run here is not a successful hipcc build of the full binary. No GPU, no network.
set -uo pipefail

REPO=fleet-mi350x  # placeholder overwritten below
REPO=fleet-mi300x
OUT=build/rocm-device-gate-20261012
mkdir -p "$HOME/code/$REPO/$OUT"

docker run --rm --pull=never --network=none --platform linux/amd64 \
  -v "$HOME/code:/code" -w "/code/$REPO" -e OUT="$OUT" etx-rocm22 bash -c '
set -u
R=/code/rocm-headers
RL=$R/rocm-llvm/opt/rocm-7.2.4/lib/llvm
DL=$R/devlibs/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode
mkdir -p "$OUT"
echo "clang: $($RL/bin/clang++ --version | head -1)"

compile() {
  name="$1"; shift
  extra=""
  for d in "$@"; do extra="$extra -D$d=1"; done
  echo
  echo "=== device compile $name"
  # shellcheck disable=SC2086
  $RL/bin/clang++ -x hip --cuda-device-only --offload-arch=gfx942 -O3 -std=c++17 \
    -nogpuinc --rocm-device-lib-path=$DL -isystem $R/inc \
    -include __clang_hip_runtime_wrapper.h -D__HIP_PLATFORM_AMD__ -Isrc $extra \
    -Rpass-analysis=kernel-resource-usage -S src/kernels/fleet_kernel.hip \
    -o "$OUT/$name.s" 2> "$OUT/$name.log"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "  FAILED rc=$rc"
    grep -E "error" "$OUT/$name.log" | head -10 | sed "s/^/    /"
    return
  fi
  echo "  ok -> $OUT/$name.s ($(stat -c%s "$OUT/$name.s") bytes of assembly)"
  grep -E "VGPRs:|SGPRs Spill|Scratch|Occupancy|LDS|AGPRs:" "$OUT/$name.log" \
    | sed -E "s/.*remark: +//; s/ \[-R.*//" | sed "s/^/    /" | head -20
}

compile basis
compile F FLEET_LOCAL_FOLD
compile K FLEET_LOCAL_KVA
compile FK FLEET_LOCAL_FOLD FLEET_LOCAL_KVA
compile F_verify FLEET_LOCAL_FOLD FLEET_VERIFY_LOCAL_ASSEMBLY
'
echo
echo "exit: $?"
