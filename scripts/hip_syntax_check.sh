#!/usr/bin/env bash
# Parse the .hip sources in HIP language mode with a stock clang + stubs.
#
# There is no ROCm toolchain on the laptop, so the first real compile happens on
# the MI300X — where the clock is running. Any clang built with the AMDGPU
# backend (Homebrew's llvm is) can still run the *front end* over HIP code:
# `-x hip -nogpuinc -nogpulib -fsyntax-only` exercises the real
# `__builtin_amdgcn_*` / `__hip_atomic_*` builtins, the host/device call rules,
# `__shared__`/`__launch_bounds__`, and inline asm parsing, with only the
# runtime API stubbed (scripts/hip_stubs). Each file is checked in both the
# device and the host compilation pass, as hipcc would.
#
#   bash scripts/hip_syntax_check.sh
#   HIP_SYNTAX_CLANG=/path/to/clang++ bash scripts/hip_syntax_check.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STUBS="$ROOT/scripts/hip_stubs"

# Find a clang that knows the amdgcn target. Apple's does not.
candidates=("${HIP_SYNTAX_CLANG:-}" /opt/homebrew/opt/llvm/bin/clang++
            /usr/local/opt/llvm/bin/clang++)
# Versioned installs, newest first: older LLVMs lack the __HIP_MEMORY_SCOPE_*
# macros the runtime uses.
while IFS= read -r c; do candidates+=("$c"); done < <(
    ls -d /opt/homebrew/opt/llvm@*/bin/clang++ /usr/lib/llvm-*/bin/clang++ 2>/dev/null \
        | sort -t@ -k2 -V -r)
candidates+=(clang++)
CXX=""
for c in "${candidates[@]}"; do
    [[ -n "$c" ]] || continue
    if command -v "$c" >/dev/null 2>&1 && "$c" -print-targets 2>/dev/null | grep -q amdgcn; then
        CXX="$c"; break
    fi
done
if [[ -z "$CXX" ]]; then
    echo "SKIPPED: no clang with the AMDGPU backend found (brew install llvm)" >&2
    exit 0
fi

# The parse still needs the host's C/C++ headers (macOS: the SDK sysroot).
SDKFLAGS=()
if [[ "$(uname -s)" == "Darwin" ]]; then
    _sdk="$(xcrun --show-sdk-path 2>/dev/null || true)"
    [[ -z "$_sdk" || ! -d "$_sdk" ]] && _sdk="$(ls -d /Library/Developer/CommandLineTools/SDKs/MacOSX*.sdk 2>/dev/null | tail -1)"
    [[ -n "$_sdk" && -d "$_sdk" ]] && SDKFLAGS=(-isysroot "$_sdk")
fi

# Newest gfx9 this clang accepts: gfx942 if it knows it, else gfx90a. Both are
# wave64 with the same builtins; only `#if defined(__gfx942__)` differs.
ARCH=""
for a in gfx942 gfx90a gfx908; do
    if echo "" | "$CXX" -x hip --cuda-device-only --offload-arch=$a -nogpuinc -nogpulib \
            -fsyntax-only "${SDKFLAGS[@]}" - >/dev/null 2>&1; then
        ARCH=$a; break
    fi
done
[[ -n "$ARCH" ]] || { echo "SKIPPED: $CXX accepts none of gfx942/gfx90a/gfx908" >&2; exit 0; }

echo "syntax check: $($CXX --version | head -1)"
echo "target:       $ARCH (front end only, -fsyntax-only)"
echo "stubs:        ${STUBS#$ROOT/}  (runtime API only; builtins are the compiler's)"
echo

fail=0
for src in "$ROOT"/bench/microbench.hip "$ROOT"/src/kernels/fleet_kernel.hip \
           "$ROOT"/src/host/fleet_launch.hip; do
    for mode in --cuda-device-only --cuda-host-only; do
        printf '  %-30s %-19s ' "${src#$ROOT/}" "${mode#--cuda-}"
        # -D__gfx942__=1: parse the gfx942-only branches (the XCC_ID inline
        # asm) even when the local clang only knows an older gfx9.
        out=$("$CXX" -x hip $mode --offload-arch=$ARCH -nogpuinc -nogpulib \
                -std=c++17 -fsyntax-only "${SDKFLAGS[@]}" -D__gfx942__=1 \
                -I"$STUBS" -I"$ROOT/src" \
                -Wall -Wextra -Wno-unused-parameter -Wno-unused-function \
                -Wno-unused-variable -Wno-missing-field-initializers \
                "$src" 2>&1)
        errs=$(printf '%s\n' "$out" | grep -c 'error:')
        warns=$(printf '%s\n' "$out" | grep -c 'warning:')
        if [[ "$errs" -gt 0 ]]; then
            echo "$errs error(s), $warns warning(s)"
            printf '%s\n' "$out" | grep -E 'error:' | head -15 | sed 's/^/      /'
            fail=1
        elif [[ "$warns" -gt 0 ]]; then
            echo "OK, $warns warning(s)"
            printf '%s\n' "$out" | grep -E 'warning:' | head -8 | sed 's/^/      /'
        else
            echo "OK"
        fi
    done
done

cat <<'EOF'

Not covered here — verify on device:
  * semantics of the builtins and the inline asm (s_getreg_b32 HW_REG_XCC_ID)
  * register/LDS pressure and occupancy (-Rpass-analysis=kernel-resource-usage)
  * availability of hipExtMallocWithFlags / hipDeviceMallocUncached in the
    installed ROCm
  * cooperative-launch residency for grid=304
EOF
exit $fail
