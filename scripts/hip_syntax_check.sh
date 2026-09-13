#!/usr/bin/env bash
# Parse the .hip sources with a host compiler and minimal HIP stubs.
#
# There is no ROCm toolchain on the laptop, so the first real compile happens on
# the MI300X — where the clock is running. This catches the ordinary C++ errors
# (braces, typos, wrong arity, template mistakes) before that, and says plainly
# what it cannot catch.
#
#   bash scripts/hip_syntax_check.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STUBS="$ROOT/scripts/hip_stubs"
# Prefer the platform compiler: a Homebrew clang first on PATH does not know
# where the macOS SDK headers live and dies on <stdio.h>.
if [[ -z "${CXX:-}" ]]; then
    if [[ -x /usr/bin/clang++ ]]; then CXX=/usr/bin/clang++; else CXX=clang++; fi
fi

command -v "$CXX" >/dev/null || { echo "no $CXX on PATH" >&2; exit 1; }

# `-x c++` on a .hip file also discards the implicit SDK include path on macOS,
# so <cstddef> stops resolving. Put it back explicitly.
#
# Pass it as an array: `${SDK:+-isysroot "$SDK"}` expands to a single word with
# an embedded space, and clang then looks for a sysroot literally named
# " /Library/...", warns, and carries on without it.
SDKFLAGS=()
if [[ "$(uname -s)" == "Darwin" ]] && command -v xcrun >/dev/null; then
    _sdk="$(xcrun --show-sdk-path 2>/dev/null || true)"
    [[ -n "$_sdk" && -d "$_sdk" ]] && SDKFLAGS=(-isysroot "$_sdk")
fi

echo "syntax check: $($CXX --version | head -1)"
echo "stubs:        $STUBS  (host-only, never linked)"

# Establish that the toolchain can compile *anything* before blaming our
# sources. On a machine with CommandLineTools but no Xcode, clang may fail to
# resolve <cstddef> for even a trivial file; reporting that as an error in
# microbench.hip would be actively misleading.
probe=$(mktemp -t hipprobe).cpp
printf '#include <cstddef>\nint main(){return 0;}\n' > "$probe"
if ! "$CXX" -fsyntax-only -std=c++17 "${SDKFLAGS[@]}" "$probe" >/dev/null 2>&1; then
    rm -f "$probe"
    cat <<'EOF'

SKIPPED: this toolchain cannot compile a trivial C++ file (<cstddef> not found),
so it cannot say anything about the .hip sources either. This is an environment
problem, not a code problem — on macOS it usually means CommandLineTools without
a full Xcode install.

Static checks that do run without a compiler:
  python3 tests/test_descriptor_layout.py   host/device wire format
  python3 tests/test_kernel_interface.py    symbols, task kinds, struct fields

First real parse happens on the MI300X via scripts/setup_env.sh, which compiles
bench/microbench.hip first so a failure there is a toolchain diagnosis.
EOF
    exit 0
fi
rm -f "$probe"
echo

fail=0
shopt -s nullglob
for src in "$ROOT"/bench/*.hip "$ROOT"/src/kernels/*.hip; do
    printf '  %-34s ' "${src#$ROOT/}"
    out=$("$CXX" -fsyntax-only -std=c++17 -x c++ \
            "${SDKFLAGS[@]}" \
            -I"$STUBS" -I"$ROOT/src" \
            -Wno-unused-value -Wno-unused-function -Wno-unused-variable \
            "$src" 2>&1)
    if [[ -z "$out" ]]; then
        echo "OK"
    else
        errs=$(printf '%s\n' "$out" | grep -c 'error:')
        if [[ "$errs" -gt 0 ]]; then
            echo "$errs error(s)"
            printf '%s\n' "$out" | grep 'error:' | head -12 | sed 's/^/      /'
            fail=1
        else
            echo "OK (warnings only)"
        fi
    fi
done

cat <<'EOF'

Not covered here — verify on device:
  * __builtin_amdgcn_* semantics and inline asm (s_getreg_b32 HW_REG_XCC_ID)
  * register/LDS pressure and occupancy (-Rpass-analysis=kernel-resource-usage)
  * availability of hipExtMallocWithFlags / hipDeviceMallocUncached in the
    installed ROCm
  * cooperative-launch residency for grid=304
EOF
exit $fail
