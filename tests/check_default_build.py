#!/usr/bin/env python3
"""Compare v0.18, the new default, and explicit-off emitted bytes locally.

    .venv/bin/python tests/check_default_build.py --out /tmp/fleet-default-build
    python3 tests/check_default_build.py --native --out /tmp/fleet-native-build

Uses the repository HIP syntax stubs, never links or runs them. Besides both
preprocessor passes it compiles those unchanged streams into optimized IR,
gfx942 ELF device objects and the launcher's host object, for ordinary and
non-temporal weight builds. Clang's -fminimize-whitespace keeps harmless #line
formatting out of preprocessor output; a fixed per-TU -cuid and stdin compile
avoid command-line hashes and original-source-offset metadata. No generated
output is edited or filtered before the byte-for-byte comparisons.
These are real compiler outputs, but NOT production ROCm-linked binaries:
the stub math/shuffle/runtime declarations cannot establish device behavior.
Repeat a full hipcc executable comparison on the MI300X before measurement.
--native selects that production comparison (requires hipcc and ROCm; no GPU
execution is performed). It compares both TU objects and linked executables.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "85b087e223d43c1e677db19f9cdd74ca73b78532"
OFF = ["-DFLEET_LOCAL_FOLD=0", "-DFLEET_LOCAL_KVA=0",
       "-DFLEET_VERIFY_LOCAL_ASSEMBLY=0"]
SOURCES = ("src/kernels/fleet_kernel.hip", "src/host/fleet_launch.hip")


def compiler() -> str:
    candidates = [os.environ.get("HIP_SYNTAX_CLANG", ""),
                  "/opt/homebrew/opt/llvm/bin/clang++",
                  "/usr/local/opt/llvm/bin/clang++", "clang++"]
    for candidate in candidates:
        path = shutil.which(candidate) if candidate else None
        if not path:
            continue
        result = subprocess.run([path, "-print-targets"], capture_output=True, text=True)
        if result.returncode == 0 and "amdgcn" in result.stdout:
            return path
    raise RuntimeError("no clang with AMDGPU support; set HIP_SYNTAX_CLANG")


def run(command: list[str], cwd: Path, log: Path, stdin: Path | None = None) -> None:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                            input=stdin.read_text() if stdin else None)
    with log.open("a") as stream:
        redirect = f" < {shlex.quote(str(stdin))}" if stdin else ""
        stream.write(f"cwd={cwd}\n$ {shlex.join(command)}{redirect}\n")
        stream.write(result.stdout + result.stderr)
        stream.write(f"exit={result.returncode}\n\n")
    if result.returncode:
        raise RuntimeError(f"command failed; see {log}: {result.stderr[-1500:]}")


def build(root: Path, output: Path, cxx: str, explicit_off: bool) -> None:
    output.mkdir(parents=True)
    log = output / "commands.log"
    sdk = []
    if sys.platform == "darwin":
        sdk_path = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
        sdk = ["-isysroot", sdk_path]
    # Relative source/include paths keep __FILE__ and source_filename identical.
    common = [cxx, "-x", "hip", "--offload-arch=gfx942", "-nogpuinc", "-nogpulib",
              "-std=c++17", "-O3", *sdk, "-D__gfx942__=1",
              "-Iscripts/hip_stubs", "-Isrc", "-Wno-unused-command-line-argument"]
    if explicit_off:
        common += OFF
    for weights, defs in (("plain", []), ("nt", ["-DFLEET_NT_WEIGHTS=1"])):
        for source in SOURCES:
            for mode in ("device", "host"):
                name = f"{weights}.{Path(source).stem}.{mode}"
                command = [*common, *defs, f"--cuda-{mode}-only",
                           f"-cuid={Path(source).stem}"]
                preprocessed = output / f"{name}.ii"
                run([*command, "-E", "-P", "-fminimize-whitespace", source,
                     "-o", str(preprocessed)], root, log)
                # Kernel host IR needs a hipLaunchKernel declaration absent in
                # the syntax stubs. Do not alter those stubs just for this test.
                if mode == "host" and source == SOURCES[0]:
                    continue
                ir = output / f"{name}.ll"
                run([*command, "-S", "-emit-llvm", "-", "-o", str(ir)], root, log,
                    stdin=preprocessed)
                if mode == "device":
                    # The HIP driver tries the host macOS lld when linking a
                    # device image. Direct LLVM IR -> ELF needs no linker.
                    run([cxx, "-x", "ir", "--target=amdgcn-amd-amdhsa", "-mcpu=gfx942",
                         "-nogpulib", "-O3", "-c", str(ir), "-o", str(output / f"{name}.o")],
                        root, log)
                else:
                    run([*command, "-c", "-", "-o", str(output / f"{name}.o")], root, log,
                        stdin=preprocessed)
    # Includes the d16 shipped configuration and the builder's d8 CLI default.
    for chunks in (8, 16):
        extra = ", local_fold=False, local_kva=False" if explicit_off else ""
        code = (
            "from pathlib import Path; import runpy; "
            "m = runpy.run_path('src/host/taskgraph.py'); "
            f"cfg = m['load_cfg'](Path('reference/dsv2lite_config.json'), {chunks}{extra}); "
            "g = m['build'](cfg); assert not m['validate'](g); "
            f"m['emit'](g, Path({str(output / f'graph_d{chunks}.bin')!r}))"
        )
        run([sys.executable, "-c", code], root, log)


def build_native(root: Path, output: Path, hipcc: str, explicit_off: bool) -> None:
    output.mkdir(parents=True)
    log = output / "commands.log"
    for weights, defs in (("plain", []), ("nt", ["-DFLEET_NT_WEIGHTS=1"])):
        objects = []
        for source in SOURCES:
            target = output / f"{weights}.{Path(source).stem}.o"
            # IDs must match for the same TU even when -D flags/output paths
            # differ; default hashed IDs otherwise make off/on comparisons
            # fail before any code change is involved.
            run([hipcc, "--offload-arch=gfx942", "-O3", "-std=c++17", "-Isrc",
                 f"-cuid={Path(source).stem}", *defs, *(OFF if explicit_off else []),
                 "-c", source, "-o", str(target)], root, log)
            objects.append(str(target))
        run([hipcc, "--offload-arch=gfx942", *objects, "-o",
             str(output / f"{weights}.fleet_decode")], root, log)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default=BASELINE)
    parser.add_argument("--out", type=Path, required=True,
                        help="new directory for emitted artifacts and command logs")
    parser.add_argument("--native", action="store_true",
                        help="use hipcc/ROCm to compare native objects and linked executables")
    args = parser.parse_args()
    output = args.out.resolve()
    if output.exists():
        parser.error(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    cxx = shutil.which("hipcc") if args.native else compiler()
    if cxx is None:
        parser.error("--native requires hipcc in PATH")
    print(subprocess.check_output([cxx, "--version"], text=True).splitlines()[0], flush=True)
    with tempfile.TemporaryDirectory(prefix="fleet-v018-build-") as temp:
        baseline = Path(temp)
        archive = subprocess.check_output(["git", "archive", args.baseline_ref], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(baseline, filter="data")
        for label, root, off in (("baseline", baseline, False),
                                 ("default", ROOT, False), ("explicit-off", ROOT, True)):
            (build_native if args.native else build)(root, output / label, cxx, off)
            print(f"built {label}", flush=True)
    artifacts = sorted(path.name for path in (output / "baseline").iterdir()
                       if path.name != "commands.log")
    failures = []
    hashes = {}
    for name in artifacts:
        contents = [(output / label / name).read_bytes()
                    for label in ("baseline", "default", "explicit-off")]
        hashes[name] = {label: hashlib.sha256(data).hexdigest()
                       for label, data in zip(("baseline", "default", "explicit-off"), contents)}
        identical = contents[0] == contents[1] == contents[2]
        print(f"{'PASS' if identical else 'FAIL'} {name} {len(contents[0])} bytes "
              f"sha256={hashes[name]['baseline']}")
        if not identical:
            failures.append(name)
            if name.endswith((".ii", ".ll", ".txt")):
                diff = difflib.unified_diff(contents[0].decode().splitlines(),
                                            contents[1].decode().splitlines(),
                                            fromfile="baseline", tofile="default")
                (output / f"{name}.diff").write_text("\n".join(diff) + "\n")
    (output / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(f"{len(artifacts) - len(failures)}/{len(artifacts)} emitted artifacts byte-identical "
          "across v0.18/default/explicit-off")
    print("Scope: production ROCm objects and linked executables; no GPU execution."
          if args.native else
          "Scope: syntax-stub compiler outputs only; production ROCm executable comparison pending.")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
