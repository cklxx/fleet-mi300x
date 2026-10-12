#!/usr/bin/env bash
# Stage the five inputs tests/run_fk_session.py requires, on a fresh MI300X box,
# BEFORE the timed session starts. Produces, under the repository root:
#
#   build/weights.bin            packed weights      (src/host/pack_weights.py)
#   build/weights.bin.manifest   its manifest
#   build/fleet_cache.bin        converted KV cache   (src/host/kv_convert.py, via reference_run)
#   build/golden_tokens.txt      >= 33 greedy token ids
#   build/golden_hidden.bin      27 x 2048 fp32 = 221184 bytes
#
# Why this exists: tests/run_fk_session.py deliberately does NOT invoke
# scripts/hotaisle_bootstrap.sh (that flow filters logs and runs unrelated arms),
# so the input production had no owner. This script is that owner.
#
# Usage, on the machine, from the repository root, at the same commit as the controller:
#   bash scripts/stage_session_inputs.sh            # skip steps whose output already verifies
#   bash scripts/stage_session_inputs.sh --force    # redo everything
#
# It does not build kernels and does not touch the GPU scheduler. Nothing here is
# timed; the session clock starts later, in tests/run_fk_session.py.

set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

MODEL="${MODEL_DIR:-$HOME/models/dsv2-lite-base}"
MODEL_REPO="deepseek-ai/DeepSeek-Coder-V2-Lite-Base"
MIN_FREE_GB=40
START=$(date -u +%s)

log()  { printf '\n=== %s  [t+%ss] %s\n' "$(date -u +%H:%M:%S)" "$(( $(date -u +%s) - START ))" "$*"; }
die()  { printf 'STAGE FAIL: %s\n' "$*" >&2; exit 1; }
free_gb() { df -Pk . | awk 'NR==2{printf "%d", $4/1048576}'; }

# ---------------------------------------------------------------- 1. preflight
log "preflight"
[ -n "${HOME:-}" ] || die "HOME is unset"
echo "host:   $(uname -sr)"
echo "commit: $(git rev-parse HEAD)"
echo "model:  $MODEL"
echo "free:   $(free_gb) GiB"

[ -f .git/HEAD ] || die "not a git checkout; run from the repository"
command -v git  >/dev/null || die "git missing"
command -v pip3 >/dev/null || command -v python3 >/dev/null || die "python3 missing"

if command -v hipcc >/dev/null; then
  echo "hipcc:  $(hipcc --version 2>&1 | grep -m1 -i version || echo present)"
else
  echo "hipcc:  MISSING — the session needs it to build; install ROCm before the clock starts"
fi
if command -v rocminfo >/dev/null; then
  rocminfo 2>/dev/null | grep -m1 -o "gfx[0-9]*" || echo "rocminfo: no gfx target found"
else
  echo "rocminfo: MISSING"
fi

# ------------------------------------------------------- 2. python environment
log "python venv + packages"
if [ ! -d .venv ]; then
  python3 -m venv .venv || die "could not create .venv (apt install python3.12-venv first)"
fi
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q numpy "huggingface_hub[hf_transfer]"
python -m pip install -q --index-url https://download.pytorch.org/whl/rocm7.0 torch
python -m pip install -q "transformers>=4.39,<5" "tokenizers<0.20" accelerate safetensors
python - <<'PY'
import sys, torch
print("torch", torch.__version__, "hip", torch.version.hip, "cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("WARNING: torch reports no HIP device; reference_run needs one")
PY
# The model repo's modeling file imports flash_attn under a guard transformers'
# import check ignores, so a stub package is required (is_flash_attn_2_available stays False).
SP=$(python -c "import site;print(site.getsitepackages()[0])")
mkdir -p "$SP/flash_attn" && echo "# stub: satisfies transformers check_imports only" > "$SP/flash_attn/__init__.py"

# ------------------------------------------------------------ 3. model weights
log "model download ($MODEL_REPO)"
if [ -f "$MODEL/config.json" ] && [ "$FORCE" -eq 0 ]; then
  echo "already present ($(du -sh "$MODEL" 2>/dev/null | cut -f1)); use --force to refetch"
else
  mkdir -p "$MODEL"
  download() {
    HF_HUB_ENABLE_HF_TRANSFER="$1" python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("$MODEL_REPO", local_dir="$MODEL",
                  allow_patterns=["*.json", "*.safetensors", "*.py", "tokenizer*"], max_workers=8)
print("MODEL-DONE")
PY
  }
  # hf_transfer is a large speed-up but an optional wheel; fall back rather than die.
  download 1 || { echo "fast download failed; retrying without hf_transfer"; rm -rf "$MODEL"; mkdir -p "$MODEL"; download 0; }
fi
[ -f "$MODEL/config.json" ] || die "model incomplete at $MODEL"
echo "model: $(du -sh "$MODEL" | cut -f1)"

need=$(free_gb)
[ "$need" -ge "$MIN_FREE_GB" ] || die "only ${need} GiB free; need ${MIN_FREE_GB} GiB for weights + cache"

# --------------------------------------------------------- 4. golden + kv cache
mkdir -p build results
if [ "$FORCE" -eq 0 ] && [ -f build/golden_hidden.bin ] && [ -f build/fleet_cache.bin ] && [ -f build/golden_tokens.txt ]; then
  log "reference run: outputs already present, verifying instead"
else
  log "reference run (golden tokens, golden hidden, fleet cache)"
  python src/host/reference_run.py --model "$MODEL" --out build/golden.npz 2>&1 \
    | grep -vE "^\[transformers\]|Loading checkpoint" | tail -8
fi

# ------------------------------------------------------------------ 5. weights
if [ "$FORCE" -eq 0 ] && [ -f build/weights.bin ] && [ -f build/weights.bin.manifest ]; then
  log "pack weights: outputs already present, verifying instead"
else
  log "pack weights"
  python src/host/pack_weights.py --model "$MODEL" --out build/weights.bin 2>&1 | tail -2
fi

# ------------------------------------------------------- 6. verify as the session does
log "verify the five inputs (same checks as tests/run_fk_session.py)"
python - <<'PY'
import hashlib, json, sys
from pathlib import Path

ROOT = Path(".").resolve()
FILES = ["build/weights.bin", "build/weights.bin.manifest", "build/fleet_cache.bin",
         "build/golden_tokens.txt", "build/golden_hidden.bin"]
missing = [f for f in FILES if not (ROOT / f).is_file()]
if missing:
    print("FAIL missing:", ", ".join(missing)); sys.exit(1)

hid = ROOT / "build/golden_hidden.bin"
expect = 27 * 2048 * 4
if hid.stat().st_size != expect:
    print(f"FAIL golden_hidden.bin is {hid.stat().st_size} bytes, expected {expect}"); sys.exit(1)

toks = (ROOT / "build/golden_tokens.txt").read_text().split()
if len(toks) < 33:
    print(f"FAIL golden_tokens.txt has {len(toks)} ids, need >= 33"); sys.exit(1)

rows = []
for f in FILES:
    p = ROOT / f
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    rows.append({"file": f, "bytes": p.stat().st_size, "sha256_16": h})
    print(f"  OK {f:32s} {p.stat().st_size:>12,d} B  {h}")

(ROOT / "build" / "inputs.sha256").write_text(
    json.dumps({"files": rows, "tokens": len(toks), "hidden_shape": [27, 2048]}, indent=2) + "\n")
print(f"golden tokens: {len(toks)} ids (>= 33 required)")
print("wrote build/inputs.sha256")
PY

log "STAGE READY"
echo "inputs verified under $ROOT/build; copy build/inputs.sha256 into the session report"
echo "next: run tests/run_fk_session.py from the controller as documented in results/session-fk-2026-10-12.md"
