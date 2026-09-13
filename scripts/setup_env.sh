#!/usr/bin/env bash
# Environment setup for the MI300X box (docs/design.md D1).
#
# Billing note: a stopped Hot Aisle VM keeps billing, so this script is written
# to run start-to-finish unattended and leave a working tree that needs no
# repetition. Model download dominates wall-clock; everything else is seconds.
#
#   bash scripts/setup_env.sh            # full setup
#   bash scripts/setup_env.sh --no-model # skip the 31 GB download
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ID="deepseek-ai/DeepSeek-Coder-V2-Lite-Base"
MODEL_DIR="${MODEL_DIR:-$HOME/models/dsv2-lite-base}"
WANT_MODEL=1
[[ "${1:-}" == "--no-model" ]] && WANT_MODEL=0

log() { printf '\n=== %s\n' "$*"; }

log "host"
uname -a
log "ROCm / GPU"
if ! command -v rocm-smi >/dev/null; then
    echo "rocm-smi not found — this script expects to run inside a ROCm image" >&2
    echo "e.g. docker run -it --device=/dev/kfd --device=/dev/dri --group-add video \\" >&2
    echo "       -v \$HOME:\$HOME rocm/pytorch:latest" >&2
    exit 1
fi
rocm-smi --showproductname --showdriverversion 2>/dev/null | head -20
hipcc --version | head -3
echo "gfx arch: $(rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9a-z]*' || echo unknown)"

log "python deps"
python3 -m pip install --quiet --upgrade pip
# transformers is pinned: the reference modeling file ships 4.x-era imports
# (is_torch_fx_available and friends were removed in 5.x).
python3 -m pip install --quiet \
    "transformers>=4.39,<5" "tokenizers<0.20" accelerate safetensors numpy
python3 - <<'PY'
import torch, transformers
print(f"  torch {torch.__version__}  hip={getattr(torch.version,'hip',None)}")
print(f"  transformers {transformers.__version__}")
print(f"  cuda_available (=ROCm here): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  device: {p.name}  {p.total_memory/1e9:.1f} GB  CUs {p.multi_processor_count}")
PY

log "local checks that need no GPU (fail here = fix before spending GPU time)"
python3 "$REPO_ROOT/tests/test_descriptor_layout.py"
python3 "$REPO_ROOT/tests/test_kernel_interface.py"
python3 "$REPO_ROOT/tests/test_queue_simulation.py"
python3 "$REPO_ROOT/tests/test_row_partition.py"
python3 "$REPO_ROOT/tests/test_expert_addressing.py"
python3 "$REPO_ROOT/tests/test_absorbed_equivalence.py"
python3 "$REPO_ROOT/tests/test_reference_vs_hf.py"

log "build HIP"
mkdir -p "$REPO_ROOT/build" "$REPO_ROOT/results"
# Microbenchmarks first: they are self-contained, so a compile failure here is
# a toolchain problem, not a kernel problem.
hipcc --offload-arch=gfx942 -O3 -std=c++17 \
    "$REPO_ROOT/bench/microbench.hip" -o "$REPO_ROOT/build/microbench" 2>&1 | tail -20
echo "  compiled -> build/microbench"

# -Rpass-analysis prints VGPR/SGPR/LDS use: the design assumes one 256-thread
# workgroup per CU, and spilling past 512 VGPR+AGPR would break that (§11).
hipcc --offload-arch=gfx942 -O3 -std=c++17 \
    -Rpass-analysis=kernel-resource-usage \
    -I"$REPO_ROOT/src" \
    "$REPO_ROOT/src/host/fleet_launch.hip" \
    "$REPO_ROOT/src/kernels/fleet_kernel.hip" \
    -o "$REPO_ROOT/build/fleet_decode" 2>&1 | tail -40
echo "  compiled -> build/fleet_decode"

log "task graph"
python3 "$REPO_ROOT/src/host/taskgraph.py" --kv-chunks 1 \
    --emit "$REPO_ROOT/build/taskgraph_d2.bin"
python3 "$REPO_ROOT/src/host/taskgraph.py" --kv-chunks 4 \
    --emit "$REPO_ROOT/build/taskgraph_d4.bin"

log "protocol smoke test (no weights): residency, XCD roles, every event"
"$REPO_ROOT/build/fleet_decode" --graph "$REPO_ROOT/build/taskgraph_d2.bin" --smoke --tokens 4

# Everything above ran without the checkpoint; only now spend the download.
if [[ $WANT_MODEL -eq 1 ]]; then
    log "model: $MODEL_ID -> $MODEL_DIR (31 GB, the slow step)"
    mkdir -p "$MODEL_DIR"
    python3 -m pip install --quiet "huggingface_hub[hf_transfer]"
    HF_HUB_ENABLE_HF_TRANSFER=1 python3 - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("$MODEL_ID", local_dir="$MODEL_DIR",
                      allow_patterns=["*.json","*.safetensors","*.py","tokenizer*"],
                      max_workers=8)
print("  downloaded to", p)
PY
    du -sh "$MODEL_DIR"

    log "pack weights (one-time; excluded from decode latency by the task spec)"
    # Flattens the checkpoint into base+id*stride form and interleaves gate/up
    # rows so SiLU(gate)*up needs no cross-lane shuffle (§12).
    python3 "$REPO_ROOT/src/host/pack_weights.py" \
        --model "$MODEL_DIR" --out "$REPO_ROOT/build/weights.bin"
fi

log "done"
cat <<EOF
Next, in the order the design's D1 expects:

  1. ./build/microbench --json results/microbench.json
       Replaces the one low-confidence row in design.md §9 (synchronisation
       overhead) with a measurement, and picks between the two event schemes
       in §4. Needs no model weights.

  2. python3 src/host/reference_run.py --model $MODEL_DIR --out build/golden.npz
       Golden hidden states + 32 greedy tokens, and the two launcher inputs
       build/fleet_cache.bin and build/golden_tokens.txt from the same prefill.

  3. python3 src/host/kv_convert.py --model $MODEL_DIR --verify
       Confirms K,V rebuilt from the compressed cache match HF's own, <= 1e-2.

  4. ./build/fleet_decode --graph build/taskgraph_d2.bin --teacher-force
       Every step fed HF's token: a mismatch is isolated to the step it appears in.

  5. ./build/fleet_decode --graph build/taskgraph_d2.bin --json results/decode_d2.json
       Free-running 32-token decode, compared with HF greedy; latency per token.
EOF
