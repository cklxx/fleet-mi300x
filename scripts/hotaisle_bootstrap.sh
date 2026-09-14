#!/usr/bin/env bash
# Fresh Ubuntu + ROCm box (Hot Aisle 1x MI300X VM) to first decode, unattended.
#
# Differences from setup_env.sh, learned on the first day on the machine:
#   * system pip is PEP 668-managed and python3-venv is not installed;
#   * the ROCm torch wheel comes from download.pytorch.org/whl/rocm7.0;
#   * the model repo's modeling file declares `import flash_attn` under a
#     guard that transformers' import check ignores, so a stub package is
#     installed (is_flash_attn_2_available() stays False: eager attention).
#
#   rsync the repo to ~/fleet-mi300x, then:
#   nohup bash ~/fleet-mi300x/scripts/hotaisle_bootstrap.sh > ~/bootstrap.log 2>&1 &
set -uo pipefail
cd ~/fleet-mi300x
MODEL=${MODEL_DIR:-$HOME/models/dsv2-lite-base}
GRAPH=${GRAPH:-build/taskgraph_d8.bin}
log() { printf '\n=== %s %s\n' "$(date -u +%H:%M:%S)" "$*"; }

log "python"
sudo apt-get update -qq > /tmp/apt.log 2>&1; sudo apt-get install -y -qq python3.12-venv >> /tmp/apt.log 2>&1
python3 -m venv .venv && . .venv/bin/activate
pip install -q --upgrade pip
pip install -q numpy "huggingface_hub[hf_transfer]"

log "model download (background)"
mkdir -p "$MODEL"
HF_HUB_ENABLE_HF_TRANSFER=1 python3 - <<PY > /tmp/download.log 2>&1 &
from huggingface_hub import snapshot_download
snapshot_download("deepseek-ai/DeepSeek-Coder-V2-Lite-Base", local_dir="$MODEL",
                  allow_patterns=["*.json", "*.safetensors", "*.py", "tokenizer*"], max_workers=8)
print("MODEL-DONE")
PY
DL=$!

log "torch + transformers"
pip install -q --index-url https://download.pytorch.org/whl/rocm7.0 torch 2>&1 | tail -1
pip install -q "transformers>=4.39,<5" "tokenizers<0.20" accelerate safetensors 2>&1 | tail -1
SP=$(python3 -c "import site;print(site.getsitepackages()[0])")
mkdir -p "$SP/flash_attn" && echo "# stub: satisfies transformers check_imports only" > "$SP/flash_attn/__init__.py"
python3 -c "import torch; print('torch', torch.__version__, torch.version.hip, torch.cuda.is_available())"

log "local checks"
for t in test_descriptor_layout test_kernel_interface test_queue_simulation test_row_partition test_expert_addressing test_absorbed_equivalence test_reference_vs_hf; do
  python3 tests/$t.py > results/$t.log 2>&1 && echo "PASS $t" || echo "FAIL $t"
done

log "build"
mkdir -p build results
hipcc --offload-arch=gfx942 -O3 -std=c++17 bench/microbench.hip -o build/microbench 2>&1 | grep -E "error"
hipcc --offload-arch=gfx942 -O3 -std=c++17 -Rpass-analysis=kernel-resource-usage -Isrc \
  src/host/fleet_launch.hip src/kernels/fleet_kernel.hip -o build/fleet_decode 2> build/compile.log
grep -E "error" build/compile.log
grep -A12 "fleet_decode_step" build/compile.log | grep -E "VGPRs:|Scratch|Occupancy|SGPRs Spill|LDS" | sed -E "s/.*remark: +//; s/ \[-R.*//" | tr "\n" ";"; echo
cp build/compile.log results/kernel_resource_usage.txt
for c in 8 4 1; do python3 src/host/taskgraph.py --kv-chunks $c --emit build/taskgraph_d$c.bin | tail -1; done

log "smoke"
./build/fleet_decode --graph $GRAPH --smoke --tokens 8 2>&1 | grep -E "placement|launch of|per-token|abort|error"

log "waiting for the model"
wait $DL; tail -1 /tmp/download.log; du -sh "$MODEL"

log "reference run"
python3 src/host/reference_run.py --model "$MODEL" --out build/golden.npz 2>&1 | grep -vE "^\[transformers\]|Loading checkpoint" | tail -8
log "pack weights"
python3 src/host/pack_weights.py --model "$MODEL" --out build/weights.bin 2>&1 | tail -2

log "decode: teacher-forced, one launch for all tokens"
./build/fleet_decode --graph $GRAPH --teacher-force --repeat 2 --json results/decode_d8_teacher.json 2>&1 | grep -vE "^  layer" | tail -12
log "decode: free-running, one launch, with trace"
./build/fleet_decode --graph $GRAPH --repeat 2 --json results/decode_d8_free.json --trace results/trace_d8.bin 2>&1 | grep -vE "^  layer" | tail -30
log "decode: free-running, one launch per token (v1, for the comparison)"
./build/fleet_decode --graph $GRAPH --tokens-per-launch 1 --repeat 2 --json results/decode_d8_v1.json 2>&1 | grep -E "per-token|wall time|tokens:"
echo "BOOTSTRAP-DONE"
