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
GRAPH=${GRAPH:-build/taskgraph_d16.bin}
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
# second binary: non-temporal weight streams (§12 step 6), same everything else
hipcc --offload-arch=gfx942 -O3 -std=c++17 -DFLEET_NT_WEIGHTS=1 -Isrc \
  src/host/fleet_launch.hip src/kernels/fleet_kernel.hip -o build/fleet_decode_nt 2>&1 | grep -E "error"
for c in 16 8 4 1; do python3 src/host/taskgraph.py --kv-chunks $c --emit build/taskgraph_d$c.bin | tail -1; done
python3 src/host/taskgraph.py --kv-chunks 8 --prefetch --emit build/taskgraph_d8_prefetch.bin | tail -1

log "microbench"
./build/microbench --json results/microbench.json 2>&1 | tee results/microbench_summary.txt | grep -E "PASS|FAIL|median" | head -12

log "smoke"
./build/fleet_decode --graph $GRAPH --smoke --tokens 8 2>&1 | grep -E "placement|launch of|per-token|abort|error"
./build/fleet_decode --graph build/taskgraph_d8_prefetch.bin --smoke --tokens 8 2>&1 | grep -E "per-token|abort|error"
./build/fleet_decode --graph $GRAPH --smoke --tokens 8 --uncached-acts 2>&1 | grep -E "per-token|abort|error"

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

# ---- the §12 matrix: every variant is a full 32-token free-running decode
# checked against the golden tokens and the 27 layer boundaries (exit 0 only
# if all match). Columns: binary x graph x --uncached-acts x prefetch loads.
log "matrix"
run_variant() {   # name binary graph extra-args...
  local name=$1 bin=$2 graph=$3; shift 3
  printf '%-22s ' "$name"
  "$bin" --graph "$graph" --repeat 2 --json "results/decode_v11_$name.json" "$@" > "results/decode_v11_$name.log" 2>&1
  local rc=$?
  grep -E "^per-token" "results/decode_v11_$name.log" | tail -1 | sed -E 's/per-token latency \(embed -> argmax on device\): //' | tr -d '\n'
  echo "   exit=$rc  $(grep -E '^tokens:' "results/decode_v11_$name.log" | tail -1)  $(grep -cE '^  layer.* ok' "results/decode_v11_$name.log") layers ok"
}
run_variant base        ./build/fleet_decode    build/taskgraph_d8.bin
run_variant uncached    ./build/fleet_decode    build/taskgraph_d8.bin          --uncached-acts
run_variant nt          ./build/fleet_decode_nt build/taskgraph_d8.bin
run_variant prefetch    ./build/fleet_decode    build/taskgraph_d8_prefetch.bin
FLEET_PREFETCH_NT=1 run_variant prefetch_ntload ./build/fleet_decode build/taskgraph_d8_prefetch.bin
run_variant nt_prefetch ./build/fleet_decode_nt build/taskgraph_d8_prefetch.bin
run_variant all         ./build/fleet_decode_nt build/taskgraph_d8_prefetch.bin --uncached-acts
run_variant uncached_nt ./build/fleet_decode_nt build/taskgraph_d8.bin          --uncached-acts

log "trace of the base and the all-in variant"
./build/fleet_decode --graph $GRAPH --repeat 1 --trace results/trace_v11_base.bin 2>&1 | grep -vE "^  layer" | tail -22 > results/trace_v11_base_summary.txt
./build/fleet_decode_nt --graph build/taskgraph_d8_prefetch.bin --uncached-acts --repeat 1 --trace results/trace_v11_all.bin 2>&1 | grep -vE "^  layer" | tail -22 > results/trace_v11_all_summary.txt
tail -18 results/trace_v11_base_summary.txt

log "bytes actually fetched (rocprofv3 FETCH_SIZE, 4 tokens, base binary)"
if command -v rocprofv3 >/dev/null 2>&1; then
  rocprofv3 --pmc FETCH_SIZE --kernel-trace -d results/prof_base -o base --output-format csv -- \
    ./build/fleet_decode --graph $GRAPH --tokens 4 --teacher-force > results/prof_base.log 2>&1
  find results/prof_base -name "*counter_collection.csv" | head -1 | xargs -I{} sh -c "head -3 {}; grep -c fleet_decode_step {}"
else
  echo "rocprofv3 not installed"
fi

log "decode: free-running, one launch per token (v1, for the comparison)"
./build/fleet_decode --graph $GRAPH --tokens-per-launch 1 --repeat 2 --json results/decode_d8_v1.json 2>&1 | grep -E "per-token|wall time|tokens:"
echo "BOOTSTRAP-DONE"
