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
python3 tests/run_all.py | tail -14

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
python3 src/host/taskgraph.py --kv-chunks 16 --k-chunk 512 --emit build/taskgraph_d16_k512.bin | tail -1
python3 src/host/taskgraph.py --kv-chunks 16 --prefetch --emit build/taskgraph_d16_prefetch.bin | tail -1
python3 src/host/taskgraph.py --kv-chunks 16 --kva-replicated --emit build/taskgraph_d16_kvarep.bin | tail -1
python3 src/host/taskgraph.py --kv-chunks 16 --topk-published --emit build/taskgraph_d16_topkpub.bin | tail -1
python3 src/host/taskgraph.py --kv-chunks 16 --qc-published --emit build/taskgraph_d16_qc.bin | tail -1
python3 src/host/taskgraph.py --kv-chunks 16 --split-workers 18 --k-chunk 512 --emit build/taskgraph_d16_split18_k512.bin | tail -1
python3 src/host/taskgraph.py --kv-chunks 16 --split-workers 18 --emit build/taskgraph_d16_split18.bin | tail -1
python3 src/host/taskgraph.py --kv-chunks 16 --split-workers 22 --k-chunk 512 --emit build/taskgraph_d16_split22_k512.bin | tail -1

log "microbench"
./build/microbench --json results/microbench.json 2>&1 | tee results/microbench_summary.txt | grep -E "PASS|FAIL|INFO|median" | head -14

log "smoke (protocol only)"
./build/fleet_decode --graph $GRAPH --smoke --tokens 8 2>&1 | grep -E "placement|launch of|per-token|abort|error"
./build/fleet_decode --graph $GRAPH --smoke --tokens 8 --coherent-acts 2>&1 | grep -E "per-token|abort|error"

log "waiting for the model"
wait $DL; tail -1 /tmp/download.log; du -sh "$MODEL"

log "reference run"
python3 src/host/reference_run.py --model "$MODEL" --out build/golden.npz 2>&1 | grep -vE "^\[transformers\]|Loading checkpoint" | tail -8
log "pack weights"
python3 src/host/pack_weights.py --model "$MODEL" --out build/weights.bin 2>&1 | tail -2

log "gpu test suite (correctness before any timing)"
python3 tests/run_gpu.py | tail -26


BIN=./build/fleet_decode_nt      # non-temporal weight loads: the headline binary
log "decode: teacher-forced, one launch for all tokens (fenced protocol)"
$BIN --graph $GRAPH --teacher-force --repeat 2 --json results/decode_teacher.json 2>&1 | grep -vE "^  layer" | tail -6
log "decode: free-running, one launch, with trace (fenced protocol) -- the headline run"
$BIN --graph $GRAPH --repeat 3 --json results/decode_free.json --trace results/trace_free.bin 2>&1 | grep -vE "^  layer" | tail -22 | tee results/trace_free_summary.txt
python3 scripts/trace_timeline.py results/trace_free.bin $GRAPH --layer 5 | tee results/timeline_free_L5.txt

# ---- variants: each a full 32-token free-running decode checked against
# the golden tokens and the 27 layer boundaries (exit 0 only if all match)
log "variants"
run_variant() {   # name binary graph extra-args...
  local name=$1 bin=$2 graph=$3; shift 3
  printf '%-18s ' "$name"
  "$bin" --graph "$graph" --repeat 2 --json "results/decode_$name.json" "$@" > "results/decode_$name.log" 2>&1
  local rc=$?
  grep -E "^per-token" "results/decode_$name.log" | tail -1 | sed -E 's/per-token latency \(embed -> argmax on device\): //' | tr -d '\n'
  echo "   exit=$rc  $(grep -E '^tokens:' "results/decode_$name.log" | tail -1)  $(grep -E 'consecutive layers' "results/decode_$name.log" | tail -1)"
}

run_variant nt_d16_fenced    $BIN                 build/taskgraph_d16.bin
run_variant nt_d16_coherent  $BIN                 build/taskgraph_d16_kvarep.bin   --coherent-acts
run_variant nt_d16_fenced_b  $BIN                 build/taskgraph_d16.bin
run_variant nt_d16_coherent_b $BIN                build/taskgraph_d16_kvarep.bin   --coherent-acts
run_variant plain_d16        ./build/fleet_decode build/taskgraph_d16.bin
run_variant nt_d8            $BIN                 build/taskgraph_d8.bin
run_variant nt_d16_kchunk512 $BIN                 build/taskgraph_d16_k512.bin
run_variant nt_d16_prefetch  $BIN                 build/taskgraph_d16_prefetch.bin
run_variant nt_d16_v1        $BIN                 build/taskgraph_d16.bin          --tokens-per-launch 1
run_variant nt_d16_kvarep    $BIN                 build/taskgraph_d16_kvarep.bin
run_variant nt_d16_topkpub   $BIN                 build/taskgraph_d16_topkpub.bin
run_variant nt_d16_pfnext    $BIN                 build/taskgraph_d16.bin          --prefetch-next
run_variant nt_d16_pfnext_b  $BIN                 build/taskgraph_d16.bin          --prefetch-next
run_variant nt_d16_split18_k512 $BIN              build/taskgraph_d16_split18_k512.bin
run_variant nt_d16_split18   $BIN                 build/taskgraph_d16_split18.bin
run_variant nt_d16_split22_k512 $BIN              build/taskgraph_d16_split22_k512.bin
for v in kvarep split18_k512; do
  $BIN --graph build/taskgraph_d16_$v.bin --repeat 1 --trace results/trace_$v.bin > /dev/null 2>&1
  python3 scripts/trace_timeline.py results/trace_$v.bin build/taskgraph_d16_$v.bin --layer 5 | tee results/timeline_${v}_L5.txt
done

log "ISA: where the scratch comes from"
mkdir -p build/isa && (cd build/isa && hipcc --offload-arch=gfx942 -O3 -std=c++17 -DFLEET_NT_WEIGHTS=1 -I../../src -save-temps -c ../../src/kernels/fleet_kernel.hip -o fleet_kernel.o > /dev/null 2>&1)
ISA=$(ls build/isa/*gfx942*.s 2>/dev/null | head -1)
if [ -n "$ISA" ]; then
  grep -E "private_segment_fixed_size|vgpr_spill_count|sgpr_spill_count|\.vgpr_count|\.agpr_count" "$ISA" | head -8 | tee results/isa_summary.txt
  echo "scratch instructions: $(grep -cE '^\s+scratch_(load|store)' "$ISA")" | tee -a results/isa_summary.txt
  # the nearest preceding label / inlined-function comment of each scratch access
  grep -nE '^\s+scratch_(load|store)|^;.*(inline|Function)|^\.LBB' "$ISA" | grep -B1 scratch_ | grep -vE scratch_ | sort | uniq -c | sort -rn | head -10 | tee -a results/isa_summary.txt
  cp "$ISA" results/fleet_kernel_gfx942.s.txt
fi


log "bytes actually fetched (rocprof, 4 tokens, teacher-forced): best effort, rocprofv3 --kernel-trace segfaulted on ROCm 7.2.4"
if command -v rocprofv3 >/dev/null 2>&1; then
  timeout 600 rocprofv3 --pmc FETCH_SIZE -d results/prof -o fetch --output-format csv -- \
    $BIN --graph $GRAPH --tokens 4 --teacher-force > results/prof_fetch.log 2>&1 && \
    find results/prof -name "*counter_collection.csv" | head -1 | xargs -I{} sh -c 'grep fleet_decode_step {} | head -3 | cut -c1-200' || echo "rocprofv3 failed (see results/prof_fetch.log)"
fi
if command -v rocprof >/dev/null 2>&1; then
  printf 'pmc : FETCH_SIZE\n' > /tmp/fetch.txt
  timeout 600 rocprof -i /tmp/fetch.txt -o results/prof_fetch_v1.csv $BIN --graph $GRAPH --tokens 4 --teacher-force > results/prof_fetch_v1.log 2>&1 && \
    grep -i fleet_decode_step results/prof_fetch_v1.csv | head -3 | cut -c1-200 || echo "rocprof v1 failed (see results/prof_fetch_v1.log)"
fi
echo "BOOTSTRAP-DONE"
