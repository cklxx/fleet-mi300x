#!/usr/bin/env python3
"""Run the production fold/gather bodies on CPU, including corruption checks.

This emulates lane execution, not GPU caches or fences. GPU publication is
tested separately by FLEET_VERIFY_LOCAL_ASSEMBLY under loaded full decode.
"""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def function(src, name):
    start = src.rfind("__device__", 0, src.index(name + "("))
    begin = src.index("{", start)
    depth = 1
    end = begin + 1
    while depth:
        depth += (src[end] == "{") - (src[end] == "}")
        end += 1
    return src[start:end]


def main():
    src = (ROOT / "src/kernels/fleet_kernel.hip").read_text()
    bodies = "\n".join(function(src, name) for name in (
        "stage_residual_f32", "stage_residual", "same_payload", "verify_local_fold"))
    begin = src.index("        case TASK_REDUCE: {")
    end = src.index("\n#else\n        case TASK_REDUCE:", begin)
    assembly = src[begin:end]
    # Also execute the actual attention consumer's K verification block.
    begin = src.index("                stage_vector(kv_a, s.part")
    end = src.index("\n#endif", begin)
    kva_check = src[begin:end]
    harness = r'''
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>
#include <cassert>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#define __device__
thread_local bool concurrent=false;
std::mutex mutex;
std::condition_variable cv;
int arrived=0,generation=0;
void cpu_barrier() {
    if(!concurrent) return;
    std::unique_lock<std::mutex> lock(mutex); int epoch=generation;
    if(++arrived==256) { arrived=0; ++generation; cv.notify_all(); }
    else cv.wait(lock,[&]{ return generation!=epoch; });
}
#define __syncthreads() cpu_barrier()
#define FLEET_LOCAL_FOLD 1
#define FLEET_LOCAL_KVA 1
#define __HIP_MEMORY_SCOPE_AGENT 0
#define __hip_atomic_load(p, order, scope) (*(p))
struct Dim { int x; }; thread_local Dim threadIdx; Dim blockDim{256};
constexpr int kBlockThreads=256, kNumXCDs=8, TASK_REDUCE=7;
constexpr int kLocalFoldFlag=4096, kLocalKvaFlag=8192, kFoldOproj=0;
constexpr unsigned kAbortLocalPayload=0x40000000u;
struct __hip_bfloat16 { uint16_t bits; };
struct alignas(16) float4 { float x,y,z,w; };
struct alignas(16) uint4 { uint32_t x,y,z,w; };
float from_bits(uint32_t u) { float f; std::memcpy(&f,&u,4); return f; }
uint32_t bits(float f) { uint32_t u; std::memcpy(&u,&f,4); return u; }
float __bfloat162float(__hip_bfloat16 b) { return from_bits(uint32_t(b.bits)<<16); }
float bf16_round(float f) { auto u=bits(f); return from_bits((u+0x7fff+((u>>16)&1))&0xffff0000); }
void unpack_bf16x2(uint32_t u, float& a, float& b) {
    a=from_bits(u<<16); b=from_bits(u&0xffff0000);
}
float ld_coh(const float* p, bool) { return *p; }
void st_coh(float* p,float v,bool) { *p=v; }
void stage_vector(const float* p,float* v,int n) {
    for(int i=threadIdx.x;i<n;i+=256) v[i]=p[i];
}
struct ModelDims { int hidden=2048,kv_lora=512,qk_rope=64; };
struct Activations { float* x[2]; float* x_norm; float* oproj_partial;
    __hip_bfloat16* expert_out; float* kv_a; };
struct RuntimeState { uint32_t* abort; };
struct Scratch { float vec[2048],part[2056]; };
struct Descriptor { int head,flags,index; };
void raise_abort(const RuntimeState& rt,uint32_t code) { if(!*rt.abort) *rt.abort=code; }
''' + bodies + r'''
void assemble(const ModelDims& d,Activations& act,Scratch& s,int xcd,int L,Descriptor t) {
    bool coh=false;
    switch(TASK_REDUCE) {
''' + assembly + r'''
    }
}
void check_kva(const ModelDims& d, Activations& act, Scratch& s,
               const RuntimeState& rt, int xcd, Descriptor t) {
    const float* kv_a=act.kv_a+(xcd+1)*576;
''' + kva_check + r'''
}
int main() {
    const ModelDims d; constexpr int H=2048;
    std::vector<float> x0(H),x1(H),norm(16*H+16,-999),op(8*H),kv(9*576+16,-999);
    std::vector<__hip_bfloat16> ex(8*H);
    for(int i=0;i<H;++i) { x0[i]=bf16_round((i%37-18)*0.03125f); x1[i]=bf16_round((i%53-26)*0.125f); }
    for(int x=0;x<8;++x) for(int i=0;i<H;++i) {
        // Cancellation and bf16 ties exercise the original ordered sums.
        op[x*H+i]=(x==0?65536.f:x==1?-65536.f:(x%2?1.f:-1.f))*(i%19-9)+0.00390625f*x;
        ex[x*H+i].bits=uint16_t(bits(bf16_round(op[x*H+i]))>>16);
    }
    for(int i=0;i<576;++i) kv[i]=float(10000*(i/72)+i);
    Activations act{{x0.data(),x1.data()},norm.data(),op.data(),ex.data(),kv.data()};
    Scratch s{}; uint32_t abort=0; RuntimeState rt{&abort};
    auto lanes=[&](auto body){ for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x) body(); };
    auto block=[&](auto body){
        std::vector<std::thread> threads;
        for(int lane=0;lane<256;++lane) threads.emplace_back([&,lane]{
            threadIdx.x=lane;concurrent=true;body();concurrent=false;
        });
        for(auto& thread:threads) thread.join();
    };
    const auto orig0=x0,orig1=x1,origkv=kv;
    for(int mode=0;mode<3;++mode) {
        // o_proj; next-layer expert fold; final lm_head expert fold.
        x0=orig0;x1=orig1;
        // vector assignment retains these sizes/capacities, but refresh pointers explicitly.
        act.x[0]=x0.data();act.x[1]=x1.data();
        std::vector<float> expected(H);
        lanes([&]{ if(mode==0) stage_residual_f32(d,act.x[0],op.data(),nullptr,expected.data(),0,0,false);
                    else stage_residual(d,act.x[1],ex.data(),nullptr,expected.data(),0,0,false); });
        for(int x=7;x>=0;--x) {
            block([&]{ assemble(d,act,s,x,mode==2?-1:2,{mode==0?0:1,kLocalFoldFlag,123}); });
            assert(!std::memcmp(norm.data()+(8+x)*H,expected.data(),H*4));
            assert(std::equal(norm.begin(),norm.begin()+8*H,std::vector<float>(8*H,-999).begin()));
            assert(std::all_of(norm.begin()+16*H,norm.end(),[](float f){return f==-999;}));
            assert(mode==0 ? x0==orig0 : x1==orig1);
            if(x!=0 || mode==2) assert(mode==0 ? x1==orig1 : x0==orig0);
            lanes([&]{ stage_vector(norm.data()+(8+x)*H,s.vec,H); });
            lanes([&]{ verify_local_fold(d,act,rt,s.vec,mode==0,123); });
            assert(abort==0);
            for(int bad: {0,255,256,2047}) {
                s.vec[bad]=from_bits(bits(s.vec[bad])^1);
                lanes([&]{ verify_local_fold(d,act,rt,s.vec,mode==0,123); });
                assert(abort==(kAbortLocalPayload|123)); abort=0;
                s.vec[bad]=from_bits(bits(s.vec[bad])^1);
            }
        }
        if(mode<2) assert(!std::memcmp((mode==0?x1:x0).data(),expected.data(),H*4));
    }
    for(int x=0;x<8;++x) {
        lanes([&]{ assemble(d,act,s,x,0,{2,kLocalKvaFlag,321}); });
        assert(!std::memcmp(kv.data()+(x+1)*576,origkv.data(),576*4));
        assert(!std::memcmp(kv.data(),origkv.data(),576*4));
        assert(std::all_of(kv.begin()+9*576,kv.end(),[](float f){return f==-999;}));
        lanes([&]{ check_kva(d,act,s,rt,x,{2,kLocalKvaFlag,321}); }); assert(abort==0);
        for(int shard=0;shard<8;++shard) {
            int i=(x+1)*576+shard*72+71; kv[i]+=1;
            lanes([&]{ check_kva(d,act,s,rt,x,{2,kLocalKvaFlag,321}); });
            assert(abort==(kAbortLocalPayload|321));abort=0;kv[i]-=1;
        }
    }
    std::puts("PASS: production F folds (3 boundaries x 8 XCDs), K all 8 shards x 8 XCDs; bounds, source lifetime, and 160 payload corruptions detected");
}
'''
    compiler = shutil.which("clang++") or shutil.which("c++")
    if sys.platform == "darwin" and Path("/opt/homebrew/opt/llvm/bin/clang++").exists():
        compiler = "/opt/homebrew/opt/llvm/bin/clang++"
    if not compiler:
        raise RuntimeError("a C++ compiler is required for the local payload test")
    sdk = (["-isysroot", subprocess.check_output(
        ["xcrun", "--show-sdk-path"], text=True).strip()] if sys.platform == "darwin" else [])
    with tempfile.TemporaryDirectory(prefix="fleet-local-payload-") as directory:
        path = Path(directory)
        (path / "payload.cpp").write_text(harness)
        subprocess.run([compiler, *sdk, "-std=c++17", "-O2", "-pthread", "-ffp-contract=off",
                        "-Wno-unknown-pragmas", str(path / "payload.cpp"),
                        "-o", str(path / "payload")], check=True)
        subprocess.run([str(path / "payload")], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
