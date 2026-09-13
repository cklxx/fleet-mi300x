// Minimal HIP stubs for host-side syntax checking only.
//
// There is no ROCm toolchain on the development laptop, so kernels are written
// blind and compiled for the first time on the MI300X. That is the most
// expensive place to discover a missing semicolon: the VM bills by wall-clock
// whether it is compiling or idle. These stubs let `clang++ -fsyntax-only`
// parse the .hip sources locally and catch ordinary C++ mistakes — unbalanced
// braces, template errors, typos, wrong argument counts.
//
// What this CANNOT catch, and what therefore still has to be checked on device:
//   * semantics of __builtin_amdgcn_* (they are declared, not modelled)
//   * inline asm validity (s_getreg_b32 etc.)
//   * register pressure, LDS limits, occupancy
//   * whether hipExtMallocWithFlags/hipDeviceMallocUncached exist in the
//     installed ROCm version
//
// Never link against these. Usage: scripts/hip_syntax_check.sh
#pragma once

#include <cstddef>
#include <cstdint>

// ---- qualifiers: no-ops for host parsing
#define __global__
#define __device__
#define __host__
#define __shared__ static
#define __constant__ const
#define __forceinline__ inline
#define __restrict__
#define __launch_bounds__(...)

// ---- built-in vector types
struct dim3 {
    unsigned x, y, z;
    dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};
struct float4 { float x, y, z, w; };
struct uint3 { unsigned x, y, z; };

// ---- built-in variables
extern uint3 blockIdx, threadIdx, blockDim, gridDim;

// ---- errors and runtime API surface used by bench/ and src/
typedef int hipError_t;
enum { hipSuccess = 0 };
typedef void* hipEvent_t;
typedef void* hipStream_t;

enum hipMemcpyKind { hipMemcpyHostToDevice, hipMemcpyDeviceToHost,
                     hipMemcpyDeviceToDevice };
enum hipDeviceAttribute_t { hipDeviceAttributeWallClockRate,
                            hipDeviceAttributeClockRate };
enum { hipDeviceMallocUncached = 1, hipDeviceMallocFinegrained = 2 };

struct hipDeviceProp_t {
    char name[256];
    char gcnArchName[256];
    int multiProcessorCount;
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
};

const char* hipGetErrorString(hipError_t);
hipError_t hipMalloc(void**, size_t);
template <typename T> hipError_t hipMalloc(T** p, size_t n) {
    return hipMalloc(reinterpret_cast<void**>(p), n);
}
hipError_t hipExtMallocWithFlags(void**, size_t, unsigned);
hipError_t hipFree(void*);
hipError_t hipMemset(void*, int, size_t);
hipError_t hipMemcpy(void*, const void*, size_t, hipMemcpyKind);
hipError_t hipDeviceSynchronize();
hipError_t hipGetDeviceProperties(hipDeviceProp_t*, int);
hipError_t hipDeviceGetAttribute(int*, hipDeviceAttribute_t, int);
hipError_t hipEventCreate(hipEvent_t*);
hipError_t hipEventRecord(hipEvent_t, hipStream_t = nullptr);
hipError_t hipEventSynchronize(hipEvent_t);
hipError_t hipEventElapsedTime(float*, hipEvent_t, hipEvent_t);
hipError_t hipEventDestroy(hipEvent_t);
hipError_t hipLaunchCooperativeKernel(const void*, dim3, dim3, void**, size_t,
                                      hipStream_t);

// hipLaunchKernelGGL: parse the kernel call and its arguments, discard the rest.
#define hipLaunchKernelGGL(kernel, grid, block, shmem, stream, ...) \
    (void)sizeof(grid), (void)sizeof(block), kernel(__VA_ARGS__)

// ---- device intrinsics (declared so calls type-check; semantics not modelled)
unsigned atomicAdd(unsigned*, unsigned);
int atomicAdd(int*, int);
float atomicAdd(float*, float);
void __syncthreads();
void __threadfence();
void __threadfence_block();
unsigned long long __builtin_amdgcn_s_memrealtime();
void __builtin_amdgcn_s_sleep(int);
void __builtin_amdgcn_s_setprio(int);
void __builtin_amdgcn_buffer_wbl2();
void __builtin_amdgcn_s_waitcnt(int);
// __builtin_nontemporal_load / __builtin_nontemporal_store are real clang
// builtins and must NOT be declared here — clang rejects a redeclaration.
