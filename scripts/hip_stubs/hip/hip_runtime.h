// Minimal HIP stubs for syntax checking without ROCm.
//
// There is no ROCm toolchain on the development laptop, so kernels would be
// compiled for the first time on the MI300X — the most expensive place to
// discover a missing semicolon. These stubs let a stock clang that carries the
// AMDGPU backend parse the .hip sources *in HIP language mode*
// (`-x hip -nogpuinc`), so the compiler's own `__builtin_amdgcn_*`,
// `__hip_atomic_*`, `__attribute__((global/device/shared))` and inline-asm
// handling are exercised for real; only the runtime API and the header-level
// helpers that ROCm's headers would normally provide are stubbed here.
//
// What this CANNOT catch, and what therefore still has to be checked on device:
//   * semantics of the builtins and the inline asm (s_getreg_b32 HW_REG_XCC_ID)
//   * register pressure, LDS limits, occupancy
//   * whether hipExtMallocWithFlags / hipDeviceMallocUncached exist in the
//     installed ROCm version
//   * gfx942-specific codegen: the parse targets the newest gfx9 the local
//     clang knows, so `#if defined(__gfx942__)` branches are not seen
//
// Never link against these. Usage: scripts/hip_syntax_check.sh
#pragma once

#if !defined(__HIP__)
#error "these stubs are for clang in HIP language mode (-x hip); see scripts/hip_syntax_check.sh"
#endif

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>

// ---- qualifiers: the real attributes, so the compiler applies CUDA/HIP rules
#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __forceinline__ inline __attribute__((always_inline))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))

// ---- built-in vector types and coordinates
struct dim3 {
    unsigned x, y, z;
    __host__ __device__ dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1)
        : x(x_), y(y_), z(z_) {}
};
struct uint3 { unsigned x, y, z; };
struct uint4 { unsigned x, y, z, w; };
struct float4 { float x, y, z, w; };

struct __fleet_coord { unsigned x, y, z; };
extern __device__ const __fleet_coord threadIdx, blockIdx, blockDim, gridDim;

// ---- device intrinsics the real headers define on top of the builtins
static __device__ inline void __syncthreads() { __builtin_amdgcn_s_barrier(); }
static __device__ inline void __threadfence() { __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "agent"); }
static __device__ inline void __threadfence_block() { __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "workgroup"); }
static __device__ inline float __shfl_xor(float v, int, int = 64) { return v; }
static __device__ inline int __shfl_xor(int v, int, int = 64) { return v; }
static __device__ inline float __shfl_down(float v, int, int = 64) { return v; }

static __device__ inline unsigned atomicAdd(unsigned* p, unsigned v) {
    return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
static __device__ inline int atomicAdd(int* p, int v) {
    return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
static __device__ inline float atomicAdd(float* p, float v) {
    return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// device overloads of the libm names the kernels use (host libm stays host)
static __device__ inline float __expf(float x) { return __builtin_expf(x); }
static __device__ inline float rsqrtf(float x) { return 1.f / __builtin_sqrtf(x); }
static __device__ inline float sqrtf(float x) { return __builtin_sqrtf(x); }
static __device__ inline float fmaxf(float a, float b) { return a > b ? a : b; }
static __device__ inline float fminf(float a, float b) { return a < b ? a : b; }
static __device__ inline float __uint_as_float(unsigned u) { return __builtin_bit_cast(float, u); }
static __device__ inline unsigned __float_as_uint(float f) { return __builtin_bit_cast(unsigned, f); }
static __host__ __device__ inline int min(int a, int b) { return a < b ? a : b; }
static __host__ __device__ inline int max(int a, int b) { return a > b ? a : b; }

// ---- runtime API surface used by bench/ and src/host
enum hipError_t { hipSuccess = 0, hipErrorUnknown = 999 };
typedef struct ihipStream_t* hipStream_t;
typedef struct ihipEvent_t* hipEvent_t;
enum hipMemcpyKind { hipMemcpyHostToDevice, hipMemcpyDeviceToHost, hipMemcpyDeviceToDevice };
enum hipDeviceAttribute_t {
    hipDeviceAttributeWallClockRate, hipDeviceAttributeClockRate,
    hipDeviceAttributeCooperativeLaunch,
};
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
hipError_t hipGetLastError();
hipError_t hipMalloc(void**, size_t);
template <typename T> hipError_t hipMalloc(T** p, size_t n) {
    return hipMalloc(reinterpret_cast<void**>(p), n);
}
hipError_t hipExtMallocWithFlags(void**, size_t, unsigned);
hipError_t hipHostMalloc(void**, size_t, unsigned = 0);
hipError_t hipHostFree(void*);
hipError_t hipFree(void*);
hipError_t hipMemset(void*, int, size_t);
hipError_t hipMemsetAsync(void*, int, size_t, hipStream_t = nullptr);
hipError_t hipMemcpy(void*, const void*, size_t, hipMemcpyKind);
hipError_t hipDeviceSynchronize();
hipError_t hipGetDeviceProperties(hipDeviceProp_t*, int);
hipError_t hipDeviceGetAttribute(int*, hipDeviceAttribute_t, int);
hipError_t hipStreamCreate(hipStream_t*);
hipError_t hipStreamSynchronize(hipStream_t);
hipError_t hipEventCreate(hipEvent_t*);
hipError_t hipEventRecord(hipEvent_t, hipStream_t = nullptr);
hipError_t hipEventSynchronize(hipEvent_t);
hipError_t hipEventElapsedTime(float*, hipEvent_t, hipEvent_t);
hipError_t hipEventDestroy(hipEvent_t);
hipError_t hipLaunchCooperativeKernel(const void*, dim3, dim3, void**, size_t, hipStream_t);
hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int*, const void*, int, size_t);

// hipLaunchKernelGGL: type-check the kernel's arguments against its
// parameters without a launch configuration (no <<<>>> support here).
template <typename... P, typename... A>
static inline void __fleet_check_launch(void (*)(P...), A&&... a) {
    [](P...) {}(static_cast<A&&>(a)...);
}
#define hipLaunchKernelGGL(kernel, grid, block, shmem, stream, ...)            \
    ((void)dim3(grid), (void)dim3(block), (void)(shmem), (void)(stream),       \
     __fleet_check_launch(kernel, ##__VA_ARGS__))
