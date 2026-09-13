// Stub of ROCm's hip_bf16.h: the type and the two conversions the kernels use.
// Syntax checking only; see hip_runtime.h in this directory.
#pragma once
#include "hip_runtime.h"

struct __hip_bfloat16 {
    unsigned short data;
};

static __host__ __device__ inline float __bfloat162float(__hip_bfloat16 a) {
    return __builtin_bit_cast(float, (unsigned)a.data << 16);
}

static __host__ __device__ inline __hip_bfloat16 __float2bfloat16(float f) {
    __hip_bfloat16 b;
    b.data = (unsigned short)(__builtin_bit_cast(unsigned, f) >> 16);
    return b;
}
