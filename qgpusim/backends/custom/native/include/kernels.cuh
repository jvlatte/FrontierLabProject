#pragma once
#include <cuComplex.h>

__global__ void apply_1q_gate_kernel(
    cuDoubleComplex* psi,
    const cuDoubleComplex* U,
    long long n,
    int q
);

__global__ void apply_2q_gate_kernel(
    cuDoubleComplex* psi,
    const cuDoubleComplex* U,
    long long n,
    int q0,
    int q1
);
