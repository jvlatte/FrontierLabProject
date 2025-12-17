#include <cuComplex.h>
#include "kernels.cuh"

__global__
void apply_1q_gate_kernel(
    cuDoubleComplex* psi,
    const cuDoubleComplex* U,
    long long n,
    int q
){
    unsigned long long dim = 1ULL << n;
    unsigned long long stride = 1ULL << q;
    unsigned long long num_pairs = dim >> 1;

    unsigned long long k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= num_pairs) return;

    unsigned long long block  = k / stride;
    unsigned long long offset = k % stride;

    unsigned long long i0 = block * 2ULL * stride + offset;
    unsigned long long i1 = i0 + stride;

    cuDoubleComplex a0 = psi[i0];
    cuDoubleComplex a1 = psi[i1];

    cuDoubleComplex u00 = U[0];
    cuDoubleComplex u01 = U[1];
    cuDoubleComplex u10 = U[2];
    cuDoubleComplex u11 = U[3];

    psi[i0] = cuCadd(cuCmul(u00, a0), cuCmul(u01, a1));
    psi[i1] = cuCadd(cuCmul(u10, a0), cuCmul(u11, a1));
}

__global__
void apply_2q_gate_kernel(
    cuDoubleComplex* psi,
    const cuDoubleComplex* U,
    long long n,
    int q0,
    int q1
){
    int low  = q0 < q1 ? q0 : q1;
    int high = q0 < q1 ? q1 : q0;

    unsigned long long dim = 1ULL << n;
    unsigned long long num_blocks = dim >> 2;

    unsigned long long k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= num_blocks) return;

    // Insert bits from k into all positions except low/high
    unsigned long long base = 0ULL;
    unsigned long long src  = k;

    for (int p = 0; p < n; ++p) {
        if (p == low || p == high) continue;
        unsigned long long bit = (src & 1ULL);
        src >>= 1;
        base |= (bit << p);
    }

    unsigned long long mask_low  = 1ULL << low;
    unsigned long long mask_high = 1ULL << high;

    unsigned long long i00 = base;
    unsigned long long i01 = base | mask_low;
    unsigned long long i10 = base | mask_high;
    unsigned long long i11 = base | mask_low | mask_high;

    cuDoubleComplex a00 = psi[i00];
    cuDoubleComplex a01 = psi[i01];
    cuDoubleComplex a10 = psi[i10];
    cuDoubleComplex a11 = psi[i11];

    // Row-major U[ row*4 + col ]
    cuDoubleComplex out0 =
        cuCadd(cuCadd(cuCmul(U[0], a00), cuCmul(U[1], a01)),
              cuCadd(cuCmul(U[2], a10), cuCmul(U[3], a11)));

    cuDoubleComplex out1 =
        cuCadd(cuCadd(cuCmul(U[4], a00), cuCmul(U[5], a01)),
              cuCadd(cuCmul(U[6], a10), cuCmul(U[7], a11)));

    cuDoubleComplex out2 =
        cuCadd(cuCadd(cuCmul(U[8], a00), cuCmul(U[9], a01)),
              cuCadd(cuCmul(U[10], a10), cuCmul(U[11], a11)));

    cuDoubleComplex out3 =
        cuCadd(cuCadd(cuCmul(U[12], a00), cuCmul(U[13], a01)),
              cuCadd(cuCmul(U[14], a10), cuCmul(U[15], a11)));

    psi[i00] = out0;
    psi[i01] = out1;
    psi[i10] = out2;
    psi[i11] = out3;
}
