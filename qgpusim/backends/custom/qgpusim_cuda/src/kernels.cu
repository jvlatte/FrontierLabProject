#include "kernels.cuh"
#include <cstdio>

// ============================================================================
// 1-Qubit Gate Kernel
// ============================================================================
__global__ void apply_1q_gate_kernel(
    cuDoubleComplex* psi,
    const cuDoubleComplex* U,
    const long long n,
    const int q
) {
    unsigned long long dim = 1ULL << n;
    unsigned long long stride = 1ULL << q;
    unsigned long long num_pairs = dim >> 1;

    unsigned long long k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= num_pairs) return;

    // Map k -> (i0, i1) such that bit q of i0 is 0 and i1 = i0 + stride
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

    cuDoubleComplex out0, out1;
    out0 = cuCadd(cuCmul(u00, a0), cuCmul(u01, a1));
    out1 = cuCadd(cuCmul(u10, a0), cuCmul(u11, a1));

    psi[i0] = out0;
    psi[i1] = out1;
}

// ============================================================================
// 2-Qubit Gate Kernel
// ============================================================================
__global__ void apply_2q_gate_kernel(
    cuDoubleComplex* psi,
    const cuDoubleComplex* U,
    const long long n,
    const int q0,
    const int q1
) {
    // Ensure low < high
    int low  = q0 < q1 ? q0 : q1;
    int high = q0 < q1 ? q1 : q0;

    unsigned long long dim = 1ULL << n;
    unsigned long long num_blocks = dim >> 2;

    unsigned long long k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= num_blocks) return;

    // Build base index by inserting zero bits at positions low and high
    unsigned long long base = 0ULL;
    unsigned long long src  = k;

    for (int p = 0; p < n; ++p) {
        if (p == low || p == high) {
            continue;
        }
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

    // U is 4x4 row-major
    cuDoubleComplex U00 = U[0];   cuDoubleComplex U01 = U[1];
    cuDoubleComplex U02 = U[2];   cuDoubleComplex U03 = U[3];
    cuDoubleComplex U10 = U[4];   cuDoubleComplex U11 = U[5];
    cuDoubleComplex U12 = U[6];   cuDoubleComplex U13 = U[7];
    cuDoubleComplex U20 = U[8];   cuDoubleComplex U21 = U[9];
    cuDoubleComplex U22 = U[10];  cuDoubleComplex U23 = U[11];
    cuDoubleComplex U30 = U[12];  cuDoubleComplex U31 = U[13];
    cuDoubleComplex U32 = U[14];  cuDoubleComplex U33 = U[15];

    cuDoubleComplex out0, out1, out2, out3;

    out0 = cuCadd(cuCadd(cuCmul(U00, a00), cuCmul(U01, a01)),
                  cuCadd(cuCmul(U02, a10), cuCmul(U03, a11)));

    out1 = cuCadd(cuCadd(cuCmul(U10, a00), cuCmul(U11, a01)),
                  cuCadd(cuCmul(U12, a10), cuCmul(U13, a11)));

    out2 = cuCadd(cuCadd(cuCmul(U20, a00), cuCmul(U21, a01)),
                  cuCadd(cuCmul(U22, a10), cuCmul(U23, a11)));

    out3 = cuCadd(cuCadd(cuCmul(U30, a00), cuCmul(U31, a01)),
                  cuCadd(cuCmul(U32, a10), cuCmul(U33, a11)));

    psi[i00] = out0;
    psi[i01] = out1;
    psi[i10] = out2;
    psi[i11] = out3;
}

// ============================================================================
// Host Wrapper Functions
// ============================================================================

void launch_apply_1q_gate(
    cuDoubleComplex* psi,
    const cuDoubleComplex* U,
    long long num_qubits,
    int target_qubit,
    cudaStream_t stream
) {
    unsigned long long dim = 1ULL << num_qubits;
    unsigned long long num_pairs = dim >> 1;

    int threads_per_block = 256;
    int blocks = (num_pairs + threads_per_block - 1) / threads_per_block;

    apply_1q_gate_kernel<<<blocks, threads_per_block, 0, stream>>>(
        psi, U, num_qubits, target_qubit
    );
}

void launch_apply_2q_gate(
    cuDoubleComplex* psi,
    const cuDoubleComplex* U,
    long long num_qubits,
    int q0,
    int q1,
    cudaStream_t stream
) {
    unsigned long long dim = 1ULL << num_qubits;
    unsigned long long num_blocks_total = dim >> 2;

    int threads_per_block = 256;
    int blocks = (num_blocks_total + threads_per_block - 1) / threads_per_block;

    apply_2q_gate_kernel<<<blocks, threads_per_block, 0, stream>>>(
        psi, U, num_qubits, q0, q1
    );
}
