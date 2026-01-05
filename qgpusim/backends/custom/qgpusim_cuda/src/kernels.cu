#include "kernels.cuh"
#include <cstdio>

// ============================================================================
// 1-Qubit Gate Kernel (optimized with shared memory)
// ============================================================================
__global__ void apply_1q_gate_kernel(
    cuDoubleComplex* __restrict__ psi,
    const cuDoubleComplex* __restrict__ U,
    const long long n,
    const int q
) {
    // Load gate matrix into shared memory (only 4 elements)
    __shared__ cuDoubleComplex U_shared[4];
    if (threadIdx.x < 4) {
        U_shared[threadIdx.x] = U[threadIdx.x];
    }
    __syncthreads();

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

    // Use shared memory for gate matrix
    cuDoubleComplex out0, out1;
    out0 = cuCadd(cuCmul(U_shared[0], a0), cuCmul(U_shared[1], a1));
    out1 = cuCadd(cuCmul(U_shared[2], a0), cuCmul(U_shared[3], a1));

    psi[i0] = out0;
    psi[i1] = out1;
}

// ============================================================================
// 2-Qubit Gate Kernel (optimized with shared memory)
// ============================================================================
__global__ void apply_2q_gate_kernel(
    cuDoubleComplex* __restrict__ psi,
    const cuDoubleComplex* __restrict__ U,
    const long long n,
    const int q0,
    const int q1
) {
    // Load gate matrix into shared memory (16 elements)
    __shared__ cuDoubleComplex U_shared[16];
    if (threadIdx.x < 16) {
        U_shared[threadIdx.x] = U[threadIdx.x];
    }
    __syncthreads();

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

    // Compute outputs using shared memory gate matrix
    cuDoubleComplex out0, out1, out2, out3;

    out0 = cuCadd(cuCadd(cuCmul(U_shared[0], a00), cuCmul(U_shared[1], a01)),
                  cuCadd(cuCmul(U_shared[2], a10), cuCmul(U_shared[3], a11)));

    out1 = cuCadd(cuCadd(cuCmul(U_shared[4], a00), cuCmul(U_shared[5], a01)),
                  cuCadd(cuCmul(U_shared[6], a10), cuCmul(U_shared[7], a11)));

    out2 = cuCadd(cuCadd(cuCmul(U_shared[8], a00), cuCmul(U_shared[9], a01)),
                  cuCadd(cuCmul(U_shared[10], a10), cuCmul(U_shared[11], a11)));

    out3 = cuCadd(cuCadd(cuCmul(U_shared[12], a00), cuCmul(U_shared[13], a01)),
                  cuCadd(cuCmul(U_shared[14], a10), cuCmul(U_shared[15], a11)));

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
