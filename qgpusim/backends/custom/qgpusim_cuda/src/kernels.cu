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

    // Build base index using optimized bitwise operations (no loop)
    // Split k into three regions: below low, between low and high, above high
    unsigned long long mask_low  = 1ULL << low;
    unsigned long long mask_high = 1ULL << high;
    
    unsigned long long below_low_mask = mask_low - 1;
    unsigned long long below_low = k & below_low_mask;
    unsigned long long between_mask = (1ULL << (high - low - 1)) - 1;
    unsigned long long between = ((k >> low) & between_mask) << (low + 1);
    unsigned long long above = (k >> (high - 1)) << (high + 1);
    unsigned long long base = below_low | between | above;

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
// Diagonal Gate Kernel (optimized for RZ, P, T, S, Z gates)
// Only multiplies by phase factors - no matrix multiplication needed
// ============================================================================
__global__ void apply_diagonal_1q_gate_kernel(
    cuDoubleComplex* __restrict__ psi,
    const cuDoubleComplex phase0,  // Phase for |0> component
    const cuDoubleComplex phase1,  // Phase for |1> component
    const long long n,
    const int q
) {
    unsigned long long dim = 1ULL << n;
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;

    // Check if bit q is set
    unsigned long long mask = 1ULL << q;
    cuDoubleComplex phase = (idx & mask) ? phase1 : phase0;
    
    psi[idx] = cuCmul(psi[idx], phase);
}

// ============================================================================
// Host Wrapper Functions
// ============================================================================

__global__ void permute_bits_kernel(
    const cuDoubleComplex* __restrict__ in,
    cuDoubleComplex* __restrict__ out,
    const int* __restrict__ map_old_to_new,
    int n,
    unsigned long long dim
) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;

    unsigned long long j = 0ULL;
    #pragma unroll
    for (int p_old = 0; p_old < 64; ++p_old) { // safe upper bound
        if (p_old >= n) break;
        unsigned long long bit = (i >> p_old) & 1ULL;
        j |= (bit << (unsigned long long)map_old_to_new[p_old]);
    }

    out[j] = in[i];
}

void launch_permute_bits(
    const cuDoubleComplex* in,
    cuDoubleComplex* out,
    const int* d_map_old_to_new,
    int n,
    unsigned long long dim,
    cudaStream_t stream
) {
    const int threads = 256;
    const unsigned long long blocks = (dim + threads - 1ULL) / (unsigned long long)threads;
    permute_bits_kernel<<<(unsigned int)blocks, threads>>>(in, out, d_map_old_to_new, n, dim);
}



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

void launch_apply_diagonal_1q_gate(
    cuDoubleComplex* psi,
    cuDoubleComplex phase0,
    cuDoubleComplex phase1,
    long long num_qubits,
    int target_qubit,
    cudaStream_t stream
) {
    unsigned long long dim = 1ULL << num_qubits;

    int threads_per_block = 256;
    int blocks = (dim + threads_per_block - 1) / threads_per_block;

    apply_diagonal_1q_gate_kernel<<<blocks, threads_per_block, 0, stream>>>(
        psi, phase0, phase1, num_qubits, target_qubit
    );
}
