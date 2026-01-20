#include "kernels.cuh"
#include <cstdio>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// ============================================================================
// TEMPLATED 1-Qubit Gate Kernel (optimized with shared memory)
// ============================================================================
template<typename Complex>
__global__ void apply_1q_gate_kernel_t(
    Complex* __restrict__ psi,
    const Complex* __restrict__ U,
    const long long n,
    const int q
) {
    using Traits = ComplexTraits<Complex>;
    
    // Load gate matrix into shared memory (only 4 elements)
    __shared__ Complex U_shared[4];
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

    Complex a0 = psi[i0];
    Complex a1 = psi[i1];

    // Use shared memory for gate matrix
    Complex out0, out1;
    out0 = Traits::add(Traits::mul(U_shared[0], a0), Traits::mul(U_shared[1], a1));
    out1 = Traits::add(Traits::mul(U_shared[2], a0), Traits::mul(U_shared[3], a1));

    psi[i0] = out0;
    psi[i1] = out1;
}

// ============================================================================
// TEMPLATED 2-Qubit Gate Kernel (optimized with shared memory)
// Qiskit uses LITTLE-ENDIAN matrix indexing: matrix_index = b_q0 + 2*b_q1
// ============================================================================
template<typename Complex>
__global__ void apply_2q_gate_kernel_t(
    Complex* __restrict__ psi,
    const Complex* __restrict__ U,
    const long long n,
    const int q0,
    const int q1
) {
    using Traits = ComplexTraits<Complex>;
    
    // Load gate matrix into shared memory (16 elements)
    __shared__ Complex U_shared[16];
    if (threadIdx.x < 16) {
        U_shared[threadIdx.x] = U[threadIdx.x];
    }
    __syncthreads();

    // Determine which qubit is at a lower bit position in memory
    int low  = q0 < q1 ? q0 : q1;
    int high = q0 < q1 ? q1 : q0;
    
    // Flag: is q0 at the lower bit position in memory?
    bool q0_is_low = (q0 < q1);

    unsigned long long dim = 1ULL << n;
    unsigned long long num_blocks = dim >> 2;

    unsigned long long k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= num_blocks) return;

    // Build base index using optimized bitwise operations
    unsigned long long mask_low  = 1ULL << low;
    unsigned long long mask_high = 1ULL << high;
    
    unsigned long long below_low_mask = mask_low - 1;
    unsigned long long below_low = k & below_low_mask;
    unsigned long long between_mask = (1ULL << (high - low - 1)) - 1;
    unsigned long long between = ((k >> low) & between_mask) << (low + 1);
    unsigned long long above = (k >> (high - 1)) << (high + 1);
    unsigned long long base = below_low | between | above;

    // State indices in memory
    unsigned long long i_00 = base;
    unsigned long long i_0L = base | mask_low;
    unsigned long long i_H0 = base | mask_high;
    unsigned long long i_HL = base | mask_low | mask_high;

    Complex a_00 = psi[i_00];
    Complex a_0L = psi[i_0L];
    Complex a_H0 = psi[i_H0];
    Complex a_HL = psi[i_HL];

    // Map to QISKIT LITTLE-ENDIAN matrix indices: index = b_q0 + 2*b_q1
    Complex a0, a1, a2, a3;
    if (q0_is_low) {
        a0 = a_00; a1 = a_0L; a2 = a_H0; a3 = a_HL;
    } else {
        a0 = a_00; a1 = a_H0; a2 = a_0L; a3 = a_HL;
    }

    // Compute outputs
    Complex out0, out1, out2, out3;

    out0 = Traits::add(Traits::add(Traits::mul(U_shared[0], a0), Traits::mul(U_shared[1], a1)),
                       Traits::add(Traits::mul(U_shared[2], a2), Traits::mul(U_shared[3], a3)));

    out1 = Traits::add(Traits::add(Traits::mul(U_shared[4], a0), Traits::mul(U_shared[5], a1)),
                       Traits::add(Traits::mul(U_shared[6], a2), Traits::mul(U_shared[7], a3)));

    out2 = Traits::add(Traits::add(Traits::mul(U_shared[8], a0), Traits::mul(U_shared[9], a1)),
                       Traits::add(Traits::mul(U_shared[10], a2), Traits::mul(U_shared[11], a3)));

    out3 = Traits::add(Traits::add(Traits::mul(U_shared[12], a0), Traits::mul(U_shared[13], a1)),
                       Traits::add(Traits::mul(U_shared[14], a2), Traits::mul(U_shared[15], a3)));

    // Map outputs back to state indices
    if (q0_is_low) {
        psi[i_00] = out0; psi[i_0L] = out1; psi[i_H0] = out2; psi[i_HL] = out3;
    } else {
        psi[i_00] = out0; psi[i_H0] = out1; psi[i_0L] = out2; psi[i_HL] = out3;
    }
}

// ============================================================================
// TEMPLATED Diagonal Gate Kernel
// ============================================================================
template<typename Complex>
__global__ void apply_diagonal_1q_gate_kernel_t(
    Complex* __restrict__ psi,
    const Complex phase0,
    const Complex phase1,
    const long long n,
    const int q
) {
    using Traits = ComplexTraits<Complex>;
    
    unsigned long long dim = 1ULL << n;
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;

    unsigned long long mask = 1ULL << q;
    Complex phase = (idx & mask) ? phase1 : phase0;
    
    psi[idx] = Traits::mul(psi[idx], phase);
}

// ============================================================================
// TEMPLATED Permute Bits Kernel
// ============================================================================
template<typename Complex>
__global__ void permute_bits_kernel_t(
    const Complex* __restrict__ in,
    Complex* __restrict__ out,
    const int* __restrict__ map_old_to_new,
    int n,
    unsigned long long dim
) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;

    unsigned long long j = 0ULL;
    #pragma unroll
    for (int p_old = 0; p_old < 64; ++p_old) {
        if (p_old >= n) break;
        unsigned long long bit = (i >> p_old) & 1ULL;
        j |= (bit << (unsigned long long)map_old_to_new[p_old]);
    }

    out[j] = in[i];
}

// ============================================================================
// TEMPLATED Device functions for tile kernel
// ============================================================================
template<typename Complex>
__device__ void tile_apply_1q_gate_t(
    Complex* __restrict__ psi,
    const Complex* __restrict__ U,
    const long long n,
    const int q,
    const unsigned long long global_tid,
    const unsigned long long total_threads
) {
    using Traits = ComplexTraits<Complex>;
    
    unsigned long long dim = 1ULL << n;
    unsigned long long stride = 1ULL << q;
    unsigned long long num_pairs = dim >> 1;

    for (unsigned long long k = global_tid; k < num_pairs; k += total_threads) {
        unsigned long long block  = k / stride;
        unsigned long long offset = k % stride;

        unsigned long long i0 = block * 2ULL * stride + offset;
        unsigned long long i1 = i0 + stride;

        Complex a0 = psi[i0];
        Complex a1 = psi[i1];

        Complex out0 = Traits::add(Traits::mul(U[0], a0), Traits::mul(U[1], a1));
        Complex out1 = Traits::add(Traits::mul(U[2], a0), Traits::mul(U[3], a1));

        psi[i0] = out0;
        psi[i1] = out1;
    }
}

template<typename Complex>
__device__ void tile_apply_2q_gate_t(
    Complex* __restrict__ psi,
    const Complex* __restrict__ U,
    const long long n,
    const int q0,
    const int q1,
    const unsigned long long global_tid,
    const unsigned long long total_threads
) {
    using Traits = ComplexTraits<Complex>;
    
    int low  = q0 < q1 ? q0 : q1;
    int high = q0 < q1 ? q1 : q0;
    bool q0_is_low = (q0 < q1);

    unsigned long long dim = 1ULL << n;
    unsigned long long num_blocks_total = dim >> 2;

    unsigned long long mask_low  = 1ULL << low;
    unsigned long long mask_high = 1ULL << high;

    for (unsigned long long k = global_tid; k < num_blocks_total; k += total_threads) {
        unsigned long long below_low_mask = mask_low - 1;
        unsigned long long below_low = k & below_low_mask;
        unsigned long long between_mask = (1ULL << (high - low - 1)) - 1;
        unsigned long long between = ((k >> low) & between_mask) << (low + 1);
        unsigned long long above = (k >> (high - 1)) << (high + 1);
        unsigned long long base = below_low | between | above;

        unsigned long long i_00 = base;
        unsigned long long i_0L = base | mask_low;
        unsigned long long i_H0 = base | mask_high;
        unsigned long long i_HL = base | mask_low | mask_high;

        Complex a_00 = psi[i_00];
        Complex a_0L = psi[i_0L];
        Complex a_H0 = psi[i_H0];
        Complex a_HL = psi[i_HL];

        Complex a0, a1, a2, a3;
        if (q0_is_low) {
            a0 = a_00; a1 = a_0L; a2 = a_H0; a3 = a_HL;
        } else {
            a0 = a_00; a1 = a_H0; a2 = a_0L; a3 = a_HL;
        }

        Complex out0 = Traits::add(Traits::add(Traits::mul(U[0], a0), Traits::mul(U[1], a1)),
                                   Traits::add(Traits::mul(U[2], a2), Traits::mul(U[3], a3)));
        Complex out1 = Traits::add(Traits::add(Traits::mul(U[4], a0), Traits::mul(U[5], a1)),
                                   Traits::add(Traits::mul(U[6], a2), Traits::mul(U[7], a3)));
        Complex out2 = Traits::add(Traits::add(Traits::mul(U[8], a0), Traits::mul(U[9], a1)),
                                   Traits::add(Traits::mul(U[10], a2), Traits::mul(U[11], a3)));
        Complex out3 = Traits::add(Traits::add(Traits::mul(U[12], a0), Traits::mul(U[13], a1)),
                                   Traits::add(Traits::mul(U[14], a2), Traits::mul(U[15], a3)));

        if (q0_is_low) {
            psi[i_00] = out0; psi[i_0L] = out1; psi[i_H0] = out2; psi[i_HL] = out3;
        } else {
            psi[i_00] = out0; psi[i_H0] = out1; psi[i_0L] = out2; psi[i_HL] = out3;
        }
    }
}

// ============================================================================
// TEMPLATED Tile Kernel
// ============================================================================
template<typename Complex>
__global__ void apply_tile_kernel_t(
    Complex* __restrict__ psi,
    const TileGateOpT<Complex>* __restrict__ ops,
    const int num_ops,
    const long long num_qubits
) {
    cg::grid_group grid = cg::this_grid();
    
    unsigned long long global_tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total_threads = (unsigned long long)gridDim.x * blockDim.x;

    for (int op_idx = 0; op_idx < num_ops; ++op_idx) {
        const TileGateOpT<Complex>& op = ops[op_idx];
        
        if (op.kind == 1) {
            tile_apply_1q_gate_t<Complex>(psi, op.U, num_qubits, op.q0, global_tid, total_threads);
        } else if (op.kind == 2) {
            tile_apply_2q_gate_t<Complex>(psi, op.U, num_qubits, op.q0, op.q1, global_tid, total_threads);
        }
        
        grid.sync();
    }
}

// ============================================================================
// TEMPLATED Host Launcher Functions
// ============================================================================

template<typename Complex>
void launch_apply_1q_gate_t(
    Complex* psi,
    const Complex* U,
    long long num_qubits,
    int target_qubit,
    cudaStream_t stream
) {
    unsigned long long dim = 1ULL << num_qubits;
    unsigned long long num_pairs = dim >> 1;

    int threads_per_block = 256;
    int blocks = (num_pairs + threads_per_block - 1) / threads_per_block;

    apply_1q_gate_kernel_t<Complex><<<blocks, threads_per_block, 0, stream>>>(
        psi, U, num_qubits, target_qubit
    );
}

template<typename Complex>
void launch_apply_2q_gate_t(
    Complex* psi,
    const Complex* U,
    long long num_qubits,
    int q0,
    int q1,
    cudaStream_t stream
) {
    unsigned long long dim = 1ULL << num_qubits;
    unsigned long long num_blocks_total = dim >> 2;

    int threads_per_block = 256;
    int blocks = (num_blocks_total + threads_per_block - 1) / threads_per_block;

    apply_2q_gate_kernel_t<Complex><<<blocks, threads_per_block, 0, stream>>>(
        psi, U, num_qubits, q0, q1
    );
}

template<typename Complex>
void launch_apply_diagonal_1q_gate_t(
    Complex* psi,
    Complex phase0,
    Complex phase1,
    long long num_qubits,
    int target_qubit,
    cudaStream_t stream
) {
    unsigned long long dim = 1ULL << num_qubits;

    int threads_per_block = 256;
    int blocks = (dim + threads_per_block - 1) / threads_per_block;

    apply_diagonal_1q_gate_kernel_t<Complex><<<blocks, threads_per_block, 0, stream>>>(
        psi, phase0, phase1, num_qubits, target_qubit
    );
}

template<typename Complex>
void launch_permute_bits_t(
    const Complex* in,
    Complex* out,
    const int* d_map_old_to_new,
    int n,
    unsigned long long dim,
    cudaStream_t stream
) {
    const int threads = 256;
    const unsigned long long blocks = (dim + threads - 1ULL) / (unsigned long long)threads;
    permute_bits_kernel_t<Complex><<<(unsigned int)blocks, threads, 0, stream>>>(
        in, out, d_map_old_to_new, n, dim
    );
}

template<typename Complex>
int get_tile_kernel_max_blocks_t(int threads_per_block) {
    int num_blocks = 0;
    int device = 0;
    cudaGetDevice(&device);
    
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks,
        apply_tile_kernel_t<Complex>,
        threads_per_block,
        0
    );
    
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
    
    return num_blocks * num_sms;
}

template<typename Complex>
void launch_apply_tile_t(
    Complex* psi,
    const TileGateOpT<Complex>* d_ops,
    int num_ops,
    long long num_qubits,
    cudaStream_t stream
) {
    if (num_ops == 0) return;
    
    const int threads_per_block = 256;
    int max_blocks = get_tile_kernel_max_blocks_t<Complex>(threads_per_block);
    
    unsigned long long dim = 1ULL << num_qubits;
    unsigned long long needed_blocks = (dim + threads_per_block - 1) / threads_per_block;
    
    int num_blocks = (int)min((unsigned long long)max_blocks, needed_blocks);
    if (num_blocks < 1) num_blocks = 1;
    
    void* kernel_args[] = {
        &psi,
        &d_ops,
        &num_ops,
        &num_qubits
    };
    
    cudaLaunchCooperativeKernel(
        (void*)apply_tile_kernel_t<Complex>,
        dim3(num_blocks),
        dim3(threads_per_block),
        kernel_args,
        0,
        stream
    );
}

// ============================================================================
// EXPLICIT TEMPLATE INSTANTIATIONS
// ============================================================================

// Float64 (cuDoubleComplex) instantiations
template __global__ void apply_1q_gate_kernel_t<cuDoubleComplex>(cuDoubleComplex*, const cuDoubleComplex*, long long, int);
template __global__ void apply_2q_gate_kernel_t<cuDoubleComplex>(cuDoubleComplex*, const cuDoubleComplex*, long long, int, int);
template __global__ void apply_diagonal_1q_gate_kernel_t<cuDoubleComplex>(cuDoubleComplex*, cuDoubleComplex, cuDoubleComplex, long long, int);
template __global__ void permute_bits_kernel_t<cuDoubleComplex>(const cuDoubleComplex*, cuDoubleComplex*, const int*, int, unsigned long long);
template __global__ void apply_tile_kernel_t<cuDoubleComplex>(cuDoubleComplex*, const TileGateOpT<cuDoubleComplex>*, int, long long);

template void launch_apply_1q_gate_t<cuDoubleComplex>(cuDoubleComplex*, const cuDoubleComplex*, long long, int, cudaStream_t);
template void launch_apply_2q_gate_t<cuDoubleComplex>(cuDoubleComplex*, const cuDoubleComplex*, long long, int, int, cudaStream_t);
template void launch_apply_diagonal_1q_gate_t<cuDoubleComplex>(cuDoubleComplex*, cuDoubleComplex, cuDoubleComplex, long long, int, cudaStream_t);
template void launch_permute_bits_t<cuDoubleComplex>(const cuDoubleComplex*, cuDoubleComplex*, const int*, int, unsigned long long, cudaStream_t);
template void launch_apply_tile_t<cuDoubleComplex>(cuDoubleComplex*, const TileGateOpT<cuDoubleComplex>*, int, long long, cudaStream_t);
template int get_tile_kernel_max_blocks_t<cuDoubleComplex>(int);

// Float32 (cuFloatComplex) instantiations
template __global__ void apply_1q_gate_kernel_t<cuFloatComplex>(cuFloatComplex*, const cuFloatComplex*, long long, int);
template __global__ void apply_2q_gate_kernel_t<cuFloatComplex>(cuFloatComplex*, const cuFloatComplex*, long long, int, int);
template __global__ void apply_diagonal_1q_gate_kernel_t<cuFloatComplex>(cuFloatComplex*, cuFloatComplex, cuFloatComplex, long long, int);
template __global__ void permute_bits_kernel_t<cuFloatComplex>(const cuFloatComplex*, cuFloatComplex*, const int*, int, unsigned long long);
template __global__ void apply_tile_kernel_t<cuFloatComplex>(cuFloatComplex*, const TileGateOpT<cuFloatComplex>*, int, long long);

template void launch_apply_1q_gate_t<cuFloatComplex>(cuFloatComplex*, const cuFloatComplex*, long long, int, cudaStream_t);
template void launch_apply_2q_gate_t<cuFloatComplex>(cuFloatComplex*, const cuFloatComplex*, long long, int, int, cudaStream_t);
template void launch_apply_diagonal_1q_gate_t<cuFloatComplex>(cuFloatComplex*, cuFloatComplex, cuFloatComplex, long long, int, cudaStream_t);
template void launch_permute_bits_t<cuFloatComplex>(const cuFloatComplex*, cuFloatComplex*, const int*, int, unsigned long long, cudaStream_t);
template void launch_apply_tile_t<cuFloatComplex>(cuFloatComplex*, const TileGateOpT<cuFloatComplex>*, int, long long, cudaStream_t);
template int get_tile_kernel_max_blocks_t<cuFloatComplex>(int);

// ============================================================================
// LEGACY HOST LAUNCHER WRAPPERS (for backward compatibility)
// These call the templated versions with cuDoubleComplex
// ============================================================================

void launch_permute_bits(
    const cuDoubleComplex* in,
    cuDoubleComplex* out,
    const int* d_map_old_to_new,
    int n,
    unsigned long long dim,
    cudaStream_t stream
) {
    launch_permute_bits_t<cuDoubleComplex>(in, out, d_map_old_to_new, n, dim, stream);
}

void launch_apply_1q_gate(
    cuDoubleComplex* psi,
    const cuDoubleComplex* U,
    long long num_qubits,
    int target_qubit,
    cudaStream_t stream
) {
    launch_apply_1q_gate_t<cuDoubleComplex>(psi, U, num_qubits, target_qubit, stream);
}

void launch_apply_2q_gate(
    cuDoubleComplex* psi,
    const cuDoubleComplex* U,
    long long num_qubits,
    int q0,
    int q1,
    cudaStream_t stream
) {
    launch_apply_2q_gate_t<cuDoubleComplex>(psi, U, num_qubits, q0, q1, stream);
}

void launch_apply_diagonal_1q_gate(
    cuDoubleComplex* psi,
    cuDoubleComplex phase0,
    cuDoubleComplex phase1,
    long long num_qubits,
    int target_qubit,
    cudaStream_t stream
) {
    launch_apply_diagonal_1q_gate_t<cuDoubleComplex>(psi, phase0, phase1, num_qubits, target_qubit, stream);
}

void launch_apply_tile(
    cuDoubleComplex* psi,
    const TileGateOp* d_ops,
    int num_ops,
    long long num_qubits,
    cudaStream_t stream
) {
    launch_apply_tile_t<cuDoubleComplex>(psi, d_ops, num_ops, num_qubits, stream);
}

int get_tile_kernel_max_blocks(int threads_per_block) {
    return get_tile_kernel_max_blocks_t<cuDoubleComplex>(threads_per_block);
}

// ============================================================================
// FLOAT32 WRAPPERS
// ============================================================================

void launch_permute_bits_f32(
    const cuFloatComplex* in,
    cuFloatComplex* out,
    const int* d_map_old_to_new,
    int n,
    unsigned long long dim,
    cudaStream_t stream
) {
    launch_permute_bits_t<cuFloatComplex>(in, out, d_map_old_to_new, n, dim, stream);
}

void launch_apply_1q_gate_f32(
    cuFloatComplex* psi,
    const cuFloatComplex* U,
    long long num_qubits,
    int target_qubit,
    cudaStream_t stream
) {
    launch_apply_1q_gate_t<cuFloatComplex>(psi, U, num_qubits, target_qubit, stream);
}

void launch_apply_2q_gate_f32(
    cuFloatComplex* psi,
    const cuFloatComplex* U,
    long long num_qubits,
    int q0,
    int q1,
    cudaStream_t stream
) {
    launch_apply_2q_gate_t<cuFloatComplex>(psi, U, num_qubits, q0, q1, stream);
}

void launch_apply_tile_f32(
    cuFloatComplex* psi,
    const TileGateOpF32* d_ops,
    int num_ops,
    long long num_qubits,
    cudaStream_t stream
) {
    launch_apply_tile_t<cuFloatComplex>(psi, d_ops, num_ops, num_qubits, stream);
}

int get_tile_kernel_max_blocks_f32(int threads_per_block) {
    return get_tile_kernel_max_blocks_t<cuFloatComplex>(threads_per_block);
}
