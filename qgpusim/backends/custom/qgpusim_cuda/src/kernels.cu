#include "kernels.cuh"
#include <cstdio>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

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
// Qiskit uses LITTLE-ENDIAN matrix indexing: matrix_index = b_q0 + 2*b_q1
// I.e., q0 is the "fast" (low) bit of the 2-bit matrix index.
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

    // Determine which qubit is at a lower bit position in memory
    int low  = q0 < q1 ? q0 : q1;
    int high = q0 < q1 ? q1 : q0;
    
    // Flag: is q0 at the lower bit position in memory?
    bool q0_is_low = (q0 < q1);

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

    // State indices in memory (based on low/high bit positions in STATE)
    unsigned long long i_00 = base;                        // both bits 0
    unsigned long long i_0L = base | mask_low;             // low bit set, high bit 0
    unsigned long long i_H0 = base | mask_high;            // high bit set, low bit 0
    unsigned long long i_HL = base | mask_low | mask_high; // both bits set

    cuDoubleComplex a_00 = psi[i_00];
    cuDoubleComplex a_0L = psi[i_0L];
    cuDoubleComplex a_H0 = psi[i_H0];
    cuDoubleComplex a_HL = psi[i_HL];

    // Map to QISKIT LITTLE-ENDIAN matrix indices: index = b_q0 + 2*b_q1
    // Matrix ordering is |00>, |10>, |01>, |11> where the bits are (q1, q0)
    // 
    // If q0_is_low: memory has [low=q0, high=q1]
    //   i_0L has low=1, high=0 -> q0=1, q1=0 -> matrix_idx = 1 + 2*0 = 1
    //   i_H0 has low=0, high=1 -> q0=0, q1=1 -> matrix_idx = 0 + 2*1 = 2
    // If !q0_is_low: memory has [low=q1, high=q0]
    //   i_0L has low=1, high=0 -> q1=1, q0=0 -> matrix_idx = 0 + 2*1 = 2
    //   i_H0 has low=0, high=1 -> q1=0, q0=1 -> matrix_idx = 1 + 2*0 = 1
    
    // a[i] corresponds to matrix index i (Qiskit ordering)
    cuDoubleComplex a0, a1, a2, a3;  // matrix indices 0,1,2,3
    if (q0_is_low) {
        // memory low = q0, memory high = q1
        a0 = a_00;  // q0=0, q1=0 -> idx 0
        a1 = a_0L;  // q0=1, q1=0 -> idx 1
        a2 = a_H0;  // q0=0, q1=1 -> idx 2
        a3 = a_HL;  // q0=1, q1=1 -> idx 3
    } else {
        // memory low = q1, memory high = q0
        a0 = a_00;  // q0=0, q1=0 -> idx 0
        a1 = a_H0;  // q0=1, q1=0 -> idx 1 (high bit is q0)
        a2 = a_0L;  // q0=0, q1=1 -> idx 2 (low bit is q1)
        a3 = a_HL;  // q0=1, q1=1 -> idx 3
    }

    // Compute outputs: out[i] = sum_j U[i*4 + j] * a[j]
    cuDoubleComplex out0, out1, out2, out3;

    out0 = cuCadd(cuCadd(cuCmul(U_shared[0], a0), cuCmul(U_shared[1], a1)),
                  cuCadd(cuCmul(U_shared[2], a2), cuCmul(U_shared[3], a3)));

    out1 = cuCadd(cuCadd(cuCmul(U_shared[4], a0), cuCmul(U_shared[5], a1)),
                  cuCadd(cuCmul(U_shared[6], a2), cuCmul(U_shared[7], a3)));

    out2 = cuCadd(cuCadd(cuCmul(U_shared[8], a0), cuCmul(U_shared[9], a1)),
                  cuCadd(cuCmul(U_shared[10], a2), cuCmul(U_shared[11], a3)));

    out3 = cuCadd(cuCadd(cuCmul(U_shared[12], a0), cuCmul(U_shared[13], a1)),
                  cuCadd(cuCmul(U_shared[14], a2), cuCmul(U_shared[15], a3)));

    // Map outputs back to state indices
    if (q0_is_low) {
        psi[i_00] = out0;
        psi[i_0L] = out1;
        psi[i_H0] = out2;
        psi[i_HL] = out3;
    } else {
        psi[i_00] = out0;
        psi[i_H0] = out1;
        psi[i_0L] = out2;
        psi[i_HL] = out3;
    }
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
    // permute_bits_kernel<<<(unsigned int)blocks, threads>>>(in, out, d_map_old_to_new, n, dim);
    permute_bits_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(
    in, out, d_map_old_to_new, n, dim
);

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

// ============================================================================
// TRUE TILE KERNEL - Single kernel processes ALL gates in a tile
// Uses cooperative groups for grid-wide synchronization between gates
// ============================================================================

// Device functions for applying gates within the tile kernel
__device__ void tile_apply_1q_gate(
    cuDoubleComplex* __restrict__ psi,
    const cuDoubleComplex* __restrict__ U,
    const long long n,
    const int q,
    const unsigned long long global_tid,
    const unsigned long long total_threads
) {
    unsigned long long dim = 1ULL << n;
    unsigned long long stride = 1ULL << q;
    unsigned long long num_pairs = dim >> 1;

    // Each thread processes multiple pairs if needed
    for (unsigned long long k = global_tid; k < num_pairs; k += total_threads) {
        unsigned long long block  = k / stride;
        unsigned long long offset = k % stride;

        unsigned long long i0 = block * 2ULL * stride + offset;
        unsigned long long i1 = i0 + stride;

        cuDoubleComplex a0 = psi[i0];
        cuDoubleComplex a1 = psi[i1];

        cuDoubleComplex out0 = cuCadd(cuCmul(U[0], a0), cuCmul(U[1], a1));
        cuDoubleComplex out1 = cuCadd(cuCmul(U[2], a0), cuCmul(U[3], a1));

        psi[i0] = out0;
        psi[i1] = out1;
    }
}

__device__ void tile_apply_2q_gate(
    cuDoubleComplex* __restrict__ psi,
    const cuDoubleComplex* __restrict__ U,
    const long long n,
    const int q0,
    const int q1,
    const unsigned long long global_tid,
    const unsigned long long total_threads
) {
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

        // State indices in memory
        unsigned long long i_00 = base;
        unsigned long long i_0L = base | mask_low;
        unsigned long long i_H0 = base | mask_high;
        unsigned long long i_HL = base | mask_low | mask_high;

        cuDoubleComplex a_00 = psi[i_00];
        cuDoubleComplex a_0L = psi[i_0L];
        cuDoubleComplex a_H0 = psi[i_H0];
        cuDoubleComplex a_HL = psi[i_HL];

        // Map to QISKIT LITTLE-ENDIAN matrix indices: index = b_q0 + 2*b_q1
        cuDoubleComplex a0, a1, a2, a3;
        if (q0_is_low) {
            a0 = a_00; a1 = a_0L; a2 = a_H0; a3 = a_HL;
        } else {
            a0 = a_00; a1 = a_H0; a2 = a_0L; a3 = a_HL;
        }

        cuDoubleComplex out0 = cuCadd(cuCadd(cuCmul(U[0], a0), cuCmul(U[1], a1)),
                                      cuCadd(cuCmul(U[2], a2), cuCmul(U[3], a3)));
        cuDoubleComplex out1 = cuCadd(cuCadd(cuCmul(U[4], a0), cuCmul(U[5], a1)),
                                      cuCadd(cuCmul(U[6], a2), cuCmul(U[7], a3)));
        cuDoubleComplex out2 = cuCadd(cuCadd(cuCmul(U[8], a0), cuCmul(U[9], a1)),
                                      cuCadd(cuCmul(U[10], a2), cuCmul(U[11], a3)));
        cuDoubleComplex out3 = cuCadd(cuCadd(cuCmul(U[12], a0), cuCmul(U[13], a1)),
                                      cuCadd(cuCmul(U[14], a2), cuCmul(U[15], a3)));

        // Map outputs back to state indices
        if (q0_is_low) {
            psi[i_00] = out0; psi[i_0L] = out1; psi[i_H0] = out2; psi[i_HL] = out3;
        } else {
            psi[i_00] = out0; psi[i_H0] = out1; psi[i_0L] = out2; psi[i_HL] = out3;
        }
    }
}

// Main tile kernel - processes all gates sequentially with grid-wide sync
// Sync is required after EVERY gate due to read-write dependencies within gates
__global__ void apply_tile_kernel(
    cuDoubleComplex* __restrict__ psi,
    const TileGateOp* __restrict__ ops,
    const int num_ops,
    const long long num_qubits
) {
    // Get grid-wide thread group for synchronization
    cg::grid_group grid = cg::this_grid();
    
    unsigned long long global_tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total_threads = (unsigned long long)gridDim.x * blockDim.x;

    // Process each gate in the tile
    for (int op_idx = 0; op_idx < num_ops; ++op_idx) {
        const TileGateOp& op = ops[op_idx];
        
        if (op.kind == 1) {
            // 1-qubit gate
            tile_apply_1q_gate(psi, op.U, num_qubits, op.q0, global_tid, total_threads);
        } else if (op.kind == 2) {
            // 2-qubit gate
            tile_apply_2q_gate(psi, op.U, num_qubits, op.q0, op.q1, global_tid, total_threads);
        }
        
        // Must sync after every gate - threads may read values being written by others
        grid.sync();
    }
}

// Query maximum blocks for cooperative launch
int get_tile_kernel_max_blocks(int threads_per_block) {
    int num_blocks = 0;
    int device = 0;
    cudaGetDevice(&device);
    
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks,
        apply_tile_kernel,
        threads_per_block,
        0  // dynamic shared memory
    );
    
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
    
    return num_blocks * num_sms;
}

// Host wrapper for cooperative launch
void launch_apply_tile(
    cuDoubleComplex* psi,
    const TileGateOp* d_ops,
    int num_ops,
    long long num_qubits,
    cudaStream_t stream
) {
    
    if (num_ops == 0) return;
    
    const int threads_per_block = 256;
    
    // Get max blocks for cooperative launch
    int max_blocks = get_tile_kernel_max_blocks(threads_per_block);
    
    // Calculate needed blocks based on statevector size
    unsigned long long dim = 1ULL << num_qubits;
    unsigned long long needed_blocks = (dim + threads_per_block - 1) / threads_per_block;
    
    // Use minimum of max and needed
    int num_blocks = (int)min((unsigned long long)max_blocks, needed_blocks);
    if (num_blocks < 1) num_blocks = 1;
    
    // Setup cooperative launch parameters
    void* kernel_args[] = {
        &psi,
        &d_ops,
        &num_ops,
        &num_qubits
    };
    
    // Launch cooperatively
    cudaLaunchCooperativeKernel(
        (void*)apply_tile_kernel,
        dim3(num_blocks),
        dim3(threads_per_block),
        kernel_args,
        0,      // shared memory
        stream
    );


}
