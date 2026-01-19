#ifndef QGPUSIM_KERNELS_CUH
#define QGPUSIM_KERNELS_CUH

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

// Structure to represent a single gate operation in a tile
struct TileGateOp {
    int kind;           // 1 = 1-qubit gate, 2 = 2-qubit gate
    int q0;             // First qubit index
    int q1;             // Second qubit index (only for 2Q gates)
    int needs_sync;     // 1 = sync after this gate, 0 = no sync needed
    cuDoubleComplex* U; // Pointer to gate matrix on device (4 or 16 elements)
};

// Apply a 1-qubit gate to a statevector (optimized with shared memory)
// psi: statevector of length 2^n
// U: 2x2 unitary matrix (flattened row-major, length 4)
// n: number of qubits
// q: target qubit index
__global__ void apply_1q_gate_kernel(
    cuDoubleComplex* __restrict__ psi,
    const cuDoubleComplex* __restrict__ U,
    const long long n,
    const int q
);

// Apply a 2-qubit gate to a statevector (optimized with shared memory)
// psi: statevector of length 2^n
// U: 4x4 unitary matrix (flattened row-major, length 16)
// n: number of qubits
// q0: first qubit index
// q1: second qubit index
__global__ void apply_2q_gate_kernel(
    cuDoubleComplex* __restrict__ psi,
    const cuDoubleComplex* __restrict__ U,
    const long long n,
    const int q0,
    const int q1
);

// Apply a diagonal 1-qubit gate (optimized for RZ, P, T, S, Z gates)
// Only multiplies by phase factors - much faster than general 1Q gate
// psi: statevector of length 2^n
// phase0: phase factor for |0> component
// phase1: phase factor for |1> component
// n: number of qubits
// q: target qubit index
__global__ void apply_diagonal_1q_gate_kernel(
    cuDoubleComplex* __restrict__ psi,
    const cuDoubleComplex phase0,
    const cuDoubleComplex phase1,
    const long long n,
    const int q
);

__global__ void permute_bits_kernel(
    const cuDoubleComplex* __restrict__ in,
    cuDoubleComplex* __restrict__ out,
    const int* __restrict__ map_old_to_new,
    int n,
    unsigned long long dim
);

void launch_permute_bits(
    const cuDoubleComplex* in,
    cuDoubleComplex* out,
    const int* d_map_old_to_new,
    int n,
    unsigned long long dim,
    cudaStream_t stream = 0
);


// Host wrapper functions
void launch_apply_1q_gate(
    cuDoubleComplex* psi,
    const cuDoubleComplex* U,
    long long num_qubits,
    int target_qubit,
    cudaStream_t stream = 0
);

void launch_apply_2q_gate(
    cuDoubleComplex* psi,
    const cuDoubleComplex* U,
    long long num_qubits,
    int q0,
    int q1,
    cudaStream_t stream = 0
);

void launch_apply_diagonal_1q_gate(
    cuDoubleComplex* psi,
    cuDoubleComplex phase0,
    cuDoubleComplex phase1,
    long long num_qubits,
    int target_qubit,
    cudaStream_t stream = 0
);

// ============================================================================
// TRUE TILE KERNEL - Executes all gates in a tile in a SINGLE kernel launch
// Uses cooperative groups for grid-wide synchronization between gates
// ============================================================================

// The actual tile kernel (uses cooperative groups)
__global__ void apply_tile_kernel(
    cuDoubleComplex* __restrict__ psi,
    const TileGateOp* __restrict__ ops,
    const int num_ops,
    const long long num_qubits
);

// Host wrapper that launches the tile kernel cooperatively
void launch_apply_tile(
    cuDoubleComplex* psi,
    const TileGateOp* d_ops,
    int num_ops,
    long long num_qubits,
    cudaStream_t stream = 0
);

// Query maximum number of blocks for cooperative launch
int get_tile_kernel_max_blocks(int threads_per_block);

#endif // QGPUSIM_KERNELS_CUH
