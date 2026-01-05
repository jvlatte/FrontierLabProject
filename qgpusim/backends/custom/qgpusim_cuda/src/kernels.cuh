#ifndef QGPUSIM_KERNELS_CUH
#define QGPUSIM_KERNELS_CUH

#include <cuComplex.h>
#include <cuda_runtime.h>

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

#endif // QGPUSIM_KERNELS_CUH
