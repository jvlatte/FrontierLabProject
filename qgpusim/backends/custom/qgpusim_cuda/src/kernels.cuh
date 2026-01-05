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

#endif // QGPUSIM_KERNELS_CUH
