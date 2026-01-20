#ifndef QGPUSIM_KERNELS_CUH
#define QGPUSIM_KERNELS_CUH

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

// ============================================================================
// Type traits for complex types - enables templated kernels
// ============================================================================

template<typename T> struct ComplexTraits;

template<>
struct ComplexTraits<cuDoubleComplex> {
    using Real = double;
    static __device__ __host__ cuDoubleComplex make(double r, double i) {
        return make_cuDoubleComplex(r, i);
    }
    static __device__ cuDoubleComplex add(cuDoubleComplex a, cuDoubleComplex b) {
        return cuCadd(a, b);
    }
    static __device__ cuDoubleComplex mul(cuDoubleComplex a, cuDoubleComplex b) {
        return cuCmul(a, b);
    }
    static constexpr size_t size = sizeof(cuDoubleComplex);
};

template<>
struct ComplexTraits<cuFloatComplex> {
    using Real = float;
    static __device__ __host__ cuFloatComplex make(float r, float i) {
        return make_cuFloatComplex(r, i);
    }
    static __device__ cuFloatComplex add(cuFloatComplex a, cuFloatComplex b) {
        return cuCaddf(a, b);
    }
    static __device__ cuFloatComplex mul(cuFloatComplex a, cuFloatComplex b) {
        return cuCmulf(a, b);
    }
    static constexpr size_t size = sizeof(cuFloatComplex);
};

// ============================================================================
// Structure to represent a single gate operation in a tile (for cooperative kernel)
// ============================================================================
template<typename Complex>
struct TileGateOpT {
    int kind;           // 1 = 1-qubit gate, 2 = 2-qubit gate
    int q0;             // First qubit index
    int q1;             // Second qubit index (only for 2Q gates)
    int needs_sync;     // 1 = sync after this gate, 0 = no sync needed
    Complex* U;         // Pointer to gate matrix on device (4 or 16 elements)
};

// Legacy typedef for backward compatibility
using TileGateOp = TileGateOpT<cuDoubleComplex>;
using TileGateOpF32 = TileGateOpT<cuFloatComplex>;

// ============================================================================
// TEMPLATED KERNELS - Work for both cuDoubleComplex and cuFloatComplex
// ============================================================================

// 1-Qubit Gate Kernel
template<typename Complex>
__global__ void apply_1q_gate_kernel_t(
    Complex* __restrict__ psi,
    const Complex* __restrict__ U,
    const long long n,
    const int q
);

// 2-Qubit Gate Kernel (Qiskit little-endian convention)
template<typename Complex>
__global__ void apply_2q_gate_kernel_t(
    Complex* __restrict__ psi,
    const Complex* __restrict__ U,
    const long long n,
    const int q0,
    const int q1
);

// Diagonal 1-Qubit Gate Kernel
template<typename Complex>
__global__ void apply_diagonal_1q_gate_kernel_t(
    Complex* __restrict__ psi,
    const Complex phase0,
    const Complex phase1,
    const long long n,
    const int q
);

// Permute Bits Kernel
template<typename Complex>
__global__ void permute_bits_kernel_t(
    const Complex* __restrict__ in,
    Complex* __restrict__ out,
    const int* __restrict__ map_old_to_new,
    int n,
    unsigned long long dim
);

// Tile Kernel (cooperative groups)
template<typename Complex>
__global__ void apply_tile_kernel_t(
    Complex* __restrict__ psi,
    const TileGateOpT<Complex>* __restrict__ ops,
    const int num_ops,
    const long long num_qubits
);

// ============================================================================
// HOST LAUNCHER FUNCTIONS (templated)
// ============================================================================

template<typename Complex>
void launch_apply_1q_gate_t(
    Complex* psi,
    const Complex* U,
    long long num_qubits,
    int target_qubit,
    cudaStream_t stream = 0
);

template<typename Complex>
void launch_apply_2q_gate_t(
    Complex* psi,
    const Complex* U,
    long long num_qubits,
    int q0,
    int q1,
    cudaStream_t stream = 0
);

template<typename Complex>
void launch_apply_diagonal_1q_gate_t(
    Complex* psi,
    Complex phase0,
    Complex phase1,
    long long num_qubits,
    int target_qubit,
    cudaStream_t stream = 0
);

template<typename Complex>
void launch_permute_bits_t(
    const Complex* in,
    Complex* out,
    const int* d_map_old_to_new,
    int n,
    unsigned long long dim,
    cudaStream_t stream = 0
);

template<typename Complex>
void launch_apply_tile_t(
    Complex* psi,
    const TileGateOpT<Complex>* d_ops,
    int num_ops,
    long long num_qubits,
    cudaStream_t stream = 0
);

template<typename Complex>
int get_tile_kernel_max_blocks_t(int threads_per_block);

// ============================================================================
// LEGACY NON-TEMPLATED DECLARATIONS (for backward compatibility)
// These just call the templated versions with cuDoubleComplex
// ============================================================================

__global__ void apply_1q_gate_kernel(
    cuDoubleComplex* __restrict__ psi,
    const cuDoubleComplex* __restrict__ U,
    const long long n,
    const int q
);

__global__ void apply_2q_gate_kernel(
    cuDoubleComplex* __restrict__ psi,
    const cuDoubleComplex* __restrict__ U,
    const long long n,
    const int q0,
    const int q1
);

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

__global__ void apply_tile_kernel(
    cuDoubleComplex* __restrict__ psi,
    const TileGateOp* __restrict__ ops,
    const int num_ops,
    const long long num_qubits
);

void launch_permute_bits(
    const cuDoubleComplex* in,
    cuDoubleComplex* out,
    const int* d_map_old_to_new,
    int n,
    unsigned long long dim,
    cudaStream_t stream = 0
);

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

void launch_apply_tile(
    cuDoubleComplex* psi,
    const TileGateOp* d_ops,
    int num_ops,
    long long num_qubits,
    cudaStream_t stream = 0
);

int get_tile_kernel_max_blocks(int threads_per_block);

// ============================================================================
// FLOAT32 DECLARATIONS
// ============================================================================

void launch_permute_bits_f32(
    const cuFloatComplex* in,
    cuFloatComplex* out,
    const int* d_map_old_to_new,
    int n,
    unsigned long long dim,
    cudaStream_t stream = 0
);

void launch_apply_1q_gate_f32(
    cuFloatComplex* psi,
    const cuFloatComplex* U,
    long long num_qubits,
    int target_qubit,
    cudaStream_t stream = 0
);

void launch_apply_2q_gate_f32(
    cuFloatComplex* psi,
    const cuFloatComplex* U,
    long long num_qubits,
    int q0,
    int q1,
    cudaStream_t stream = 0
);

void launch_apply_tile_f32(
    cuFloatComplex* psi,
    const TileGateOpF32* d_ops,
    int num_ops,
    long long num_qubits,
    cudaStream_t stream = 0
);

int get_tile_kernel_max_blocks_f32(int threads_per_block);

#endif // QGPUSIM_KERNELS_CUH
