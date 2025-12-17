#include "statevector.hpp"
#include "kernels.cuh"

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdexcept>

static inline void cuda_check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
    }
}

Statevector::Statevector(int n_qubits)
: n_(n_qubits), dim_(1LL << n_qubits), d_psi_(nullptr)
{
    cuda_check(cudaMalloc(&d_psi_, sizeof(cuDoubleComplex) * dim_), "cudaMalloc d_psi");

    // initialize |0...0>
    std::vector<std::complex<double>> h(dim_, {0.0, 0.0});
    h[0] = {1.0, 0.0};
    cuda_check(cudaMemcpy(d_psi_, h.data(),
                          sizeof(cuDoubleComplex) * dim_,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy init psi");
}

Statevector::~Statevector() {
    if (d_psi_) cudaFree(d_psi_);
}

void Statevector::apply_1q(int q, const std::vector<std::complex<double>>& U2x2) {
    if (U2x2.size() != 4) throw std::runtime_error("apply_1q expects 4 complex numbers");

    cuDoubleComplex* psi = reinterpret_cast<cuDoubleComplex*>(d_psi_);

    // copy U to device (tiny)
    cuDoubleComplex dU[4];
    for (int i = 0; i < 4; ++i) dU[i] = make_cuDoubleComplex(U2x2[i].real(), U2x2[i].imag());

    cuDoubleComplex* dU_ptr = nullptr;
    cuda_check(cudaMalloc(&dU_ptr, sizeof(dU)), "cudaMalloc dU");
    cuda_check(cudaMemcpy(dU_ptr, dU, sizeof(dU), cudaMemcpyHostToDevice), "cudaMemcpy dU");

    unsigned long long num_pairs = (unsigned long long)dim_ / 2ULL;
    int threads = 256;
    int blocks = (int)((num_pairs + threads - 1) / threads);

    apply_1q_gate_kernel<<<blocks, threads>>>(psi, dU_ptr, (long long)n_, q);
    cuda_check(cudaGetLastError(), "apply_1q kernel launch");

    cuda_check(cudaFree(dU_ptr), "cudaFree dU");
}

void Statevector::apply_2q(int q0, int q1, const std::vector<std::complex<double>>& U4x4) {
    if (U4x4.size() != 16) throw std::runtime_error("apply_2q expects 16 complex numbers");

    cuDoubleComplex* psi = reinterpret_cast<cuDoubleComplex*>(d_psi_);

    cuDoubleComplex dU[16];
    for (int i = 0; i < 16; ++i) dU[i] = make_cuDoubleComplex(U4x4[i].real(), U4x4[i].imag());

    cuDoubleComplex* dU_ptr = nullptr;
    cuda_check(cudaMalloc(&dU_ptr, sizeof(dU)), "cudaMalloc dU");
    cuda_check(cudaMemcpy(dU_ptr, dU, sizeof(dU), cudaMemcpyHostToDevice), "cudaMemcpy dU");

    unsigned long long num_groups = (unsigned long long)dim_ / 4ULL;
    int threads = 256;
    int blocks = (int)((num_groups + threads - 1) / threads);

    apply_2q_gate_kernel<<<blocks, threads>>>(psi, dU_ptr, (long long)n_, q0, q1);
    cuda_check(cudaGetLastError(), "apply_2q kernel launch");

    cuda_check(cudaFree(dU_ptr), "cudaFree dU");
}

std::vector<std::complex<double>> Statevector::to_host() const {
    std::vector<std::complex<double>> h(dim_);
    cuda_check(cudaMemcpy(h.data(), d_psi_,
                          sizeof(cuDoubleComplex) * dim_,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy psi to host");
    return h;
}
