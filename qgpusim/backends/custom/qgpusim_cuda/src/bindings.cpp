#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <complex>
#include <stdexcept>
#include <vector>

#include "kernels.cuh"

namespace py = pybind11;

// CUDA error checking macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            throw std::runtime_error(std::string("CUDA error: ") +            \
                                     cudaGetErrorString(err));                \
        }                                                                      \
    } while (0)


// Statevector class that manages GPU memory
class Statevector {
public:
    Statevector(int num_qubits) : num_qubits_(num_qubits) {
        dim_ = 1ULL << num_qubits;
        
        // Allocate GPU memory for statevector
        CUDA_CHECK(cudaMalloc(&d_psi_, dim_ * sizeof(cuDoubleComplex)));
        
        // Allocate GPU memory for gate matrix (max 4x4 = 16 elements)
        CUDA_CHECK(cudaMalloc(&d_U_, 16 * sizeof(cuDoubleComplex)));
        
        // Initialize to |0...0> state
        CUDA_CHECK(cudaMemset(d_psi_, 0, dim_ * sizeof(cuDoubleComplex)));
        
        cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
        CUDA_CHECK(cudaMemcpy(d_psi_, &one, sizeof(cuDoubleComplex), 
                              cudaMemcpyHostToDevice));
    }
    
    ~Statevector() {
        if (d_psi_) cudaFree(d_psi_);
        if (d_U_) cudaFree(d_U_);
    }
    
    // Disable copy
    Statevector(const Statevector&) = delete;
    Statevector& operator=(const Statevector&) = delete;
    
    // Move semantics
    Statevector(Statevector&& other) noexcept 
        : d_psi_(other.d_psi_), d_U_(other.d_U_), 
          num_qubits_(other.num_qubits_), dim_(other.dim_) {
        other.d_psi_ = nullptr;
        other.d_U_ = nullptr;
    }
    
    // Apply 1-qubit gate
    void apply_1q(int target_qubit, py::array_t<std::complex<double>> U_numpy) {
        auto U_buf = U_numpy.request();
        
        if (U_buf.ndim != 2 || U_buf.shape[0] != 2 || U_buf.shape[1] != 2) {
            throw std::runtime_error("U must be a 2x2 matrix");
        }
        
        if (target_qubit < 0 || target_qubit >= num_qubits_) {
            throw std::runtime_error("Invalid target qubit");
        }
        
        // Copy gate matrix to device
        auto U_ptr = static_cast<std::complex<double>*>(U_buf.ptr);
        CUDA_CHECK(cudaMemcpy(d_U_, U_ptr, 4 * sizeof(cuDoubleComplex),
                              cudaMemcpyHostToDevice));
        
        // Launch kernel
        launch_apply_1q_gate(d_psi_, d_U_, num_qubits_, target_qubit);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Apply 2-qubit gate
    void apply_2q(int q0, int q1, py::array_t<std::complex<double>> U_numpy) {
        auto U_buf = U_numpy.request();
        
        if (U_buf.ndim != 2 || U_buf.shape[0] != 4 || U_buf.shape[1] != 4) {
            throw std::runtime_error("U must be a 4x4 matrix");
        }
        
        if (q0 < 0 || q0 >= num_qubits_ || q1 < 0 || q1 >= num_qubits_) {
            throw std::runtime_error("Invalid qubit indices");
        }
        
        if (q0 == q1) {
            throw std::runtime_error("q0 and q1 must be different");
        }
        
        // Copy gate matrix to device
        auto U_ptr = static_cast<std::complex<double>*>(U_buf.ptr);
        CUDA_CHECK(cudaMemcpy(d_U_, U_ptr, 16 * sizeof(cuDoubleComplex),
                              cudaMemcpyHostToDevice));
        
        // Launch kernel
        launch_apply_2q_gate(d_psi_, d_U_, num_qubits_, q0, q1);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Get statevector as numpy array
    py::array_t<std::complex<double>> to_numpy() {
        // Create numpy array
        py::array_t<std::complex<double>> result(dim_);
        auto result_buf = result.request();
        auto result_ptr = static_cast<std::complex<double>*>(result_buf.ptr);
        
        // Copy from device to host
        CUDA_CHECK(cudaMemcpy(result_ptr, d_psi_, 
                              dim_ * sizeof(cuDoubleComplex),
                              cudaMemcpyDeviceToHost));
        
        return result;
    }
    
    // Set statevector from numpy array
    void from_numpy(py::array_t<std::complex<double>> psi_numpy) {
        auto psi_buf = psi_numpy.request();
        
        if (psi_buf.ndim != 1 || static_cast<size_t>(psi_buf.shape[0]) != dim_) {
            throw std::runtime_error("Input array must have size 2^num_qubits");
        }
        
        auto psi_ptr = static_cast<std::complex<double>*>(psi_buf.ptr);
        CUDA_CHECK(cudaMemcpy(d_psi_, psi_ptr, 
                              dim_ * sizeof(cuDoubleComplex),
                              cudaMemcpyHostToDevice));
    }
    
    // Reset to |0...0> state
    void reset() {
        CUDA_CHECK(cudaMemset(d_psi_, 0, dim_ * sizeof(cuDoubleComplex)));
        cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
        CUDA_CHECK(cudaMemcpy(d_psi_, &one, sizeof(cuDoubleComplex), 
                              cudaMemcpyHostToDevice));
    }
    
    int num_qubits() const { return num_qubits_; }
    size_t dim() const { return dim_; }

private:
    cuDoubleComplex* d_psi_ = nullptr;  // Device statevector
    cuDoubleComplex* d_U_ = nullptr;    // Device gate matrix
    int num_qubits_;
    size_t dim_;
};


// Standalone function to apply 1-qubit gate to a CuPy array
void apply_1q_gate_cupy(
    py::object psi_cupy,  // CuPy array
    py::array_t<std::complex<double>> U_numpy,
    int target_qubit,
    int num_qubits
) {
    // Get the raw pointer from CuPy array using __cuda_array_interface__
    py::dict cuda_iface = psi_cupy.attr("__cuda_array_interface__").cast<py::dict>();
    py::tuple data_tuple = cuda_iface["data"].cast<py::tuple>();
    uintptr_t psi_ptr = data_tuple[0].cast<uintptr_t>();
    
    auto U_buf = U_numpy.request();
    if (U_buf.ndim != 2 || U_buf.shape[0] != 2 || U_buf.shape[1] != 2) {
        throw std::runtime_error("U must be a 2x2 matrix");
    }
    
    // Allocate temporary device memory for the gate matrix
    cuDoubleComplex* d_U;
    CUDA_CHECK(cudaMalloc(&d_U, 4 * sizeof(cuDoubleComplex)));
    
    auto U_ptr = static_cast<std::complex<double>*>(U_buf.ptr);
    CUDA_CHECK(cudaMemcpy(d_U, U_ptr, 4 * sizeof(cuDoubleComplex),
                          cudaMemcpyHostToDevice));
    
    // Launch kernel
    launch_apply_1q_gate(
        reinterpret_cast<cuDoubleComplex*>(psi_ptr),
        d_U,
        num_qubits,
        target_qubit
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaFree(d_U);
}

// Standalone function to apply 2-qubit gate to a CuPy array
void apply_2q_gate_cupy(
    py::object psi_cupy,  // CuPy array
    py::array_t<std::complex<double>> U_numpy,
    int q0,
    int q1,
    int num_qubits
) {
    // Get the raw pointer from CuPy array using __cuda_array_interface__
    py::dict cuda_iface = psi_cupy.attr("__cuda_array_interface__").cast<py::dict>();
    py::tuple data_tuple = cuda_iface["data"].cast<py::tuple>();
    uintptr_t psi_ptr = data_tuple[0].cast<uintptr_t>();
    
    auto U_buf = U_numpy.request();
    if (U_buf.ndim != 2 || U_buf.shape[0] != 4 || U_buf.shape[1] != 4) {
        throw std::runtime_error("U must be a 4x4 matrix");
    }
    
    if (q0 == q1) {
        throw std::runtime_error("q0 and q1 must be different");
    }
    
    // Allocate temporary device memory for the gate matrix
    cuDoubleComplex* d_U;
    CUDA_CHECK(cudaMalloc(&d_U, 16 * sizeof(cuDoubleComplex)));
    
    auto U_ptr = static_cast<std::complex<double>*>(U_buf.ptr);
    CUDA_CHECK(cudaMemcpy(d_U, U_ptr, 16 * sizeof(cuDoubleComplex),
                          cudaMemcpyHostToDevice));
    
    // Launch kernel
    launch_apply_2q_gate(
        reinterpret_cast<cuDoubleComplex*>(psi_ptr),
        d_U,
        num_qubits,
        q0, q1
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaFree(d_U);
}


PYBIND11_MODULE(qgpusim_cuda, m) {
    m.doc() = "CUDA-accelerated quantum gate simulation";
    
    // Statevector class
    py::class_<Statevector>(m, "Statevector")
        .def(py::init<int>(), py::arg("num_qubits"),
             "Create a statevector initialized to |0...0>")
        .def("apply_1q", &Statevector::apply_1q,
             py::arg("target_qubit"), py::arg("U"),
             "Apply a 1-qubit gate")
        .def("apply_2q", &Statevector::apply_2q,
             py::arg("q0"), py::arg("q1"), py::arg("U"),
             "Apply a 2-qubit gate")
        .def("to_numpy", &Statevector::to_numpy,
             "Get the statevector as a numpy array")
        .def("from_numpy", &Statevector::from_numpy,
             py::arg("psi"), "Set the statevector from a numpy array")
        .def("reset", &Statevector::reset,
             "Reset to |0...0> state")
        .def_property_readonly("num_qubits", &Statevector::num_qubits)
        .def_property_readonly("dim", &Statevector::dim);
    
    // Standalone functions for CuPy arrays
    m.def("apply_1q_gate", &apply_1q_gate_cupy,
          py::arg("psi"), py::arg("U"), py::arg("target_qubit"), py::arg("num_qubits"),
          "Apply a 1-qubit gate to a CuPy statevector array");
    
    m.def("apply_2q_gate", &apply_2q_gate_cupy,
          py::arg("psi"), py::arg("U"), py::arg("q0"), py::arg("q1"), py::arg("num_qubits"),
          "Apply a 2-qubit gate to a CuPy statevector array");
}
