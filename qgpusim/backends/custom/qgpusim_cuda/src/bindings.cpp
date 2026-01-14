#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
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

// Lightweight error check - only in debug builds
#ifdef NDEBUG
    #define CUDA_CHECK_KERNEL() ((void)0)
#else
    #define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaPeekAtLastError())
#endif


// ===========================================================================
// Persistent GPU buffers for standalone functions (avoid per-gate malloc/free)
// ===========================================================================
class PersistentGateBuffers {
public:
    static PersistentGateBuffers& instance() {
        static PersistentGateBuffers inst;
        return inst;
    }
    
    cuDoubleComplex* get_1q_buffer() {
        if (!d_U_1q_) {
            CUDA_CHECK(cudaMalloc(&d_U_1q_, 4 * sizeof(cuDoubleComplex)));
        }
        return d_U_1q_;
    }
    
    cuDoubleComplex* get_2q_buffer() {
        if (!d_U_2q_) {
            CUDA_CHECK(cudaMalloc(&d_U_2q_, 16 * sizeof(cuDoubleComplex)));
        }
        return d_U_2q_;
    }
    
    ~PersistentGateBuffers() {
        if (d_U_1q_) cudaFree(d_U_1q_);
        if (d_U_2q_) cudaFree(d_U_2q_);
    }
    
private:
    PersistentGateBuffers() : d_U_1q_(nullptr), d_U_2q_(nullptr) {}
    PersistentGateBuffers(const PersistentGateBuffers&) = delete;
    PersistentGateBuffers& operator=(const PersistentGateBuffers&) = delete;
    
    cuDoubleComplex* d_U_1q_;
    cuDoubleComplex* d_U_2q_;
};


static cuDoubleComplex* get_cupy_ptr(py::object cupy_arr) {
    py::dict cuda_iface = cupy_arr.attr("__cuda_array_interface__").cast<py::dict>();
    py::tuple data_tuple = cuda_iface["data"].cast<py::tuple>();
    uintptr_t ptr = data_tuple[0].cast<uintptr_t>();
    if (ptr == 0) throw std::runtime_error("CuPy array has null device pointer");
    return reinterpret_cast<cuDoubleComplex*>(ptr);
}


// Statevector class that manages GPU memory
class Statevector {
public:
    Statevector(int num_qubits) : num_qubits_(num_qubits) {
        dim_ = 1ULL << num_qubits;
        
        // Allocate GPU memory for statevector
        CUDA_CHECK(cudaMalloc(&d_psi_, dim_ * sizeof(cuDoubleComplex)));
        
        // Allocate GPU memory for gate matrix (max 4x4 = 16 elements)
        CUDA_CHECK(cudaMalloc(&d_U_, 16 * sizeof(cuDoubleComplex)));

        CUDA_CHECK(cudaMalloc(&d_tmp_, dim_ * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_map_, num_qubits_ * sizeof(int)));

        // Initialize to |0...0> state
        CUDA_CHECK(cudaMemset(d_psi_, 0, dim_ * sizeof(cuDoubleComplex)));
        
        cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
        CUDA_CHECK(cudaMemcpy(d_psi_, &one, sizeof(cuDoubleComplex), 
                              cudaMemcpyHostToDevice));
    }
    
    ~Statevector() {
        if (d_psi_) cudaFree(d_psi_);
        if (d_U_) cudaFree(d_U_);

        if (d_tmp_) cudaFree(d_tmp_);
        if (d_map_) cudaFree(d_map_);
    }
    
    // Disable copy
    Statevector(const Statevector&) = delete;
    Statevector& operator=(const Statevector&) = delete;
    
    // Move semantics
    Statevector(Statevector&& other) noexcept 
        : d_psi_(other.d_psi_), d_U_(other.d_U_), 
          d_tmp_(other.d_tmp_), d_map_(other.d_map_),
          num_qubits_(other.num_qubits_), dim_(other.dim_) {
        other.d_psi_ = nullptr;
        other.d_U_ = nullptr;
        other.d_tmp_ = nullptr;
        other.d_map_ = nullptr;
    }

    void permute(py::list layout_old_py, py::list layout_new_py) {
        if ((int)py::len(layout_old_py) != num_qubits_ ||
            (int)py::len(layout_new_py) != num_qubits_) {
            throw std::runtime_error("layout_old/layout_new must have length num_qubits");
        }

        std::vector<int> layout_old(num_qubits_);
        std::vector<int> layout_new(num_qubits_);
        for (int i = 0; i < num_qubits_; ++i) {
            layout_old[i] = layout_old_py[i].cast<int>();
            layout_new[i] = layout_new_py[i].cast<int>();
        }

        // Build new_pos[global_qubit] = new bit position
        std::vector<int> new_pos(num_qubits_, -1);
        for (int p = 0; p < num_qubits_; ++p) {
            int gq = layout_new[p];
            if (gq < 0 || gq >= num_qubits_) throw std::runtime_error("layout_new contains invalid qubit id");
            if (new_pos[gq] != -1) throw std::runtime_error("layout_new is not a permutation");
            new_pos[gq] = p;
        }

        // Build map_old_to_new[p_old] = p_new
        std::vector<int> map_old_to_new(num_qubits_, -1);
        for (int p_old = 0; p_old < num_qubits_; ++p_old) {
            int gq = layout_old[p_old];
            if (gq < 0 || gq >= num_qubits_) throw std::runtime_error("layout_old contains invalid qubit id");
            int p_new = new_pos[gq];
            if (p_new < 0) throw std::runtime_error("layout_old contains qubit not in layout_new");
            map_old_to_new[p_old] = p_new;
        }

        // Copy map to device
        CUDA_CHECK(cudaMemcpy(d_map_, map_old_to_new.data(),
                            num_qubits_ * sizeof(int),
                            cudaMemcpyHostToDevice));

        // Launch permute: d_tmp_[j] = d_psi_[i]
        launch_permute_bits(d_psi_, d_tmp_, d_map_, num_qubits_, dim_);
        CUDA_CHECK_KERNEL();

        // Swap buffers (now d_psi_ contains permuted state)
        std::swap(d_psi_, d_tmp_);
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
        CUDA_CHECK_KERNEL();
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
        CUDA_CHECK_KERNEL();
    }
    
    // Get statevector as numpy array
    py::array_t<std::complex<double>> to_numpy() {
        CUDA_CHECK(cudaDeviceSynchronize());
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

    // Synchronize device
    void synchronize() {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void apply_1q_dev(int target_qubit, py::object U_cupy) {
        if (target_qubit < 0 || target_qubit >= num_qubits_) {
            throw std::runtime_error("Invalid target qubit");
        }
        cuDoubleComplex* dU = get_cupy_ptr(U_cupy);
        launch_apply_1q_gate(d_psi_, dU, num_qubits_, target_qubit);
        CUDA_CHECK_KERNEL();
    }

    void apply_2q_dev(int q0, int q1, py::object U_cupy) {
        if (q0 < 0 || q0 >= num_qubits_ || q1 < 0 || q1 >= num_qubits_) {
            throw std::runtime_error("Invalid qubit indices");
        }
        if (q0 == q1) {
            throw std::runtime_error("q0 and q1 must be different");
        }
        cuDoubleComplex* dU = get_cupy_ptr(U_cupy);
        launch_apply_2q_gate(d_psi_, dU, num_qubits_, q0, q1);
        CUDA_CHECK_KERNEL();
    }

    // Apply diagonal 1-qubit gate (optimized for RZ, P, T, S, Z)
    void apply_diagonal_1q(int target_qubit, std::complex<double> phase0, std::complex<double> phase1) {
        if (target_qubit < 0 || target_qubit >= num_qubits_) {
            throw std::runtime_error("Invalid target qubit");
        }
        cuDoubleComplex p0 = make_cuDoubleComplex(phase0.real(), phase0.imag());
        cuDoubleComplex p1 = make_cuDoubleComplex(phase1.real(), phase1.imag());
        launch_apply_diagonal_1q_gate(d_psi_, p0, p1, num_qubits_, target_qubit);
        CUDA_CHECK_KERNEL();
    }

    
    int num_qubits() const { return num_qubits_; }
    size_t dim() const { return dim_; }

private:
    cuDoubleComplex* d_psi_ = nullptr;  // Device statevector
    cuDoubleComplex* d_U_ = nullptr;    // Device gate matrix
    int num_qubits_;
    size_t dim_;
    cuDoubleComplex* d_tmp_ = nullptr;
    int* d_map_ = nullptr;

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
    
    // Use persistent buffer instead of per-call malloc/free
    cuDoubleComplex* d_U = PersistentGateBuffers::instance().get_1q_buffer();
    
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
    CUDA_CHECK_KERNEL();
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
    
    // Use persistent buffer instead of per-call malloc/free
    cuDoubleComplex* d_U = PersistentGateBuffers::instance().get_2q_buffer();
    
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
    CUDA_CHECK_KERNEL();
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
        .def_property_readonly("dim", &Statevector::dim)

        .def("apply_1q_dev", &Statevector::apply_1q_dev,
            py::arg("target_qubit"), py::arg("U_dev"),
            "Apply a 1-qubit gate using a device-resident (CuPy) 2x2 matrix")

        .def("apply_2q_dev", &Statevector::apply_2q_dev,
            py::arg("q0"), py::arg("q1"), py::arg("U_dev"),
            "Apply a 2-qubit gate using a device-resident (CuPy) 4x4 matrix")

        .def("apply_diagonal_1q", &Statevector::apply_diagonal_1q,
            py::arg("target_qubit"), py::arg("phase0"), py::arg("phase1"),
            "Apply a diagonal 1-qubit gate (optimized for RZ, P, T, S, Z gates)")

        .def("synchronize", &Statevector::synchronize,
            "Synchronize the device (use sparingly)")

        .def("permute", &Statevector::permute,
            py::arg("layout_old"), py::arg("layout_new"),
            "Permute qubit bit-positions by reindexing the statevector");


    
    // Standalone functions for CuPy arrays
    m.def("apply_1q_gate", &apply_1q_gate_cupy,
          py::arg("psi"), py::arg("U"), py::arg("target_qubit"), py::arg("num_qubits"),
          "Apply a 1-qubit gate to a CuPy statevector array");
    
    m.def("apply_2q_gate", &apply_2q_gate_cupy,
          py::arg("psi"), py::arg("U"), py::arg("q0"), py::arg("q1"), py::arg("num_qubits"),
          "Apply a 2-qubit gate to a CuPy statevector array");
}
