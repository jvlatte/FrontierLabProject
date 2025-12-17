#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "statevector.hpp"

namespace py = pybind11;

static std::vector<std::complex<double>> matrix_to_vec(py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> arr) {
    auto buf = arr.request();
    auto* ptr = static_cast<std::complex<double>*>(buf.ptr);
    return std::vector<std::complex<double>>(ptr, ptr + buf.size);
}

PYBIND11_MODULE(qgpusim_cuda, m) {
    py::class_<Statevector>(m, "Statevector")
        .def(py::init<int>())
        .def("apply_1q", [](Statevector& sv, int q, py::array_t<std::complex<double>> U) {
            auto v = matrix_to_vec(U);
            sv.apply_1q(q, v);
        })
        .def("apply_2q", [](Statevector& sv, int q0, int q1, py::array_t<std::complex<double>> U) {
            auto v = matrix_to_vec(U);
            sv.apply_2q(q0, q1, v);
        })
        .def("to_numpy", [](const Statevector& sv) {
            auto h = sv.to_host();
            // Return 1D complex128 numpy array
            py::array_t<std::complex<double>> out(h.size());
            auto buf = out.request();
            auto* ptr = static_cast<std::complex<double>*>(buf.ptr);
            std::copy(h.begin(), h.end(), ptr);
            return out;
        })
        .def_property_readonly("n_qubits", &Statevector::n_qubits)
        .def_property_readonly("dim", &Statevector::dim);
}
