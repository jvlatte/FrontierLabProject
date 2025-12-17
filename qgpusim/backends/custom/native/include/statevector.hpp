#pragma once
#include <vector>
#include <complex>

class Statevector {
public:
    explicit Statevector(int n_qubits);
    ~Statevector();

    void apply_1q(int q, const std::vector<std::complex<double>>& U2x2);
    void apply_2q(int q0, int q1, const std::vector<std::complex<double>>& U4x4);

    std::vector<std::complex<double>> to_host() const;

    int n_qubits() const { return n_; }
    long long dim() const { return dim_; }

private:
    int n_;
    long long dim_;
    void* d_psi_; // device pointer
};
