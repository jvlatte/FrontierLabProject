import argparse
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.library import SaveStatevector




def build_circuit(circuit: str, n_qubits: int, depth: int, seed: int):
    pass


def main():
    parser = argparse.ArgumentParser(description="Benchmarking Script")

    parser.add_argument("--backend", type=str, choices=["cpu", "gpu", "compare"], default="cpu")
    parser.add_argument("--tasks", type=str, choices=["statevector", "sampling"], default="statevector")
    parser.add_argument("--circuit", type=str, choices=["random", "ghz", "qft"], default="random")
    parser.add_argument("--nqubits", type=int, default=10)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # prepare circuit
    qc = build_circuit(circuit=args.circuit, n_qubits=args.nqubits, depth=args.depth, seed=args.seed)


if __name__ == "__main__":
    main()