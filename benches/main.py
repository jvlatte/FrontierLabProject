import argparse
from qiskit import QuantumCircuit, transpile    
from qiskit_aer import AerSimulator
from qiskit_aer.library import l
from qiskit.circuit.library import QFT

import numpy as np
import random



def build_circuit(circuit: str, n_qubits: int, depth: int, seed: int):
    """builds test circuit
    ghz: GHZ state
    random: random layers of H, RX, RZ, CX
    qft: quantum fourier transform
    """
    qc = QuantumCircuit(n_qubits)
    if circuit == "ghz":
        qc.h(0)
        for i in range(1, n_qubits):
            qc.cx(0, i)
    
    elif circuit == "random":
        rng = random.Random(seed)
        for _ in range(max(1, depth)):
            for i in range(n_qubits):
                # random single-qubit rotations
                qc.rz(2 * np.pi * rng.random(), i)
                qc.rx(2 * np.pi * rng.random(), i)
            # random nearest-neighbor CX chain
            for i in range(0, n_qubits - 1, 2):
                qc.cx(i, i + 1)
            for i in range(1, n_qubits - 1, 2):
                qc.cx(i, i + 1)
        return qc

    elif circuit == "qft":
        qc = QFT(num_qubits=n_qubits, do_swaps=False)
        return qc
    
    else:
        raise ValueError(f"Unknown circuit type: {circuit}")


def create_simulator(backend: str, task: str):
    """creates AerSimulator"""
    method = "automatic"
    if task == "statevector":
        method = "statevector"
    elif task == "sampling":
        method = "automatic"
    
    if backend == "cpu":
        sim = AerSimulator(method=method)  # default is CPU
    elif backend == "gpu":
        try:
            sim = AerSimulator(method=method, device="GPU")
        except TypeError:
            sim = AerSimulator(method=method)
            print("GPU backend not available, falling back to CPU.")
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return sim


if __name__ == "__main__":
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

        # prepare backends
        if args.backend == "compare":
            # compare both cpu and gpu statevector
            pass
        else:
            # only cpu or gpu+cpu
            sim = create_simulator(backend=args.backend, task=args.tasks)

    if __name__ == "__main__":
        main()