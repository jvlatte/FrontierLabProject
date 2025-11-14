from qiskit import QuantumCircuit   
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

