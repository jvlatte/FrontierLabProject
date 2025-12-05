from qiskit import QuantumCircuit
from qgpusim.transpiler.passes import Tile
from typing import List
import numpy as np


def apply_1q_gate(
    psi: np.ndarray,
    U: np.ndarray,
    q: int,
    num_qubits: int,
):
    """
    Apply a 2x2 unitary U to qubit q (global index) on statevector psi.
    psi is length 2^num_qubits.
    """
    dim = psi.shape[0]
    assert dim == (1 << num_qubits)
    mask = 1 << q

    for i in range(dim):
        if (i & mask) == 0:
            j = i | mask
            a0 = psi[i]
            a1 = psi[j]
            psi[i] = U[0, 0] * a0 + U[0, 1] * a1
            psi[j] = U[1, 0] * a0 + U[1, 1] * a1


def apply_2q_gate(
    psi: np.ndarray,
    U: np.ndarray,
    q0: int,
    q1: int,
    num_qubits: int,
):
    """
    Apply a 4x4 unitary U to qubits (q0, q1) on psi.
    We assume U is ordered in computational basis |00>,|01>,|10>,|11>.
    """
    dim = psi.shape[0]
    assert dim == (1 << num_qubits)
    if q0 == q1:
        raise ValueError("q0 and q1 must be different")

    low, high = sorted((q0, q1))
    mask_low = 1 << low
    mask_high = 1 << high

    for i in range(dim):
        # only process each 4-state group once
        if (i & mask_low) or (i & mask_high):
            continue

        i00 = i
        i01 = i | mask_low
        i10 = i | mask_high
        i11 = i | mask_low | mask_high

        v = np.array([psi[i00], psi[i01], psi[i10], psi[i11]], dtype=np.complex128)
        v_new = U @ v

        psi[i00], psi[i01], psi[i10], psi[i11] = v_new


def run_custom_backend(qc: QuantumCircuit, tile_plan: List[Tile], task: str, shots: int=None):
    print("\n\nRunning custom GPU backend (not implemented yet)\n\n")
    num_qubits = qc.num_qubits
    dim = 1 << num_qubits
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0

    print("\n=== CUSTOM TILE BACKEND (CPU stub) ===")
    print(f"  task: {task}, shots: {shots}")
    print(f"  num_qubits: {num_qubits}")
    print(f"  num_tiles: {len(tile_plan)}")

    for tile in tile_plan:
        print(f"\n  Tile {tile.tile_idx}:")
        print(f"    logical_qubits (global): {tile.logical_qubits}")
        print(f"    global_to_local: {tile.global_to_local_map}")
        print(f"    num_gates: {len(tile.gates)}")

        for node in tile.gates:
            op = node.op

            # Skip non-unitary / snapshot ops like SaveStatevector, barriers, etc.
            if not hasattr(op, "to_matrix"):
                print(f"      [skip] non-unitary op: {op.name}")
                continue

            try:
                U = op.to_matrix()
            except Exception:
                print(f"      [skip] op {op.name} has no matrix representation")
                continue

            qargs = node.qargs
            global_qubits = [qc.find_bit(q).index for q in qargs]

            if U.shape == (2, 2) and len(global_qubits) == 1:
                apply_1q_gate(psi, U, global_qubits[0], num_qubits)
            elif U.shape == (4, 4) and len(global_qubits) == 2:
                apply_2q_gate(psi, U, global_qubits[0], global_qubits[1], num_qubits)
            else:
                print(
                    f"      [skip] unsupported gate {op.name} "
                    f"shape={U.shape} on {len(global_qubits)} qubits"
                )
                continue

    return psi
