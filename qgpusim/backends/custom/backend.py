from qiskit import QuantumCircuit
from qgpusim.transpiler.passes import Tile
from typing import List
import numpy as np
import cupy as cp

# cache dict
GATE_CACHE = {} # (name, parameters) -> cp matrix

# gate cache helper
def get_gate_matrix(op):
    try:
        params = tuple(float(p) for p in getattr(op, 'params', []))
    except TypeError:
        params = None
    
    key = (op.name, params)

    if key in GATE_CACHE:
        return GATE_CACHE[key]

    # compute and cache
    U_cpu = op.to_matrix()
    U_gpu = cp.asarray(U_cpu, dtype=cp.complex128)
    GATE_CACHE[key] = U_gpu
    return U_gpu

# cpu functions

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


# gpu functions

def apply_1q_gate_gpu(
    psi: cp.ndarray,
    U: cp.ndarray,
    q: int,
    num_qubits: int,
    idx: cp.ndarray = None
):
    dim = psi.shape[0]
    assert dim == (1 << num_qubits)
    mask = 1 << q

    # # indices 0...2^n - 1 on gpu
    # idx = cp.arange(dim, dtype=cp.int64)

    lower_mask = (idx & mask) == 0
    i = idx[lower_mask]
    j = i | mask

    a0 = psi[i]
    a1 = psi[j]

    psi[i] = U[0, 0] * a0 + U[0, 1] * a1
    psi[j] = U[1, 0] * a0 + U[1, 1] * a1

def apply_2q_gate_gpu(
    psi: cp.ndarray,
    U: cp.ndarray,
    q0: int,
    q1: int,
    num_qubits: int,
    idx: cp.ndarray = None
): 
    dim = psi.shape[0]
    assert dim == (1 << num_qubits)
    if q0 == q1:
        raise ValueError("q0 and q1 must be different")

    low, high = sorted((q0, q1))
    mask_low = 1 << low
    mask_high = 1 << high

    # idx = cp.arange(dim, dtype=cp.int64)
    base_mask = ((idx & mask_low) == 0) & ((idx & mask_high) == 0)
    base = idx[base_mask]

    if base.size == 0:
        return
    
    i00 = base
    i01 = base | mask_low
    i10 = base | mask_high
    i11 = base | mask_low | mask_high

    v00 = psi[i00]
    v01 = psi[i01]
    v10 = psi[i10]
    v11 = psi[i11]

    vecs = cp.stack([v00, v01, v10, v11], axis=0)
    new_vecs = U @ vecs

    psi[i00] = new_vecs[0]
    psi[i01] = new_vecs[1]
    psi[i10] = new_vecs[2]
    psi[i11] = new_vecs[3]

def run_custom_backend(qc: QuantumCircuit, tile_plan: List[Tile], task: str, shots: int=None):
    print("\n\nRunning custom GPU backend\n\n")
    num_qubits = qc.num_qubits
    dim = 1 << num_qubits
    # psi = np.zeros(dim, dtype=np.complex128)
    # psi[0] = 1.0
    psi_gpu = cp.zeros(dim, dtype=cp.complex128)
    psi_gpu[0] = 1.0 + 0.0j

    idx = cp.arange(dim, dtype=cp.int64)

    print("\n=== CUSTOM TILE BACKEND (CPU stub) ===")
    print(f"  task: {task}, shots: {shots}")
    print(f"  num_qubits: {num_qubits}")
    print(f"  num_tiles: {len(tile_plan)}")

    for tile in tile_plan:
       for node in tile.gates:
            op = node.op

            # Skip non-unitary / snapshot ops like SaveStatevector, barriers, etc.
            if not hasattr(op, "to_matrix"):
                print(f"      [skip] non-unitary op: {op.name}")
                continue

            try:
                # U_cpu = op.to_matrix()
                U_gpu = get_gate_matrix(op)
            except Exception:
                print(f"      [skip] op {op.name} has no matrix representation")
                continue

            # U_gpu = cp.asarray(U_cpu, dtype=cp.complex128)

            qargs = node.qargs
            global_qubits = [qc.find_bit(q).index for q in qargs]

            if U_gpu.shape == (2, 2) and len(global_qubits) == 1:
                apply_1q_gate_gpu(psi_gpu, U_gpu, global_qubits[0], num_qubits, idx)
            elif U_gpu.shape == (4, 4) and len(global_qubits) == 2:
                apply_2q_gate_gpu(psi_gpu, U_gpu, global_qubits[0], global_qubits[1], num_qubits, idx)
            else:
                print(
                    f"      [skip] unsupported gate {op.name} "
                    f"shape={U_gpu.shape} on {len(global_qubits)} qubits"
                )
                continue
    psi_cpu = cp.asnumpy(psi_gpu)

    return psi_cpu
