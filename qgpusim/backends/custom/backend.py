from qiskit import QuantumCircuit
from qgpusim.transpiler.passes import Tile
from typing import List
import numpy as np
import cupy as cp

# cache dict
GATE_CACHE = {} # (name, parameters) -> numpy matrix for native module

# Try to import the native CUDA module (pybind11-based)
try:
    from .native import qgpusim_cuda
    USE_NATIVE = True
    print("[qgpusim] Using native pybind11 CUDA backend")
except ImportError as e:
    USE_NATIVE = False
    print(f"[qgpusim] Native CUDA module not available: {e}")
    print("[qgpusim] Falling back to CuPy-based implementation")


# gate cache helper
def get_gate_matrix(op, for_native=False):
    """Get gate matrix from cache or compute it.
    
    Args:
        op: The quantum gate operation
        for_native: If True, return numpy array for native module.
                   If False, return CuPy array for fallback mode.
    """
    try:
        params = tuple(float(p) for p in getattr(op, 'params', []))
    except TypeError:
        params = None
    
    key = (op.name, params, for_native)

    if key in GATE_CACHE:
        return GATE_CACHE[key]

    # compute and cache
    U_cpu = op.to_matrix()
    if for_native:
        # Native module expects numpy arrays (will transfer to GPU internally)
        U = np.asarray(U_cpu, dtype=np.complex128)
    else:
        # Fallback CuPy mode
        U = cp.asarray(U_cpu, dtype=cp.complex128)
    GATE_CACHE[key] = U
    return U


# Native module wrapper functions (using pybind11 compiled CUDA)
def apply_1q_gate_native(psi_cupy, U_numpy, q, num_qubits):
    """Apply 1-qubit gate using native pybind11 CUDA module."""
    qgpusim_cuda.apply_1q_gate(psi_cupy, U_numpy, q, num_qubits)


def apply_2q_gate_native(psi_cupy, U_numpy, q0, q1, num_qubits):
    """Apply 2-qubit gate using native pybind11 CUDA module."""
    qgpusim_cuda.apply_2q_gate(psi_cupy, U_numpy, q0, q1, num_qubits)


# Fallback CuPy-based functions (used when native module is not available)
def apply_1q_gate_cupy_fallback(
    psi: cp.ndarray,
    U: cp.ndarray,
    q: int,
    num_qubits: int,
    idx: cp.ndarray = None
):
    """Fallback 1-qubit gate using pure CuPy (no custom kernels)."""
    dim = psi.shape[0]
    assert dim == (1 << num_qubits)
    mask = 1 << q

    if idx is None:
        idx = cp.arange(dim, dtype=cp.int64)

    lower_mask = (idx & mask) == 0
    i = idx[lower_mask]
    j = i | mask

    a0 = psi[i]
    a1 = psi[j]

    psi[i] = U[0, 0] * a0 + U[0, 1] * a1
    psi[j] = U[1, 0] * a0 + U[1, 1] * a1


def apply_2q_gate_cupy_fallback(
    psi: cp.ndarray,
    U: cp.ndarray,
    q0: int,
    q1: int,
    num_qubits: int,
    idx: cp.ndarray = None
):
    """Fallback 2-qubit gate using pure CuPy (no custom kernels)."""
    dim = psi.shape[0]
    assert dim == (1 << num_qubits)
    if q0 == q1:
        raise ValueError("q0 and q1 must be different")

    low, high = sorted((q0, q1))
    mask_low = 1 << low
    mask_high = 1 << high

    if idx is None:
        idx = cp.arange(dim, dtype=cp.int64)

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
    print("\n\n=== Running custom GPU backend ===\n")
    num_qubits = qc.num_qubits

    # Use fully native Statevector class (all GPU operations, minimal CPU transfers)
    if USE_NATIVE:
        print("  Mode: Native pybind11 CUDA (Statevector class)\n")
        sv = qgpusim_cuda.Statevector(num_qubits)

        for tile in tile_plan:
            for node in tile.gates:
                op = node.op

                if not hasattr(op, "to_matrix"):
                    continue

                try:
                    U = get_gate_matrix(op, for_native=True)  # numpy array
                except Exception:
                    continue

                qargs = node.qargs
                global_qubits = [qc.find_bit(q).index for q in qargs]

                if U.shape == (2, 2) and len(global_qubits) == 1:
                    sv.apply_1q(global_qubits[0], U)
                elif U.shape == (4, 4) and len(global_qubits) == 2:
                    sv.apply_2q(global_qubits[0], global_qubits[1], U)
                else:
                    continue
        print("  Finished circuit execution using native CUDA module.\n")
        return sv.to_numpy()

    # Fallback: Use CuPy arrays with native kernel functions (if available) or pure CuPy
    print("  Mode: CuPy-based fallback\n")
    dim = 1 << num_qubits
    psi_gpu = cp.zeros(dim, dtype=cp.complex128)
    psi_gpu[0] = 1.0 + 0.0j
    idx = cp.arange(dim, dtype=cp.int64)

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
                U_gpu = get_gate_matrix(op, for_native=False)  # CuPy array
            except Exception:
                print(f"      [skip] op {op.name} has no matrix representation")
                continue

            qargs = node.qargs
            global_qubits = [qc.find_bit(q).index for q in qargs]

            if U_gpu.shape == (2, 2) and len(global_qubits) == 1:
                apply_1q_gate_cupy_fallback(psi_gpu, U_gpu, global_qubits[0], num_qubits, idx)
            elif U_gpu.shape == (4, 4) and len(global_qubits) == 2:
                apply_2q_gate_cupy_fallback(psi_gpu, U_gpu, global_qubits[0], global_qubits[1], num_qubits, idx)
            else:
                print(
                    f"      [skip] unsupported gate {op.name} "
                    f"shape={U_gpu.shape} on {len(global_qubits)} qubits"
                )
                continue
    
    psi_cpu = cp.asnumpy(psi_gpu)
    return psi_cpu
