from qiskit import QuantumCircuit
from qgpusim.transpiler.passes import Tile
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict
import numpy as np
import cupy as cp


# LRU-bounded cache for gate matrices
# (name, params) -> {"U_host": np.ndarray, "U_dev": cp.ndarray}
MAX_CACHE_SIZE = 512
GATE_CACHE = OrderedDict()


def _evict_cache_if_needed():
    """Evict oldest entries if cache exceeds max size."""
    while len(GATE_CACHE) > MAX_CACHE_SIZE:
        # Pop oldest item and free GPU memory if present
        _, item = GATE_CACHE.popitem(last=False)
        if "U_dev" in item:
            del item["U_dev"]

def _gate_key(op):
    try:
        params = tuple(float(p) for p in getattr(op, "params", []))
    except TypeError:
        params = None
    return (op.name, params)

def get_gate_matrix_host(op, dtype=np.complex128):
    """Always returns a numpy matrix (cached) with LRU eviction."""
    key = _gate_key(op)
    item = GATE_CACHE.get(key)
    if item is not None and "U_host" in item:
        # Move to end for LRU tracking
        GATE_CACHE.move_to_end(key)
        return item["U_host"]

    U_cpu = op.to_matrix()
    U_host = np.asarray(U_cpu, dtype=dtype, order="C")

    if item is None:
        _evict_cache_if_needed()
        item = {}
        GATE_CACHE[key] = item
    item["U_host"] = U_host
    return U_host

def get_gate_matrix_dev(op, dtype=cp.complex128):
    """Returns a CuPy matrix on GPU (cached) with LRU eviction."""
    key = _gate_key(op)
    item = GATE_CACHE.get(key)
    if item is not None and "U_dev" in item:
        # Move to end for LRU tracking
        GATE_CACHE.move_to_end(key)
        return item["U_dev"]

    # Build from cached host (ensures stability)
    np_dtype = np.complex128 if dtype == cp.complex128 else np.complex64
    U_host = get_gate_matrix_host(op, dtype=np_dtype)
    U_dev = cp.asarray(U_host, dtype=dtype, order="C")

    if item is None:
        _evict_cache_if_needed()
        item = {}
        GATE_CACHE[key] = item
    item["U_dev"] = U_dev
    return U_dev


def clear_gate_cache():
    """Manually clear the gate cache and free GPU memory."""
    global GATE_CACHE
    for item in GATE_CACHE.values():
        if "U_dev" in item:
            del item["U_dev"]
    GATE_CACHE.clear()
    cp.get_default_memory_pool().free_all_blocks()

def permute_statevector(psi: cp.ndarray, layout_old, layout_new):
    n = len(layout_old)
    dim = 1 << n

    old_pos = {q: p for p, q in enumerate(layout_old)}
    new_pos = {q: p for p, q in enumerate(layout_new)}

    map_old_to_new = cp.asarray(
        [new_pos[layout_old[p]] for p in range(n)],
        dtype=cp.int32
    )

    idx_old = cp.arange(dim, dtype=cp.uint32)
    idx_new = cp.zeros_like(idx_old)

    for p_old in range(n):
        bit = (idx_old >> p_old) & 1
        idx_new |= (bit << map_old_to_new[p_old])

    psi_new = psi[idx_new]
    return psi_new



# Try to import the native CUDA module (pybind11-based)
try:
    from .native import qgpusim_cuda
    USE_NATIVE = True
    print("Using native pybind11 CUDA backend")
except ImportError as e:
    USE_NATIVE = False
    print(f"Native CUDA module not available: {e}")
    print("Falling back to CuPy-based implementation")


# Unified gate matrix access - use get_gate_matrix_host() or get_gate_matrix_dev()


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
    num_qubits: int
):
    """Fallback 1-qubit gate using pure CuPy (no custom kernels).
    
    Memory-optimized: computes indices on-demand without pre-allocation.
    """
    dim = psi.shape[0]
    assert dim == (1 << num_qubits)
    mask = 1 << q

    # Compute base indices where bit q is 0 (half the state space)
    # This avoids allocating a full dim-sized index array
    half_dim = dim >> 1
    base = cp.arange(half_dim, dtype=cp.int64)
    # Insert 0 at bit position q
    i = ((base >> q) << (q + 1)) | (base & ((1 << q) - 1))
    j = i | mask

    a0 = psi[i]
    a1 = psi[j]

    psi[i] = U[0, 0] * a0 + U[0, 1] * a1
    psi[j] = U[1, 0] * a0 + U[1, 1] * a1
    
    # Free intermediate arrays
    del base, i, j, a0, a1


def apply_2q_gate_cupy_fallback(
    psi: cp.ndarray,
    U: cp.ndarray,
    q0: int,
    q1: int,
    num_qubits: int
):
    """Fallback 2-qubit gate using pure CuPy (no custom kernels).
    
    Memory-optimized: computes indices on-demand, uses in-place math
    to avoid intermediate cp.stack() allocations.
    """
    dim = psi.shape[0]
    assert dim == (1 << num_qubits)
    if q0 == q1:
        raise ValueError("q0 and q1 must be different")

    low, high = sorted((q0, q1))
    mask_low = 1 << low
    mask_high = 1 << high

    # Compute base indices where both bits are 0 (1/4 of state space)
    quarter_dim = dim >> 2
    k = cp.arange(quarter_dim, dtype=cp.int64)
    
    # Insert 0s at bit positions low and high
    # Split k into three parts: below low, between low and high, above high
    below_low = k & ((1 << low) - 1)
    between = (k >> low) & ((1 << (high - low - 1)) - 1)
    above_high = k >> (high - 1)
    
    base = below_low | (between << (low + 1)) | (above_high << (high + 1))
    del k, below_low, between, above_high  # Free immediately

    i00 = base
    i01 = base | mask_low
    i10 = base | mask_high
    i11 = base | mask_low | mask_high

    # Load values
    v00 = psi[i00]
    v01 = psi[i01]
    v10 = psi[i10]
    v11 = psi[i11]

    # In-place matrix multiplication (avoids cp.stack allocation)
    psi[i00] = U[0, 0] * v00 + U[0, 1] * v01 + U[0, 2] * v10 + U[0, 3] * v11
    psi[i01] = U[1, 0] * v00 + U[1, 1] * v01 + U[1, 2] * v10 + U[1, 3] * v11
    psi[i10] = U[2, 0] * v00 + U[2, 1] * v01 + U[2, 2] * v10 + U[2, 3] * v11
    psi[i11] = U[3, 0] * v00 + U[3, 1] * v01 + U[3, 2] * v10 + U[3, 3] * v11
    
    # Free intermediate arrays
    del base, i00, i01, i10, i11, v00, v01, v10, v11


def run_custom_backend(qc: QuantumCircuit, tile_plan: List[Tile], task: str, shots: int=None, use_float32: bool=False):
    """Run quantum circuit simulation on custom GPU backend.
    
    Args:
        qc: The quantum circuit to simulate
        tile_plan: List of Tile objects containing gates to execute
        task: Task description string
        shots: Number of measurement shots (optional)
        use_float32: If True, use complex64 (float32) for ~2x memory savings.
                     Default False uses complex128 (float64) for higher precision.
    """
    print("Running custom GPU backend")
    num_qubits = qc.num_qubits
    
    # Select precision
    cp_dtype = cp.complex64 if use_float32 else cp.complex128
    np_dtype = np.complex64 if use_float32 else np.complex128
    print(f"Precision: {'complex64' if use_float32 else 'complex128'}")

    # Pre-build qubit index map (avoid repeated find_bit calls)
    qubit_map = {q: qc.find_bit(q).index for q in qc.qubits}

    # Use fully native Statevector class (gate-by-gate execution)
    if USE_NATIVE:
        sv = qgpusim_cuda.Statevector(num_qubits)
        print("Mode: Native pybind11 CUDA")
        
        layout = list(range(num_qubits))

        for tile in tile_plan:
            if tile.layout != layout:
                sv.permute(layout, tile.layout)
                layout = tile.layout

            for node in tile.gates:
                op = node.op
                if not hasattr(op, "to_matrix"):
                    continue

                U_dev = get_gate_matrix_dev(op, dtype=cp_dtype)
                U_host = get_gate_matrix_host(op, dtype=np_dtype)

                global_qubits = [qubit_map[q] for q in node.qargs]
                local_qubits = [tile.global_to_local_map[gq] for gq in global_qubits]

                if U_host.shape == (2, 2) and len(local_qubits) == 1:
                    sv.apply_1q_dev(local_qubits[0], U_dev)
                elif U_host.shape == (4, 4) and len(local_qubits) == 2:
                    sv.apply_2q_dev(local_qubits[0], local_qubits[1], U_dev)





        # for tile in tile_plan:
        #     for node in tile.gates:
        #         op = node.op

        #         if not hasattr(op, "to_matrix"):
        #             continue

        #         try:
        #             U_dev = get_gate_matrix_dev(op, dtype=cp_dtype)
        #             U_host = get_gate_matrix_host(op, dtype=np_dtype)
        #         except Exception:
        #             continue

        #         qargs = node.qargs
        #         global_qubits = [qubit_map[q] for q in qargs]

        #         if U_host.shape == (2, 2) and len(global_qubits) == 1:
        #             sv.apply_1q_dev(global_qubits[0], U_dev)
        #         elif U_host.shape == (4, 4) and len(global_qubits) == 2:
        #             sv.apply_2q_dev(global_qubits[0], global_qubits[1], U_dev)

        print("Finished circuit execution using native CUDA module")
        result = sv.to_numpy()
        # Free GPU memory from statevector
        del sv
        cp.get_default_memory_pool().free_all_blocks()
        return result

    # Fallback: Use CuPy arrays with native kernel functions (if available) or pure CuPy
    print('CuPy-based fallback backend')
    print("Mode: CuPy-based fallback")
    dim = 1 << num_qubits
    psi_gpu = cp.zeros(dim, dtype=cp_dtype)
    psi_gpu[0] = 1.0 + 0.0j
    layout = list(range(num_qubits))  # current layout: bitpos -> global qubit


    print(f"task: {task}, shots: {shots}, num_qubits: {num_qubits}, num_tiles: {len(tile_plan)}")

    # changed this up
    for tile in tile_plan:
        # ---- tile boundary "QR" (single-GPU = statevector permutation) ----
        if tile.layout != layout:
            psi_gpu = permute_statevector(psi_gpu, layout, tile.layout)
            layout = tile.layout

        # ---- narrow-access execution using tile-local indices ----
        for node in tile.gates:
            op = node.op

            if not hasattr(op, "to_matrix"):
                print(f"[skip] non-unitary op: {op.name}")
                continue

            try:
                U_gpu = get_gate_matrix_dev(op, dtype=cp_dtype)
            except Exception:
                print(f"[skip] op {op.name} has no matrix representation")
                continue

            global_qubits = [qubit_map[q] for q in node.qargs]

            # Convert global qubit ids -> tile-local bit positions (0..|QL|-1)
            try:
                local_qubits = [tile.global_to_local_map[gq] for gq in global_qubits]
            except KeyError:
                raise RuntimeError(
                    f"Gate targets {global_qubits} not in tile QL={tile.logical_qubits} "
                    f"(tile_idx={tile.tile_idx})"
                )

            if U_gpu.shape == (2, 2) and len(local_qubits) == 1:
                apply_1q_gate_cupy_fallback(psi_gpu, U_gpu, local_qubits[0], num_qubits)
            elif U_gpu.shape == (4, 4) and len(local_qubits) == 2:
                apply_2q_gate_cupy_fallback(psi_gpu, U_gpu, local_qubits[0], local_qubits[1], num_qubits)
            else:
                print(
                    f"[skip] unsupported gate {op.name} "
                    f"shape={U_gpu.shape} on {len(local_qubits)} qubits"
                )
                continue

    
    psi_cpu = cp.asnumpy(psi_gpu)
    
    # Explicit GPU memory cleanup
    del psi_gpu
    cp.get_default_memory_pool().free_all_blocks()
    
    return psi_cpu
