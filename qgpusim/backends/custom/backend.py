from qiskit import QuantumCircuit
from qgpusim.transpiler.passes import Tile
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict
import numpy as np
import cupy as cp
import logging

# Configure module logger with NullHandler (silent by default)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def set_log_level(level: int = logging.INFO):
    """Enable logging output at the specified level.
    
    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING)
    
    Example:
        import logging
        from qgpusim.backends.custom import backend
        backend.set_log_level(logging.DEBUG)
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[qgpusim] %(levelname)s: %(message)s'))
    logger.handlers = [handler]
    logger.setLevel(level)


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


# Try to import the native CUDA module (pybind11-based)
try:
    from .native import qgpusim_cuda
    USE_NATIVE = True
    logger.info("Using native pybind11 CUDA backend")
except ImportError as e:
    USE_NATIVE = False
    logger.warning(f"Native CUDA module not available: {e}")
    logger.info("Falling back to CuPy-based implementation")


# Unified gate matrix access - use get_gate_matrix_host() or get_gate_matrix_dev()


# =============================================================================
# Gate Fusion Utilities
# =============================================================================

def _is_diagonal_gate(op) -> bool:
    """Check if a gate is diagonal (only modifies phases)."""
    return op.name in {'rz', 'p', 's', 't', 'sdg', 'tdg', 'z', 'u1'}


def _fuse_1q_gates(gates: List[Tuple], qubit: int, np_dtype) -> Optional[np.ndarray]:
    """Fuse consecutive 1-qubit gates on the same qubit by matrix multiplication.
    
    Args:
        gates: List of (op, qargs) tuples to fuse
        qubit: The target qubit
        np_dtype: NumPy dtype for the result
        
    Returns:
        Fused 2x2 unitary matrix, or None if fusion not possible
    """
    if len(gates) == 0:
        return None
    if len(gates) == 1:
        return get_gate_matrix_host(gates[0][0], dtype=np_dtype)
    
    # Multiply matrices in reverse order (last gate first in matrix product)
    fused = np.eye(2, dtype=np_dtype)
    for op, _ in gates:
        try:
            U = get_gate_matrix_host(op, dtype=np_dtype)
            fused = U @ fused
        except Exception:
            return None
    return fused


def _optimize_gate_sequence(tile_plan: List, qc: QuantumCircuit, qubit_map: Dict) -> List[Tuple]:
    """Optimize gate sequence by fusing consecutive 1-qubit gates on the same qubit.
    
    Returns list of (op_or_matrix, global_qubits, is_fused) tuples.
    """
    optimized = []
    pending_1q = {}  # qubit -> [(op, qargs), ...]
    
    def flush_pending(qubit: Optional[int] = None):
        """Flush pending 1-qubit gates for a qubit (or all if qubit is None)."""
        qubits_to_flush = [qubit] if qubit is not None else list(pending_1q.keys())
        for q in qubits_to_flush:
            if q in pending_1q and pending_1q[q]:
                gates = pending_1q.pop(q)
                if len(gates) == 1:
                    # Single gate, no fusion needed
                    op, qargs = gates[0]
                    optimized.append((op, [q], False))
                else:
                    # Multiple gates fused
                    optimized.append((gates, [q], True))
    
    for tile in tile_plan:
        for node in tile.gates:
            op = node.op
            if not hasattr(op, "to_matrix"):
                continue
                
            qargs = node.qargs
            global_qubits = [qubit_map[q] for q in qargs]
            
            if len(global_qubits) == 1:
                # 1-qubit gate: accumulate for potential fusion
                qubit = global_qubits[0]
                if qubit not in pending_1q:
                    pending_1q[qubit] = []
                pending_1q[qubit].append((op, qargs))
            else:
                # Multi-qubit gate: flush any pending 1Q gates on involved qubits
                for q in global_qubits:
                    flush_pending(q)
                optimized.append((op, global_qubits, False))
    
    # Flush any remaining pending gates
    flush_pending()
    
    return optimized


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


def run_custom_backend(qc: QuantumCircuit, tile_plan: List[Tile], task: str, shots: int=None, use_float32: bool=False, enable_fusion: bool=True):
    """Run quantum circuit simulation on custom GPU backend.
    
    Args:
        qc: The quantum circuit to simulate
        tile_plan: List of Tile objects containing gates to execute
        task: Task description string
        shots: Number of measurement shots (optional)
        use_float32: If True, use complex64 (float32) for ~2x memory savings.
                     Default False uses complex128 (float64) for higher precision.
        enable_fusion: If True, fuse consecutive 1-qubit gates on same qubit (default True).
    """
    logger.info("Running custom GPU backend")
    num_qubits = qc.num_qubits
    
    # Select precision
    cp_dtype = cp.complex64 if use_float32 else cp.complex128
    np_dtype = np.complex64 if use_float32 else np.complex128
    logger.debug(f"Precision: {'complex64' if use_float32 else 'complex128'}")

    # Pre-build qubit index map (avoid repeated find_bit calls)
    qubit_map = {q: qc.find_bit(q).index for q in qc.qubits}
    
    # Optimize gate sequence with fusion if enabled
    if enable_fusion:
        optimized_gates = _optimize_gate_sequence(tile_plan, qc, qubit_map)
        total_original = sum(len(t.gates) for t in tile_plan)
        logger.debug(f"Gate fusion: {total_original} -> {len(optimized_gates)} operations")
    else:
        optimized_gates = None

    # Use fully native Statevector class (gate-by-gate execution)
    if USE_NATIVE:
        sv = qgpusim_cuda.Statevector(num_qubits)
        logger.debug("Mode: Native pybind11 CUDA")
        print("Mode: Native pybind11 CUDA")
        
        if enable_fusion and optimized_gates is not None:
            # Execute optimized gate sequence
            for item, global_qubits, is_fused in optimized_gates:
                if is_fused:
                    # Fused gates: multiply matrices and apply once
                    fused_matrix = _fuse_1q_gates(item, global_qubits[0], np_dtype)
                    if fused_matrix is not None:
                        U_dev = cp.asarray(fused_matrix, dtype=cp_dtype, order="C")
                        sv.apply_1q_dev(global_qubits[0], U_dev)
                else:
                    # Single gate
                    op = item
                    try:
                        U_dev = get_gate_matrix_dev(op, dtype=cp_dtype)
                        U_host = get_gate_matrix_host(op, dtype=np_dtype)
                    except Exception:
                        continue

                    if U_host.shape == (2, 2) and len(global_qubits) == 1:
                        sv.apply_1q_dev(global_qubits[0], U_dev)
                    elif U_host.shape == (4, 4) and len(global_qubits) == 2:
                        sv.apply_2q_dev(global_qubits[0], global_qubits[1], U_dev)
        else:
            # Original non-fused execution
            for tile in tile_plan:
                for node in tile.gates:
                    op = node.op

                    if not hasattr(op, "to_matrix"):
                        continue

                    try:
                        U_dev = get_gate_matrix_dev(op, dtype=cp_dtype)
                        U_host = get_gate_matrix_host(op, dtype=np_dtype)
                    except Exception:
                        continue

                    qargs = node.qargs
                    global_qubits = [qubit_map[q] for q in qargs]

                    if U_host.shape == (2, 2) and len(global_qubits) == 1:
                        sv.apply_1q_dev(global_qubits[0], U_dev)
                    elif U_host.shape == (4, 4) and len(global_qubits) == 2:
                        sv.apply_2q_dev(global_qubits[0], global_qubits[1], U_dev)

        logger.debug("Finished circuit execution using native CUDA module")
        print("Finished circuit execution using native CUDA module")
        result = sv.to_numpy()
        # Free GPU memory from statevector
        del sv
        cp.get_default_memory_pool().free_all_blocks()
        return result

    # Fallback: Use CuPy arrays with native kernel functions (if available) or pure CuPy
    print('CuPy-based fallback backend')
    logger.debug("Mode: CuPy-based fallback")
    dim = 1 << num_qubits
    psi_gpu = cp.zeros(dim, dtype=cp_dtype)
    psi_gpu[0] = 1.0 + 0.0j

    logger.debug(f"task: {task}, shots: {shots}, num_qubits: {num_qubits}, num_tiles: {len(tile_plan)}")

    if enable_fusion and optimized_gates is not None:
        # Execute optimized gate sequence
        for item, global_qubits, is_fused in optimized_gates:
            if is_fused:
                fused_matrix = _fuse_1q_gates(item, global_qubits[0], np_dtype)
                if fused_matrix is not None:
                    U_gpu = cp.asarray(fused_matrix, dtype=cp_dtype, order="C")
                    apply_1q_gate_cupy_fallback(psi_gpu, U_gpu, global_qubits[0], num_qubits)
            else:
                op = item
                try:
                    U_gpu = get_gate_matrix_dev(op, dtype=cp_dtype)
                except Exception:
                    logger.debug(f"[skip] op {op.name} has no matrix representation")
                    continue

                if U_gpu.shape == (2, 2) and len(global_qubits) == 1:
                    apply_1q_gate_cupy_fallback(psi_gpu, U_gpu, global_qubits[0], num_qubits)
                elif U_gpu.shape == (4, 4) and len(global_qubits) == 2:
                    apply_2q_gate_cupy_fallback(psi_gpu, U_gpu, global_qubits[0], global_qubits[1], num_qubits)
    else:
        for tile in tile_plan:
            for node in tile.gates:
                op = node.op

                if not hasattr(op, "to_matrix"):
                    logger.debug(f"[skip] non-unitary op: {op.name}")
                    continue

                try:
                    U_gpu = get_gate_matrix_dev(op, dtype=cp_dtype)
                except Exception:
                    logger.debug(f"[skip] op {op.name} has no matrix representation")
                    continue

                qargs = node.qargs
                global_qubits = [qubit_map[q] for q in qargs]

                if U_gpu.shape == (2, 2) and len(global_qubits) == 1:
                    apply_1q_gate_cupy_fallback(psi_gpu, U_gpu, global_qubits[0], num_qubits)
                elif U_gpu.shape == (4, 4) and len(global_qubits) == 2:
                    apply_2q_gate_cupy_fallback(psi_gpu, U_gpu, global_qubits[0], global_qubits[1], num_qubits)
                else:
                    logger.debug(
                        f"[skip] unsupported gate {op.name} "
                        f"shape={U_gpu.shape} on {len(global_qubits)} qubits"
                    )
                    continue
    
    psi_cpu = cp.asnumpy(psi_gpu)
    
    # Explicit GPU memory cleanup
    del psi_gpu
    cp.get_default_memory_pool().free_all_blocks()
    
    return psi_cpu
