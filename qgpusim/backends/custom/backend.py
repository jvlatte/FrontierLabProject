from platform import node
from qiskit import QuantumCircuit
from qgpusim.transpiler.passes import Tile
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict
import numpy as np
import cupy as cp
import time

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


def fuse_tile_ops(tile_ops, cp_dtype=cp.complex128):
    """
    Fuse consecutive gates on the same qubit(s) to reduce memory passes.
    
    This performs a two-pass fusion:
    Pass 1: Fuse consecutive 1Q gates on the same qubit
    Pass 2: Fuse consecutive 2Q gates on the same qubit pair
    
    The fusion is conservative - we only fuse when gates are truly consecutive
    and have no intervening gates on any of their qubits.
    
    Returns a new list of (potentially fewer) tile_ops with fused matrices.
    """
    if len(tile_ops) <= 1:
        return tile_ops
    
    # Build dependency graph approach:
    # We can fuse gate G2 with G1 if:
    # 1. They act on the same qubit(s)
    # 2. There is no gate between G1 and G2 that acts on any of those qubits
    
    # Simpler: Track last gate index per qubit and attempt fusion
    # qubit -> (index_in_result, is_fused)
    # When we see a new gate, check if we can merge with the last gate on same qubit(s)
    
    result = []
    # Track index in result of last gate touching each qubit
    last_gate_idx = {}  # qubit -> index in result
    
    fuse_count = 0
    
    for op in tile_ops:
        if op[0] == "1q":
            q, U = op[1], op[2]
            
            # Can we fuse with the last gate on this qubit?
            if q in last_gate_idx:
                idx = last_gate_idx[q]
                prev_op = result[idx]
                
                # Check if the previous op is also a 1Q gate on the same qubit
                # AND no other gate has touched this qubit since then
                if prev_op[0] == "1q" and prev_op[1] == q:
                    # Check that no intervening gates touched this qubit
                    can_fuse = True
                    for i in range(idx + 1, len(result)):
                        other = result[i]
                        if other[0] == "1q" and other[1] == q:
                            can_fuse = False
                            break
                        if other[0] == "2q" and (other[1] == q or other[2] == q):
                            can_fuse = False
                            break
                    
                    if can_fuse:
                        # Fuse: new_U = U @ prev_U (apply prev first, then current)
                        fused_U = cp.matmul(U, prev_op[2])
                        result[idx] = ("1q", q, fused_U)
                        last_gate_idx[q] = idx
                        fuse_count += 1
                        continue
            
            # Cannot fuse, add as new
            last_gate_idx[q] = len(result)
            result.append(("1q", q, U))
            
        elif op[0] == "2q":
            q0, q1, U = op[1], op[2], op[3]
            
            # For 2Q gates, check if we can fuse with another 2Q on same pair
            # Only fuse if exact same qubit pair in same order (q0, q1)
            key = (q0, q1)
            
            # Find last 2Q gate on this exact pair
            prev_idx = None
            for i in range(len(result) - 1, -1, -1):
                prev_op = result[i]
                if prev_op[0] == "2q" and prev_op[1] == q0 and prev_op[2] == q1:
                    prev_idx = i
                    break
                # If any gate touches q0 or q1, we can't fuse past it
                if prev_op[0] == "1q" and prev_op[1] in (q0, q1):
                    break
                if prev_op[0] == "2q" and (prev_op[1] in (q0, q1) or prev_op[2] in (q0, q1)):
                    break
            
            if prev_idx is not None:
                # Check no intervening gates touch q0 or q1
                can_fuse = True
                for i in range(prev_idx + 1, len(result)):
                    other = result[i]
                    if other[0] == "1q" and other[1] in (q0, q1):
                        can_fuse = False
                        break
                    if other[0] == "2q" and (other[1] in (q0, q1) or other[2] in (q0, q1)):
                        can_fuse = False
                        break
                
                if can_fuse:
                    prev_op = result[prev_idx]
                    fused_U = cp.matmul(U, prev_op[3])
                    result[prev_idx] = ("2q", q0, q1, fused_U)
                    last_gate_idx[q0] = prev_idx
                    last_gate_idx[q1] = prev_idx
                    fuse_count += 1
                    continue
            
            # Cannot fuse, add as new
            last_gate_idx[q0] = len(result)
            last_gate_idx[q1] = len(result)
            result.append(("2q", q0, q1, U))
    
    return result

def permute_statevector(psi: cp.ndarray, layout_old, layout_new):
    n = len(layout_old)
    dim = 1 << n

    old_pos = {q: p for p, q in enumerate(layout_old)}
    new_pos = {q: p for p, q in enumerate(layout_new)}

    # Build mapping on CPU (small array)
    map_old_to_new_list = [new_pos[layout_old[p]] for p in range(n)]

    idx_old = cp.arange(dim, dtype=cp.uint64)
    idx_new = cp.zeros(dim, dtype=cp.uint64)

    for p_old in range(n):
        p_new = map_old_to_new_list[p_old]  # Python int
        bit = (idx_old >> p_old) & 1
        idx_new |= (bit << p_new)

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


def run_custom_backend(qc: QuantumCircuit, tile_plan: List[Tile], task: str, shots: int=None, use_float32: bool=False, use_cuda_graphs: bool=True, tile_mode: str="cooperative", enable_fusion: bool=True):
    """Run quantum circuit simulation on custom GPU backend.
    
    Args:
        qc: The quantum circuit to simulate
        tile_plan: List of Tile objects containing gates to execute
        task: Task description string
        shots: Number of measurement shots (optional)
        use_float32: If True, use complex64 (float32) for ~2x memory savings.
                     Default False uses complex128 (float64) for higher precision.
        use_cuda_graphs: If True, use CUDA Graphs (only if tile_mode is "graphed").
                         This reduces kernel launch overhead. Default True.
        tile_mode: Tile execution mode:
                   - "cooperative": TRUE tile kernel using cooperative launch (single kernel per tile)
                   - "graphed": CUDA Graphs to capture/replay kernel launches
                   - "sequential": Gate-by-gate kernel launches on a stream
        enable_fusion: If True, fuse consecutive gates on same qubit(s) to reduce memory passes.
        
    Returns:
        tuple: (result_statevector, metrics_dict) where metrics_dict contains:
            - num_tiles: Number of tiles executed
            - num_gates: Total number of gates
            - num_fused_ops: Number of operations after fusion
            - num_reorders: Number of qubit reorderings
            - num_kernel_launches: Total kernel launches (0 for graphed mode per tile)
            - num_graph_replays: Number of CUDA graph replays (graphed mode only)
            - t_perm: Time spent on permutations
            - t_apply: Time spent applying gates (CPU-side)
            - t_sync: Time spent on GPU synchronization
            - t_total: Total execution time
            - t_launch_overhead_est: Estimated kernel launch overhead saved (graphed mode)
            - tile_mode: The execution mode used
    """
    print("Running custom GPU backend")
    num_qubits = qc.num_qubits
    
    # Select precision
    cp_dtype = cp.complex64 if use_float32 else cp.complex128
    np_dtype = np.complex64 if use_float32 else np.complex128
    print(f"Precision: {'complex64' if use_float32 else 'complex128'}")
    
    mode_desc = {
        "cooperative": "Cooperative Kernel (TRUE tile-by-tile)",
        "graphed": "CUDA Graphs (reduced launch overhead)",
        "sequential": "Sequential (gate-by-gate)"
    }
    print(f"Tile execution mode: {mode_desc.get(tile_mode, tile_mode)}")
    print(f"Gate fusion: {'enabled' if enable_fusion else 'disabled'}")

    # Pre-build qubit index map (avoid repeated find_bit calls)
    qubit_map = {q: qc.find_bit(q).index for q in qc.qubits}

    # extra metrics
    t_start = time.perf_counter()

    num_tiles = len(tile_plan)
    num_reorders = 0
    num_gates = 0
    num_fused_ops = 0  # track fused gate count
    
    # Kernel launch overhead tracking
    num_kernel_launches = 0  # actual kernel launches (for sequential/cooperative)
    num_graph_replays = 0    # graph replays (for graphed mode)
    KERNEL_LAUNCH_OVERHEAD_US = 10.0  # estimated ~10μs per kernel launch

    t_perm = 0.0
    t_apply = 0.0


    # Use fully native Statevector class (gate-by-gate execution)
    # Choose StatevectorF32 for float32, Statevector for float64
    if USE_NATIVE:
        if use_float32:
            sv = qgpusim_cuda.StatevectorF32(num_qubits)
            print("Mode: Native pybind11 CUDA (float32)")
        else:
            sv = qgpusim_cuda.Statevector(num_qubits)
            print("Mode: Native pybind11 CUDA")
        layout = list(range(num_qubits))

        for tile in tile_plan:
            if tile.layout != layout:
                t0 = time.perf_counter()
                
                # added this debug print
                free, total = cp.cuda.runtime.memGetInfo()
                print("[before permute] free GB:", free/1e9)

                sv.permute(layout, tile.layout)
                # No sync needed - permute is on same stream as gates
                t_perm += time.perf_counter() - t0
                num_reorders += 1
                layout = tile.layout

            t0 = time.perf_counter()
            tile_ops = []
            for node in tile.gates:
                op = node.op
                if not hasattr(op, "to_matrix"):
                    continue
                num_gates += 1

                # U_dev = get_gate_matrix_dev(op, dtype=cp_dtype)
                # U_host = get_gate_matrix_host(op, dtype=np_dtype)

                U_dev = get_gate_matrix_dev(op, dtype=cp_dtype)
                k = len(node.qargs)


                global_qubits = [qubit_map[q] for q in node.qargs]
                local_qubits = [tile.global_to_local_map[gq] for gq in global_qubits]

                if k == 1:
                    tile_ops.append(("1q", local_qubits[0], U_dev))
                elif k == 2:
                    tile_ops.append(("2q", local_qubits[0], local_qubits[1], U_dev))

                # if U_host.shape == (2, 2) and len(local_qubits) == 1:
                #     # sv.apply_1q_dev(local_qubits[0], U_dev)
                #     tile_ops.append(("1q", local_qubits[0], U_dev))
                # elif U_host.shape == (4, 4) and len(local_qubits) == 2:
                #     # sv.apply_2q_dev(local_qubits[0], local_qubits[1], U_dev)
                #     tile_ops.append(("2q", local_qubits[0], local_qubits[1], U_dev))

            if tile_ops:
                # Apply gate fusion if enabled
                if enable_fusion:
                    tile_ops = fuse_tile_ops(tile_ops, cp_dtype)
                num_fused_ops += len(tile_ops)
                
                # Select tile execution method based on tile_mode
                if tile_mode == "cooperative":
                    # TRUE tile kernel: single cooperative kernel for ALL gates in tile
                    sv.apply_tile_cooperative(tile_ops)
                    num_kernel_launches += 1  # single kernel for entire tile
                elif tile_mode == "graphed":
                    # CUDA Graphs: capture kernel launches, replay as single graph
                    sv.apply_tile_graphed(tile_ops)
                    num_graph_replays += 1  # single graph replay for entire tile
                    # Note: individual kernel launches happen during capture (first run)
                    # but subsequent replays have minimal CPU overhead
                else:
                    # Sequential: gate-by-gate kernel launches
                    sv.apply_tile_dev(tile_ops)
                    num_kernel_launches += len(tile_ops)  # one launch per gate
                # Note: no sync needed here - operations are ordered on the stream
            t_apply += time.perf_counter() - t0

        # Single sync at the end before reading results
        t_sync_start = time.perf_counter()
        sv.synchronize()
        t_sync = time.perf_counter() - t_sync_start
        print("Finished circuit execution using native CUDA module")
        result = sv.to_numpy()
        # Free GPU memory from statevector
        del sv
        cp.get_default_memory_pool().free_all_blocks()

        t_total = time.perf_counter() - t_start
        avg_tile_size = num_gates / max(1, num_tiles)
        fusion_ratio = num_gates / max(1, num_fused_ops) if enable_fusion else 1.0
        
        # Estimate kernel launch overhead savings
        # Sequential mode would launch num_fused_ops kernels
        # Graphed mode replays num_tiles graphs (1 per tile) instead
        if tile_mode == "graphed":
            avoided_launches = num_fused_ops - num_graph_replays
            t_launch_overhead_saved_us = avoided_launches * KERNEL_LAUNCH_OVERHEAD_US
        elif tile_mode == "cooperative":
            avoided_launches = num_fused_ops - num_tiles
            t_launch_overhead_saved_us = avoided_launches * KERNEL_LAUNCH_OVERHEAD_US
        else:
            avoided_launches = 0
            t_launch_overhead_saved_us = 0.0

        print("==== TILING METRICS ====")
        print(f"num_qubits: {num_qubits}")
        print(f"num_tiles: {num_tiles}")
        print(f"num_gates: {num_gates}")
        if enable_fusion:
            print(f"num_fused_ops: {num_fused_ops} (fusion ratio: {fusion_ratio:.2f}x)")
        print(f"avg_tile_size: {avg_tile_size:.2f}")
        print(f"num_reorders: {num_reorders}")
        print(f"t_perm (CPU):  {t_perm:.6f} s")
        print(f"t_apply (CPU): {t_apply:.6f} s")
        print(f"t_sync (GPU):  {t_sync:.6f} s ({(t_sync/t_total*100):.1f}%)")
        print(f"t_total: {t_total:.6f} s")
        
        print("==== KERNEL LAUNCH OVERHEAD ====")
        print(f"tile_mode: {tile_mode}")
        print(f"num_kernel_launches: {num_kernel_launches}")
        print(f"num_graph_replays: {num_graph_replays}")
        if tile_mode in ("graphed", "cooperative"):
            print(f"avoided_launches: {avoided_launches} (vs sequential)")
            print(f"est_overhead_saved: {t_launch_overhead_saved_us:.1f} μs ({t_launch_overhead_saved_us/1000:.3f} ms)")
        
        # Build metrics dictionary for benchmark tracking
        metrics = {
            "num_tiles": num_tiles,
            "num_gates": num_gates,
            "num_fused_ops": num_fused_ops,
            "fusion_ratio": fusion_ratio,
            "num_reorders": num_reorders,
            "num_kernel_launches": num_kernel_launches,
            "num_graph_replays": num_graph_replays,
            "avoided_launches": avoided_launches,
            "t_perm_s": t_perm,
            "t_apply_s": t_apply,
            "t_sync_s": t_sync,
            "t_total_s": t_total,
            "t_launch_overhead_saved_us": t_launch_overhead_saved_us,
            "tile_mode": tile_mode,
        }

        return result, metrics

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
