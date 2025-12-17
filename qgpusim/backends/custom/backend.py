from qiskit import QuantumCircuit
from qgpusim.transpiler.passes import Tile
from typing import List
import numpy as np
import cupy as cp

# cache dict
GATE_CACHE = {} # (name, parameters) -> cp matrix

_apply_1q_gate_src = r"""
#include <cuComplex.h>

extern "C" __global__
void apply_1q_gate_kernel(
    cuDoubleComplex* psi,        // statevector, length 2^n
    const cuDoubleComplex* U,    // 2x2 unitary, flattened (row-major) length 4
    const long long n,           // num_qubits
    const int q                  // target qubit index
){
    unsigned long long dim = 1ULL << n;
    unsigned long long stride = 1ULL << q;
    unsigned long long num_pairs = dim >> 1;   // number of (i0,i1) pairs

    unsigned long long k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= num_pairs) return;

    // Map k -> (i0, i1) such that bit q of i0 is 0 and i1 = i0 + stride.
    unsigned long long block  = k / stride;
    unsigned long long offset = k % stride;

    unsigned long long i0 = block * 2ULL * stride + offset;
    unsigned long long i1 = i0 + stride;

    cuDoubleComplex a0 = psi[i0];
    cuDoubleComplex a1 = psi[i1];

    cuDoubleComplex u00 = U[0];
    cuDoubleComplex u01 = U[1];
    cuDoubleComplex u10 = U[2];
    cuDoubleComplex u11 = U[3];

    cuDoubleComplex out0, out1;
    out0 = cuCadd(cuCmul(u00, a0), cuCmul(u01, a1));
    out1 = cuCadd(cuCmul(u10, a0), cuCmul(u11, a1));

    psi[i0] = out0;
    psi[i1] = out1;
}
""";

apply_1q_gate_kernel = cp.RawKernel(
    _apply_1q_gate_src,
    "apply_1q_gate_kernel",
)

_apply_2q_gate_src = r"""
#include <cuComplex.h>

extern "C" __global__
void apply_2q_gate_kernel(
    cuDoubleComplex* psi,        // statevector, length 2^n
    const cuDoubleComplex* U,    // 4x4 unitary, flattened row-major (length 16)
    const long long n,           // num_qubits
    const int q0,                // first qubit
    const int q1                 // second qubit
){
    // Ensure low < high
    int low  = q0 < q1 ? q0 : q1;
    int high = q0 < q1 ? q1 : q0;

    unsigned long long dim = 1ULL << n;
    unsigned long long num_blocks = dim >> 2;  // each block handles 4 basis states

    unsigned long long k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= num_blocks) return;

    // k indexes over all assignments of the other (n-2) qubits.
    // We "insert" zero bits at positions low and high to build base index
    unsigned long long base = 0ULL;
    unsigned long long src  = k;

    for (int p = 0; p < n; ++p) {
        if (p == low || p == high) {
            // these bits are 0 in the base state |00>
            continue;
        }
        unsigned long long bit = (src & 1ULL);
        src >>= 1;
        base |= (bit << p);
    }

    unsigned long long mask_low  = 1ULL << low;
    unsigned long long mask_high = 1ULL << high;

    unsigned long long i00 = base;
    unsigned long long i01 = base | mask_low;
    unsigned long long i10 = base | mask_high;
    unsigned long long i11 = base | mask_low | mask_high;

    cuDoubleComplex a00 = psi[i00];
    cuDoubleComplex a01 = psi[i01];
    cuDoubleComplex a10 = psi[i10];
    cuDoubleComplex a11 = psi[i11];

    // U is 4x4 row-major: U[row*4 + col]
    cuDoubleComplex U00 = U[0];   cuDoubleComplex U01 = U[1];
    cuDoubleComplex U02 = U[2];   cuDoubleComplex U03 = U[3];
    cuDoubleComplex U10 = U[4];   cuDoubleComplex U11 = U[5];
    cuDoubleComplex U12 = U[6];   cuDoubleComplex U13 = U[7];
    cuDoubleComplex U20 = U[8];   cuDoubleComplex U21 = U[9];
    cuDoubleComplex U22 = U[10];  cuDoubleComplex U23 = U[11];
    cuDoubleComplex U30 = U[12];  cuDoubleComplex U31 = U[13];
    cuDoubleComplex U32 = U[14];  cuDoubleComplex U33 = U[15];

    cuDoubleComplex out0, out1, out2, out3;

    // out0 = sum_j U[0,j] * a_j
    out0 = cuCadd(cuCadd(cuCmul(U00, a00), cuCmul(U01, a01)),
                  cuCadd(cuCmul(U02, a10), cuCmul(U03, a11)));

    // out1 = sum_j U[1,j] * a_j
    out1 = cuCadd(cuCadd(cuCmul(U10, a00), cuCmul(U11, a01)),
                  cuCadd(cuCmul(U12, a10), cuCmul(U13, a11)));

    // out2 = sum_j U[2,j] * a_j
    out2 = cuCadd(cuCadd(cuCmul(U20, a00), cuCmul(U21, a01)),
                  cuCadd(cuCmul(U22, a10), cuCmul(U23, a11)));

    // out3 = sum_j U[3,j] * a_j
    out3 = cuCadd(cuCadd(cuCmul(U30, a00), cuCmul(U31, a01)),
                  cuCadd(cuCmul(U32, a10), cuCmul(U33, a11)));

    psi[i00] = out0;
    psi[i01] = out1;
    psi[i10] = out2;
    psi[i11] = out3;
}
""";

apply_2q_gate_kernel = cp.RawKernel(
    _apply_2q_gate_src,
    "apply_2q_gate_kernel",
)

try:

    from .native import qgpusim_cuda
    USE_NATIVE = True
except Exception:
    USE_NATIVE = False


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


# gpu functions
def apply_1q_gate_gpu_kernel(
    psi: cp.ndarray,
    U: cp.ndarray,
    q: int,
    num_qubits: int,
):
    dim = psi.size
    assert dim == (1 << num_qubits)

    # Number of (i0, i1) index pairs
    num_pairs = dim // 2

    threads_per_block = 256
    blocks = (num_pairs + threads_per_block - 1) // threads_per_block

    # Flatten U to length 4; ensure it's contiguous
    U_flat = U.ravel()

    apply_1q_gate_kernel(
        (blocks,),
        (threads_per_block,),
        (psi, U_flat, np.int64(num_qubits), np.int32(q)),
    )

def apply_2q_gate_gpu_kernel(
    psi: cp.ndarray,
    U: cp.ndarray,
    q0: int,
    q1: int,
    num_qubits: int,
):
    dim = psi.size
    assert dim == (1 << num_qubits)
    if q0 == q1:
        raise ValueError("q0 and q1 must be different")

    # each thread handles one group of 4 basis states => dim/4 groups
    num_blocks_total = dim // 4

    threads_per_block = 256
    blocks = (num_blocks_total + threads_per_block - 1) // threads_per_block

    # Flatten U to length 16; ensure contiguous
    U_flat = U.ravel()

    apply_2q_gate_kernel(
        (blocks,),
        (threads_per_block,),
        (psi, U_flat, np.int64(num_qubits), np.int32(q0), np.int32(q1)),
    )


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

    if USE_NATIVE:
        print("\n\nUsing native CUDA backend\n\n")
        sv = qgpusim_cuda.Statevector(num_qubits)

        for tile in tile_plan:
            for node in tile.gates:
                op = node.op

                if not hasattr(op, "to_matrix"):
                    continue

                try:
                    U = op.to_matrix()  # numpy array
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

        return sv.to_numpy()

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
                # apply_1q_gate_gpu(psi_gpu, U_gpu, global_qubits[0], num_qubits, idx)
                apply_1q_gate_gpu_kernel(psi_gpu, U_gpu, global_qubits[0], num_qubits)
            elif U_gpu.shape == (4, 4) and len(global_qubits) == 2:
                # apply_2q_gate_gpu(psi_gpu, U_gpu, global_qubits[0], global_qubits[1], num_qubits, idx)
                apply_2q_gate_gpu_kernel(psi_gpu, U_gpu, global_qubits[0], global_qubits[1], num_qubits)
            else:
                print(
                    f"      [skip] unsupported gate {op.name} "
                    f"shape={U_gpu.shape} on {len(global_qubits)} qubits"
                )
                continue
    psi_cpu = cp.asnumpy(psi_gpu)

    return psi_cpu
