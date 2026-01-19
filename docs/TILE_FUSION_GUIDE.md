# True Tile-by-Tile Execution Guide

Your current implementation launches a **separate CUDA kernel for each gate**, even though gates are grouped into tiles. To achieve true tile-by-tile execution, here are three approaches:

---

## Option 1: CUDA Graphs (Recommended - Best Balance)

**Concept**: Capture all kernel launches for a tile into a CUDA Graph, then execute the entire graph with a single API call. This eliminates kernel launch overhead (~5-10μs per launch).

### Changes Required:

#### 1. Add to `bindings.cpp` - New method in Statevector class:

```cpp
// Add these member variables to Statevector class:
cudaGraph_t cached_graph_ = nullptr;
cudaGraphExec_t cached_graph_exec_ = nullptr;
std::vector<TileOp> cached_ops_;

void apply_tile_graphed(py::list ops) {
    // Parse ops (same as before)
    const ssize_t nops = py::len(ops);
    std::vector<TileOp> parsed;
    parsed.reserve((size_t)nops);
    
    // ... parsing code same as apply_tile_dev ...
    
    // Check if we can reuse cached graph (same ops signature)
    bool can_reuse = (cached_graph_exec_ != nullptr) && 
                     (parsed.size() == cached_ops_.size());
    if (can_reuse) {
        for (size_t i = 0; i < parsed.size(); i++) {
            if (parsed[i].kind != cached_ops_[i].kind ||
                parsed[i].q0 != cached_ops_[i].q0 ||
                parsed[i].q1 != cached_ops_[i].q1) {
                can_reuse = false;
                break;
            }
        }
    }
    
    if (!can_reuse) {
        // Free old graph
        if (cached_graph_exec_) cudaGraphExecDestroy(cached_graph_exec_);
        if (cached_graph_) cudaGraphDestroy(cached_graph_);
        
        // Capture new graph
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        
        for (const auto& op : parsed) {
            if (op.kind == 1) {
                launch_apply_1q_gate(d_psi_, op.dU, num_qubits_, op.q0, stream);
            } else {
                launch_apply_2q_gate(d_psi_, op.dU, num_qubits_, op.q0, op.q1, stream);
            }
        }
        
        cudaStreamEndCapture(stream, &cached_graph_);
        cudaGraphInstantiate(&cached_graph_exec_, cached_graph_, NULL, NULL, 0);
        cudaStreamDestroy(stream);
        
        cached_ops_ = std::move(parsed);
    }
    
    // Execute graph (single launch for entire tile!)
    cudaGraphLaunch(cached_graph_exec_, 0);
}
```

**Speedup**: ~2-5x for tiles with many gates (eliminates launch overhead)

---

## Option 2: Gate Matrix Fusion (Best for Sequential Gates on Same Qubits)

**Concept**: Pre-multiply gate matrices that act on the same qubits sequentially.

### Python-side implementation in `backend.py`:

```python
def fuse_sequential_gates(tile_ops):
    """Fuse consecutive gates on same qubits via matrix multiplication."""
    if not tile_ops:
        return tile_ops
    
    fused = []
    i = 0
    while i < len(tile_ops):
        op = tile_ops[i]
        
        # Try to fuse with following ops on same qubits
        if op[0] == "1q":
            q = op[1]
            U_fused = op[2]
            j = i + 1
            while j < len(tile_ops) and tile_ops[j][0] == "1q" and tile_ops[j][1] == q:
                U_fused = cp.matmul(tile_ops[j][2], U_fused)  # Right-to-left composition
                j += 1
            fused.append(("1q", q, U_fused))
            i = j
        elif op[0] == "2q":
            q0, q1 = op[1], op[2]
            U_fused = op[3]
            j = i + 1
            while j < len(tile_ops) and tile_ops[j][0] == "2q":
                if (tile_ops[j][1] == q0 and tile_ops[j][2] == q1):
                    U_fused = cp.matmul(tile_ops[j][3], U_fused)
                    j += 1
                else:
                    break
            fused.append(("2q", q0, q1, U_fused))
            i = j
        else:
            fused.append(op)
            i += 1
    
    return fused
```

**Speedup**: Depends on circuit structure. Can reduce gates by 50%+ for circuits with repeated rotations.

---

## Option 3: Fused Tile Kernel with Cooperative Groups (Maximum Performance)

**Concept**: A single kernel processes ALL gates in a tile using grid-wide synchronization.

### New kernel in `kernels.cu`:

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Structure passed to the fused kernel
struct FusedGateOp {
    int kind;      // 1 = 1q, 2 = 2q
    int q0, q1;
    int U_offset;  // Offset into unified matrix buffer
};

__global__ void apply_tile_fused_kernel(
    cuDoubleComplex* __restrict__ psi,
    const cuDoubleComplex* __restrict__ U_buffer,
    const FusedGateOp* __restrict__ ops,
    int num_ops,
    int num_qubits
) {
    cg::grid_group grid = cg::this_grid();
    
    unsigned long long dim = 1ULL << num_qubits;
    
    for (int g = 0; g < num_ops; g++) {
        FusedGateOp op = ops[g];
        const cuDoubleComplex* U = U_buffer + op.U_offset;
        
        if (op.kind == 1) {
            // 1-qubit gate logic (inline, not function call)
            unsigned long long stride = 1ULL << op.q0;
            unsigned long long num_pairs = dim >> 1;
            
            for (unsigned long long k = grid.thread_rank(); k < num_pairs; k += grid.size()) {
                unsigned long long block = k / stride;
                unsigned long long offset = k % stride;
                unsigned long long i0 = block * 2ULL * stride + offset;
                unsigned long long i1 = i0 + stride;
                
                cuDoubleComplex a0 = psi[i0];
                cuDoubleComplex a1 = psi[i1];
                psi[i0] = cuCadd(cuCmul(U[0], a0), cuCmul(U[1], a1));
                psi[i1] = cuCadd(cuCmul(U[2], a0), cuCmul(U[3], a1));
            }
        } else {
            // 2-qubit gate logic (inline)
            // ... similar to apply_2q_gate_kernel but using grid stride loop
        }
        
        // Grid-wide sync between gates
        grid.sync();
    }
}

// Host launch function - must use cudaLaunchCooperativeKernel
void launch_tile_fused(
    cuDoubleComplex* psi,
    const cuDoubleComplex* U_buffer,
    const FusedGateOp* ops,
    int num_ops,
    int num_qubits
) {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    int numBlocksPerSm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm, apply_tile_fused_kernel, 256, 0);
    
    int numBlocks = numBlocksPerSm * prop.multiProcessorCount;
    
    void* args[] = {&psi, &U_buffer, &ops, &num_ops, &num_qubits};
    cudaLaunchCooperativeKernel(
        (void*)apply_tile_fused_kernel,
        numBlocks, 256, args);
}
```

### Required device capability check:
```cpp
// In constructor or init
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
if (!prop.cooperativeLaunch) {
    throw std::runtime_error("Device does not support cooperative launch");
}
```

**Speedup**: 5-10x for tiles with many small gates

---

## Recommendation

1. **Start with Option 1 (CUDA Graphs)** - Easiest to implement, good speedup
2. **Add Option 2 (Matrix Fusion)** as a preprocessing step - Complements any approach
3. **Consider Option 3** only if you need maximum throughput and have compatible hardware

## Quick Implementation (Option 1)

Here's a minimal diff to enable CUDA Graphs in your current code:

```cpp
// In Statevector class, modify apply_tile_dev:
void apply_tile_dev(py::list ops) {
    // ... existing parsing code ...
    
    py::gil_scoped_release release;
    
    // Use stream for all operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Begin graph capture
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    
    for (const auto& op : parsed) {
        if (op.kind == 1) {
            launch_apply_1q_gate(d_psi_, op.dU, num_qubits_, op.q0, stream);
        } else {
            launch_apply_2q_gate(d_psi_, op.dU, num_qubits_, op.q0, op.q1, stream);
        }
    }
    
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    
    // Execute entire tile as single graph launch
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);
    
    // Cleanup (or cache for reuse)
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
}
```

This single change converts your gate-by-gate execution to true tile-by-tile execution!
