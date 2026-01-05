from qiskit_aer import AerSimulator
from typing import Optional, Tuple


def create_simulator(
    backend: str, 
    task: str,
    precision: str = "double",
    max_parallel_threads: Optional[int] = None,
    max_parallel_experiments: int = 1,
    blocking_enable: bool = False,
    blocking_qubits: Optional[int] = None
) -> Tuple[AerSimulator, str]:
    """Creates an optimized AerSimulator instance.
    
    Args:
        backend: "cpu" or "gpu" device selection
        task: "statevector" or "sampling" simulation type
        precision: "single" or "double" precision (single is ~2x faster on GPU)
        max_parallel_threads: Number of OpenMP threads (None = auto)
        max_parallel_experiments: Number of parallel circuit executions
        blocking_enable: Enable cache blocking for large circuits
        blocking_qubits: Number of qubits per block (for cache blocking)
        
    Returns:
        Tuple of (AerSimulator instance, device string used)
    """
    method = "automatic"
    if task == "statevector":
        method = "statevector"
    elif task == "sampling":
        method = "automatic"
    
    # Build simulator options
    sim_options = {
        "method": method,
        "precision": precision,
        "max_parallel_experiments": max_parallel_experiments,
    }
    
    # Add optional threading configuration
    if max_parallel_threads is not None:
        sim_options["max_parallel_threads"] = max_parallel_threads
    
    # Add cache blocking for large circuits (reduces memory bandwidth)
    if blocking_enable:
        sim_options["blocking_enable"] = True
        if blocking_qubits is not None:
            sim_options["blocking_qubits"] = blocking_qubits
    
    device_used = "CPU"
    if backend == "cpu":
        sim = AerSimulator(**sim_options)
    elif backend == "gpu":
        try:
            sim_options["device"] = "GPU"
            # GPU-specific optimizations
            sim_options["cuStateVec_enable"] = True  # Use cuStateVec if available
            sim = AerSimulator(**sim_options)
            device_used = "GPU"
        except TypeError:
            # Fall back if GPU options not supported
            del sim_options["device"]
            if "cuStateVec_enable" in sim_options:
                del sim_options["cuStateVec_enable"]
            sim = AerSimulator(**sim_options)
            print("GPU backend not available, falling back to CPU.")
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    return sim, device_used


def create_batched_simulator(
    backend: str,
    task: str,
    batch_size: int = 4,
    precision: str = "double"
) -> Tuple[AerSimulator, str]:
    """Creates an AerSimulator optimized for batched circuit execution.
    
    Use this when running multiple similar circuits (e.g., VQE parameter sweeps).
    
    Args:
        backend: "cpu" or "gpu" device selection
        task: "statevector" or "sampling" simulation type
        batch_size: Number of circuits to execute in parallel
        precision: "single" or "double" precision
        
    Returns:
        Tuple of (AerSimulator instance, device string used)
    """
    return create_simulator(
        backend=backend,
        task=task,
        precision=precision,
        max_parallel_experiments=batch_size,
        blocking_enable=True,
    )
