import numpy as np
from typing import Dict, Tuple, Union, List


# two functions below 4 metric loggin (*****add to diff module late******)
def statevector_overlap(cpu_vec: Union[np.ndarray, List], gpu_vec: Union[np.ndarray, List]) -> Tuple[float, float]:
    """Calc and return |<cpu|gpu>|, l2 norm of diff.
    
    Accepts numpy arrays (preferred) or lists. Arrays are converted efficiently
    without Python-level iteration.
    """
    # Ensure numpy arrays - use asarray for zero-copy when already numpy
    cpu_arr = np.asarray(cpu_vec, dtype=np.complex128) if not isinstance(cpu_vec, np.ndarray) else cpu_vec
    gpu_arr = np.asarray(gpu_vec, dtype=np.complex128) if not isinstance(gpu_vec, np.ndarray) else gpu_vec
    
    cpu = cpu_arr / np.linalg.norm(cpu_arr)
    gpu = gpu_arr / np.linalg.norm(gpu_arr)
    overlap = abs(np.vdot(cpu, gpu))
    l2 = np.linalg.norm(cpu - gpu)
    return float(overlap), float(l2)
    
def total_variation_distance(counts_a: Dict[str, int], counts_b: Dict[str, int]) -> float:
    """TVD between two empirical distributions from count dicts"""
    # look more into this; might not be needed
    keys = set(counts_a.keys()) | set(counts_b.keys())
    shots_a = sum(counts_a.values()) or 1
    shots_b = sum(counts_b.values()) or 1
    tvd = 0.0
    for k in keys:
        pa = counts_a.get(k, 0) / shots_a
        pb = counts_b.get(k, 0) / shots_b
        tvd += abs(pa - pb)
    return 0.5 * tvd
