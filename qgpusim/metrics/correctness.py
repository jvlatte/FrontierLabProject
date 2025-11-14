import numpy as np
from typing import Dict, Tuple


# two functions below 4 metric loggin (*****add to diff module late******)
def statevector_overlap(cpu_vec: np.ndarray, gpu_vec: np.ndarray) -> Tuple[float, float]:
    """ calc and return |<cpu|gpu>|, l2 norm of diff"""
    cpu = cpu_vec / np.linalg.norm(cpu_vec)
    gpu = gpu_vec / np.linalg.norm(gpu_vec)
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
