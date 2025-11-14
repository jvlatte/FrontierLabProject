from qiskit import QuantumCircuit    
from typing import Dict

import os
try:
    import psutil
except Exception:
    psutil = None
try:
    import pynvml
    pynvml.nvmlInit()
except Exception:
    print("pynvml not available")
    pynvml = None


###### new functions ######
def _tqc_stats(tqc: QuantumCircuit) -> Dict[str, int]:
    ops = tqc.count_ops()
    twoq = int(ops.get("cx", 0) + ops.get("cz", 0) + ops.get("swap", 0))
    return {
        "nqubits_t": tqc.num_qubits,
        "depth_t": tqc.depth(),
        "cx_count": int(ops.get("cx", 0)),
        "cz_count": int(ops.get("cz", 0)),
        "swap_count": int(ops.get("swap", 0)),
        "rz_count": int(ops.get("rz", 0)),
        "rx_count": int(ops.get("rx", 0)),
        "twoq_count": twoq,
    }


def _mem_snapshot() -> Dict[str, float]:
    out = {"rss_mb": "", "gpu_mem_mb": "", "gpu_util": ""}
    if psutil:
        rss = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
        out["rss_mb"] = f"{rss:.2f}"
    if pynvml:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            out["gpu_mem_mb"] = f"{mem.used/1024**2:.2f}"
            out["gpu_util"] = f"{util.gpu}"
        except Exception:
            pass
    return out
