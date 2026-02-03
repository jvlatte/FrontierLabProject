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

"""
Module for collecting resource usage metrics during simulation.
"""


###### new functions ######
def _tqc_stats(tqc: QuantumCircuit) -> Dict[str, int]:
    ops = tqc.count_ops()
    twoq = int(ops.get("cx", 0) + ops.get("cz", 0) + ops.get("swap", 0))
    # Count total gates (excluding barriers, measurements, save_statevector, etc.)
    non_gate_ops = {"barrier", "measure", "reset", "save_statevector", "snapshot"}
    total_gates = sum(v for k, v in ops.items() if k not in non_gate_ops)
    return {
        "nqubits_t": tqc.num_qubits,
        "depth_t": tqc.depth(),
        "cx_count": int(ops.get("cx", 0)),
        "cz_count": int(ops.get("cz", 0)),
        "swap_count": int(ops.get("swap", 0)),
        "rz_count": int(ops.get("rz", 0)),
        "rx_count": int(ops.get("rx", 0)),
        "twoq_count": twoq,
        "total_gates": total_gates,
    }


def _mem_snapshot() -> Dict[str, str]:
    out = {"rss_mb": "", "gpu_mem_mb": "", "gpu_util": ""}

    # CPU RSS (process)
    try:
        import psutil
        rss = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
        out["rss_mb"] = f"{rss:.2f}"
    except Exception:
        pass

    # GPU VRAM (process) + GPU util (device instantaneous)
    try:
        import pynvml
        pynvml.nvmlInit()
        pid = os.getpid()

        # device util (instantaneous, device-wide)
        handle0 = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle0)
        out["gpu_util"] = str(util.gpu)

        # sum VRAM used by *this process* across all GPUs
        total_bytes = 0
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)

            for fn in (pynvml.nvmlDeviceGetComputeRunningProcesses,
                       pynvml.nvmlDeviceGetGraphicsRunningProcesses):
                try:
                    procs = fn(h) or []
                except Exception:
                    procs = []

                for p in procs:
                    if int(p.pid) == int(pid):
                        used = getattr(p, "usedGpuMemory", 0) or 0
                        if used > 0:
                            total_bytes += int(used)

        out["gpu_mem_mb"] = f"{total_bytes / (1024**2):.2f}"

    except Exception:
        pass
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    return out


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _max_usage(a, b):
    """
    Merge two res_usage dicts by taking max of numeric fields.
    Keeps strings in your current CSV-friendly format.
    """
    out = dict(a) if a else {"rss_mb": "", "gpu_mem_mb": "", "gpu_util": ""}

    for k in ["rss_mb", "gpu_mem_mb"]:
        av = _to_float(out.get(k, ""))
        bv = _to_float(b.get(k, ""))
        if av is None:
            out[k] = b.get(k, out.get(k, ""))
        elif bv is None:
            pass
        else:
            out[k] = f"{max(av, bv):.2f}"

    # gpu_util is instantaneous; "peak" here = max observed across snapshots
    av = _to_float(out.get("gpu_util", ""))
    bv = _to_float(b.get("gpu_util", ""))
    if av is None:
        out["gpu_util"] = b.get("gpu_util", out.get("gpu_util", ""))
    elif bv is None:
        pass
    else:
        out["gpu_util"] = str(int(max(av, bv)))  # keep as int-like string

    return out

