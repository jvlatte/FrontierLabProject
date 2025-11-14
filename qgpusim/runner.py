from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import time

from .metrics.correctness import statevector_overlap, total_variation_distance
from .metrics.resources import _tqc_stats, _mem_snapshot


def run_once(sim: AerSimulator, qc: QuantumCircuit, task: str, shots: int, measure: bool):
    if task == "sampling" and measure:
        qc_run = qc.copy()
        qc_run.measure_all()
    else:
        qc_run = qc

    transpile_t0 = time.perf_counter()
    if task == "statevector":
        qc_sv = qc_run.copy()
        qc_sv.save_statevector()
        tqc = transpile(qc_sv, sim)
    else:
        tqc = transpile(qc_run, sim)
    transpile_s = time.perf_counter() - transpile_t0

    simulate_t0 = time.perf_counter()
    result = sim.run(tqc, shots=shots if task == "sampling" else None).result()
    simulate_s = time.perf_counter() - simulate_t0

    if task == "statevector":
        vec = result.get_statevector(tqc)
        try:
            from qiskit.quantum_info import Statevector
            sv = Statevector(vec)
            extra = {"statevector": [complex(a) for a in sv.data]}
        except Exception:
            extra = {"statevector": None}
    else:
        counts = result.get_counts(tqc)
        extra = {"counts": counts}

    # NEW: collect stats and resources
    stats = _tqc_stats(tqc)
    res_usage = _mem_snapshot()

    return transpile_s, simulate_s, extra, stats, res_usage
