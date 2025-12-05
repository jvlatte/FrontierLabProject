from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import time

from .metrics.correctness import statevector_overlap, total_variation_distance
from .metrics.resources import _tqc_stats, _mem_snapshot
from .transpiler.pm import make_custom_pm
from .backends.custom.backend import run_custom_backend



def run_once(sim: AerSimulator, qc: QuantumCircuit, task: str, shots: int, measure: bool, 
             transpiler: str = "baseline", num_local_qubits: int = 4, gpu_backend: str = "aer"):
    print(f"the transpiler is: {transpiler}")
    if task == "sampling" and measure:
        qc_run = qc.copy()
        qc_run.measure_all()
    else:
        qc_run = qc

    # (1) transpile time
    transpile_t0 = time.perf_counter()
    if task == "statevector":
        qc_sv = qc_run.copy()
        qc_sv.save_statevector()

        if transpiler == "custom":
            print("running through custom transpiler")
            # custom pass manager
            tqc = transpile(qc_sv, sim)

            # 2) extra passes on top of that
            pm, lqr_pass = make_custom_pm(num_local_qubits)
            tqc = pm.run(tqc)

            # # code before adding on to baseline transpiler
            # pm = make_custom_pm()
            # tqc = pm.run(qc_sv)
        else:
            print("running through baseline transpiler")
            # baseline: regular qiskit transpile
            tqc = transpile(qc_sv, sim)
    else:
        if transpiler == "custom":
            pm, lqr_pass = make_custom_pm(num_local_qubits)
            tqc = pm.run(qc_run)
        else:
            tqc = transpile(qc_run, sim)
    transpile_s = time.perf_counter() - transpile_t0

    # (2) simulation time
    if gpu_backend == "aer":
        # original behavior
        simulate_t0 = time.perf_counter()
        result = sim.run(tqc, shots=shots if task == "sampling" else None).result()
        simulate_s = time.perf_counter() - simulate_t0
    else:
        # try custom gpu
        # TODO: implement this
        simulate_t0 = time.perf_
        run_custom_backend()
        result = sim.run(tqc, shots=shots if task == "sampling" else None).result()
        simulate_s = time.perf_counter() - simulate_t0


    # ----------------------------- PART THAT WAS COMMENTED OUT TO TRY NEW THINGS ----------------------------------------

    # simulate_t0 = time.perf_counter()
    # result = sim.run(tqc, shots=shots if task == "sampling" else None).result()
    # simulate_s = time.perf_counter() - simulate_t0

    # ----------------------------- PART THAT WAS COMMENTED OUT TO TRY NEW THINGS ----------------------------------------

    # get the results
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
