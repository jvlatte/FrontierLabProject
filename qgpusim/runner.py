from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import time

from .metrics.correctness import statevector_overlap, total_variation_distance
from .metrics.resources import _max_usage, _tqc_stats, _mem_snapshot
from .transpiler.pm import make_custom_pm
from .backends.custom.backend import run_custom_backend



def run_once(sim: AerSimulator, qc: QuantumCircuit, task: str, shots: int, measure: bool, 
             transpiler: str = "baseline", num_local_qubits: int = 4, backend: str = "aer", device_used: str = "CPU",
             save_sv: bool = True):
    print(f"the transpiler is: {transpiler}")
    if task == "sampling" and measure:
        qc_run = qc.copy()
        qc_run.measure_all()
    else:
        qc_run = qc

    res_peak = {"rss_mb": "", "gpu_mem_mb": "", "gpu_util": ""}
    custom_metrics = None  # Will be populated for custom backend


    # (1) transpile time
    transpile_t0 = time.perf_counter()
    if task == "statevector":
        qc_sv = qc_run.copy()
        if save_sv:
            qc_sv.save_statevector()

        if transpiler == "custom":
            print("running through custom transpiler")
            # custom pass manager
            tqc = transpile(qc_sv, sim)

            # 2) extra passes on top of that
            pm, lqr_pass = make_custom_pm(num_local_qubits)
            tqc = pm.run(tqc)
            tiling_plan = lqr_pass.property_set.get("flat_tiling_plan", None)
            # # code before adding on to baseline transpiler
            # pm = make_custom_pm()
            # tqc = pm.run(qc_sv)
        else:
            print("running through baseline transpiler")
            # baseline: regular qiskit transpile
            # tqc = transpile(qc_sv, sim, optimization_level=0,)
            tqc = qc_sv
    else:
        if transpiler == "custom":
            pm, lqr_pass = make_custom_pm(num_local_qubits)
            tqc = pm.run(qc_run)
        else:
            tqc = transpile(qc_run, sim)
    transpile_s = time.perf_counter() - transpile_t0
    print(f"device used: {device_used}")

    res_peak = _max_usage(res_peak, _mem_snapshot())

    res_peak = _max_usage(res_peak, _mem_snapshot())

    # (2) simulation time
    if transpiler == "baseline":
        # only use aer gpu backend for baseline transpiler
        # original behavior
        simulate_t0 = time.perf_counter()
        result = sim.run(tqc, shots=shots if task == "sampling" else None).result()
        simulate_s = time.perf_counter() - simulate_t0
        print("ran with aer backend")
    else:
        # custom transpiler: can use aer or custom gpu backend
        if backend == "aer":
            # aer gpu backend
            simulate_t0 = time.perf_counter()
            result = sim.run(tqc, shots=shots if task == "sampling" else None).result()
            simulate_s = time.perf_counter() - simulate_t0
            print("ran with aer gpu backend")
        else:
            # try custom gpu
            if tiling_plan is None:
                raise ValueError("backend='custom' requires transpiler='custom' and tiling_plan")

            simulate_t0 = time.perf_counter()
            psi, custom_metrics = run_custom_backend(tqc, tiling_plan, task, shots, tile_mode="graphed", use_float32=True, enable_fusion=True)
            # result = sim.run(tqc, shots=shots if task == "sampling" else None).result()
            result = None  # placeholder
            # result = sim.run(tqc, shots=shots if task == "sampling" else None).result()
            simulate_s = time.perf_counter() - simulate_t0
            print("ran with custom gpu backend")
        
    res_peak = _max_usage(res_peak, _mem_snapshot())



    # ----------------------------- PART THAT WAS COMMENTED OUT TO TRY NEW THINGS ----------------------------------------

    # simulate_t0 = time.perf_counter()
    # result = sim.run(tqc, shots=shots if task == "sampling" else None).result()
    # simulate_s = time.perf_counter() - simulate_t0

    # ----------------------------- PART THAT WAS COMMENTED OUT TO TRY NEW THINGS ----------------------------------------

    # get the results
    if result is None:
        extra = {}
        if task == "statevector":
            if backend == "custom":
                # Keep as numpy array - avoid slow Python list conversion
                extra["statevector"] = psi
    else:
        if task == "statevector":
            vec = result.get_statevector(tqc)
            try:
                from qiskit.quantum_info import Statevector
                import numpy as np
                sv = Statevector(vec)
                # Keep as numpy array - avoid slow Python list conversion
                extra = {"statevector": np.asarray(sv.data, dtype=np.complex64)}
                # Free Aer's internal buffers to match custom backend's memory cleanup

                del sv, vec
            except Exception:
                extra = {"statevector": None}
        else:
            counts = result.get_counts(tqc)
            extra = {"counts": counts}
        
        # Explicit cleanup of Aer result object and garbage collection
        # This ensures fair RSS comparison with custom backend which also frees memory
        del result
        import gc
        gc.collect()
    
    res_peak = _max_usage(res_peak, _mem_snapshot())


    # NEW: collect stats and resources
    stats = _tqc_stats(tqc)
    
    # Merge custom backend metrics into stats if available
    if custom_metrics is not None:
        stats.update({
            "num_tiles": custom_metrics.get("num_tiles", ""),
            "num_gates_custom": custom_metrics.get("num_gates", ""),
            "num_fused_ops": custom_metrics.get("num_fused_ops", ""),
            "fusion_ratio": f"{custom_metrics.get('fusion_ratio', 1.0):.2f}",
            "num_reorders": custom_metrics.get("num_reorders", ""),
            "num_kernel_launches": custom_metrics.get("num_kernel_launches", ""),
            "num_graph_replays": custom_metrics.get("num_graph_replays", ""),
            "avoided_launches": custom_metrics.get("avoided_launches", ""),
            "t_perm_s": f"{custom_metrics.get('t_perm_s', 0.0):.6f}",
            "t_apply_s": f"{custom_metrics.get('t_apply_s', 0.0):.6f}",
            "t_sync_s": f"{custom_metrics.get('t_sync_s', 0.0):.6f}",
            "t_launch_overhead_saved_us": f"{custom_metrics.get('t_launch_overhead_saved_us', 0.0):.1f}",
            "tile_mode": custom_metrics.get("tile_mode", ""),
        })
    else:
        # For Aer backend: estimate kernel launches based on gate count
        # Aer GPU uses cuStateVec which typically batches operations better than naive gate-by-gate,
        # but for comparison purposes, we estimate as if it's ~1 kernel per gate (conservative)
        # This gives a baseline to compare against our graphed approach
        total_gates = stats.get("total_gates", 0)
        KERNEL_LAUNCH_OVERHEAD_US = 10.0  # same estimate as custom backend
        
        stats.update({
            "num_tiles": "",  # N/A for Aer
            "num_gates_custom": total_gates,  # use total_gates from circuit for comparison
            "num_fused_ops": total_gates,  # Aer doesn't do our fusion (estimate as 1:1)
            "fusion_ratio": "1.00",  # no fusion
            "num_reorders": "",  # N/A for Aer
            "num_kernel_launches": total_gates,  # estimate: ~1 kernel per gate
            "num_graph_replays": 0,  # Aer doesn't use CUDA graphs (typically)
            "avoided_launches": 0,  # no graph-based optimization
            "t_perm_s": "",  # N/A
            "t_apply_s": "",  # N/A (included in simulate_s)
            "t_sync_s": "",  # N/A (included in simulate_s)
            "t_launch_overhead_saved_us": "0.0",  # no savings (baseline)
            "tile_mode": "aer_sequential",  # mark as Aer's default mode
        })
    res_usage = res_peak


    return transpile_s, simulate_s, extra, stats, res_usage
