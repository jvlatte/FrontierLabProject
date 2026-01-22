import argparse
from qiskit import QuantumCircuit
import numpy as np
import datetime
from pathlib import Path
import csv
from typing import Dict, List, Optional
from itertools import product


from .custom_logging import _ensure_csv, _env_info, _append_csv
from qgpusim.backends.aer.backend import create_simulator
from qgpusim.circuits.builders import build_circuit
from qgpusim.metrics.correctness import total_variation_distance, statevector_overlap
from qgpusim.runner import run_once
from qgpusim.transpiler.pm import make_baseline_pm, make_custom_pm



# def single_backend(combo: Dict, fieldnames: list, env: dict, qc: QuantumCircuit, args):
    # try gpu custom backend
    # if combo["backend"] == "custom":
    #     # TODO: ADD TO HERE
    #     pass
    # else:
    #     # only cpu or gpu+cpu with aer backends
    #     sim, device_used = create_simulator(combo["mode"], combo["tasks"])
    # #device_used = "GPU" if (args.backend == "gpu") else "CPU"
    # for i in range(combo["repeats"]):
    #     trans_elapsed, sim_elapsed, extra, stats, res_usage = run_once(sim=sim, qc=qc, task=combo["tasks"], 
    #                                                                    shots=combo["shots"], measure=True, 
    #                                                                    transpiler=combo["transpiler"], num_local_qubits=combo["nL"],
    #                                                                    backend=combo["backend"], device_used=device_used)
    #     final_str = (
    #     f"Run {i+1}/{combo['repeats']} on {combo['mode']} took {trans_elapsed:.4f} sec "
    #     f"to transpile and {sim_elapsed:.4f} sec to simulate. "
    #     f"Total: {(trans_elapsed + sim_elapsed):.4f}. "
    #     # f"Extra: {extra}"
    #     )
    #     print(final_str)


    #     if args.csv:
    #         row = {
    #             "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    #             "host": env["host"],
    #             "os": env["os"],
    #             "python": env["python"],
    #             "mode": combo["mode"],
    #             "device": device_used,
    #             "task": combo["tasks"],
    #             "circuit": combo["circuit"],
    #             "nqubits": combo["nqubits"],
    #             "depth": combo["depth"],
    #             "shots": combo["shots"] if combo["tasks"] == "sampling" else 0,
    #             "repeat_idx": i + 1,
    #             "transpile_s": f"{trans_elapsed:.6f}",
    #             "simulate_s": f"{sim_elapsed:.6f}",
    #             "total_s": f"{trans_elapsed + sim_elapsed:.6f}",
    #              **{k: stats[k] for k in ["nqubits_t","depth_t","twoq_count","cx_count","cz_count","swap_count","rz_count","rx_count"]},
    #             "rss_mb": res_usage.get("rss_mb",""),
    #             "gpu_mem_mb": res_usage.get("gpu_mem_mb",""),
    #             "gpu_util": res_usage.get("gpu_util",""),
    #             "notes": "",  # e.g., layout/method variants later
    #             "transpiler": combo["transpiler"],
    #             "nL": combo["nL"],
    #             "backend": combo["backend"]
    #         }
    #         _append_csv(args.csv, row, fieldnames)


def single_backend(combo: Dict, fieldnames: list, env: dict, qc: QuantumCircuit, csv_path):
    """
    GPU-only comparison:
      - Reference: GPU + baseline transpiler + Aer backend
      - Candidate: GPU + combo["transpiler"] + combo["backend"] (e.g., custom/custom)

    Writes BOTH rows to CSV (if args.csv is set) and prints PASS info.
    
    Handles OOM errors gracefully - logs the error and continues to next combination.
    """
    
    # Import cupy for GPU memory cleanup on OOM
    try:
        import cupy as cp
        HAS_CUPY = True
    except ImportError:
        HAS_CUPY = False

    def _cleanup_gpu_memory():
        """Free GPU memory after OOM or between runs."""
        if HAS_CUPY:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass
        # Also try to trigger Python garbage collection
        import gc
        gc.collect()

    def _filtered(row: Dict[str, object]) -> Dict[str, object]:
        return {k: v for k, v in row.items() if k in fieldnames}

    def _row_base(mode: str, device: str, repeat_idx: int,
                  t_trans: float, t_sim: float, notes: str = "") -> Dict[str, object]:
        return {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "host": env.get("host", ""),
            "os": env.get("os", ""),
            "python": env.get("python", ""),
            "mode": mode,                    # keep naming consistent with your CSV
            "device": device,
            "task": combo["tasks"],
            "circuit": combo["circuit"],
            "nqubits": combo["nqubits"],
            "depth": combo["depth"],
            "shots": combo["shots"] if combo["tasks"] == "sampling" else 0,
            "repeat_idx": repeat_idx,
            "transpile_s": f"{t_trans:.6f}",
            "simulate_s": f"{t_sim:.6f}",
            "total_s": f"{t_trans + t_sim:.6f}",
            "notes": notes,
            "transpiler": combo["transpiler"],
            "nL": combo["nL"],
            "backend": combo["backend"],
        }

    def _inject_stats(row: Dict[str, object], stats: Dict[str, object], res_usage: Dict[str, object]) -> None:
        for k in ["nqubits_t","depth_t","total_gates","twoq_count","cx_count","cz_count","swap_count","rz_count","rx_count"]:
            if k in fieldnames and stats is not None:
                row[k] = stats.get(k, "")
        if res_usage:
            for k in ["rss_mb","gpu_mem_mb","gpu_util"]:
                if k in fieldnames:
                    row[k] = res_usage.get(k, "")
        # Inject kernel launch overhead metrics (custom backend)
        if stats is not None:
            for k in ["num_tiles","num_gates_custom","num_fused_ops","fusion_ratio","num_reorders",
                      "num_kernel_launches","num_graph_replays","avoided_launches",
                      "t_perm_s","t_apply_s","t_sync_s","t_launch_overhead_saved_us","tile_mode"]:
                if k in fieldnames:
                    row[k] = stats.get(k, "")

    def _kl_div(counts_p: Dict[str, int], counts_q: Dict[str, int], eps: float = 1e-12) -> float:
        keys = set(counts_p.keys()) | set(counts_q.keys())
        sp = sum(counts_p.values()) or 1
        sq = sum(counts_q.values()) or 1
        k = len(keys) or 1
        kl = 0.0
        for key in keys:
            p = (counts_p.get(key, 0) + eps) / (sp + eps * k)
            q = (counts_q.get(key, 0) + eps) / (sq + eps * k)
            kl += p * np.log(p / q)
        return float(kl)

    # ----------------------------
    # For large qubit counts (>=32), skip reference to avoid OOM
    # ----------------------------
    skip_comparison = combo["nqubits"] >= 32
    if skip_comparison:
        print(f"[INFO] nqubits={combo['nqubits']} >= 32: skipping reference run to save memory")

    # ----------------------------
    # 1) Build GPU reference sim (only if comparison is needed)
    # ----------------------------
    ref_sim = None
    ref_device = None
    if not skip_comparison:
        ref_sim, ref_device = create_simulator("gpu", combo["tasks"])
        if "GPU" not in ref_device:
            raise RuntimeError("GPU not present/available for reference run")
    


    # ----------------------------
    # 2) Build candidate sim
    # ----------------------------
    if combo["backend"] == "custom":
        # HOOK: If your custom backend needs a different constructor, do it here.
        # For example: sim = create_custom_simulator(tasks=combo["tasks"], **...)
        sim, device_used = create_simulator("gpu", combo["tasks"])
        # Or, if your create_simulator already branches on combo["backend"], replace with:
        # sim, device_used = create_simulator("gpu", combo["tasks"], backend="custom")
        device_used = device_used  # keep naming consistent
    else:
        # Candidate is still a GPU Aer run but maybe different transpiler etc.
        sim, device_used = create_simulator("gpu", combo["tasks"])

    if "GPU" not in device_used:
        raise RuntimeError("GPU not present/available for candidate run")

    for i in range(combo["repeats"]):
        ref_sv = None
        ref_counts = None
        ref_oom = False  # track if reference run hit OOM

        # dont save statevector for large qubit counts (>= 32) 
        save_sv = (combo["tasks"] == "statevector") and (not skip_comparison) and (not ref_oom)

        # ---- Reference run (GPU baseline + Aer) - skip for large qubit counts ----
        if not skip_comparison:
            try:
                ref_trans_s, ref_sim_s, ref_extra, ref_stats, ref_res = run_once(
                    sim=ref_sim,
                    qc=qc,
                    task=combo["tasks"],
                    shots=combo["shots"],
                    measure=True,
                    transpiler="baseline",
                    num_local_qubits=combo["nL"],
                    backend="aer",
                    device_used=ref_device,
                    save_sv=save_sv
                )

                ref_sv = ref_extra.get("statevector") if combo["tasks"] == "statevector" else None
                ref_counts = ref_extra.get("counts") if combo["tasks"] != "statevector" else None

                # write reference row once per repeat (matches your current compare_both_backends behavior)
                if csv_path:
                    ref_row = _row_base("gpu_ref", ref_device, 0, ref_trans_s, ref_sim_s, notes="reference")
                    # override these so the CSV clearly shows what the reference actually used
                    if "transpiler" in ref_row: ref_row["transpiler"] = "baseline"
                    if "backend" in ref_row:    ref_row["backend"] = "aer"
                    _inject_stats(ref_row, ref_stats, ref_res)
                    _append_csv(csv_path, _filtered(ref_row), fieldnames)
                    
            except (MemoryError, RuntimeError, Exception) as e:
                # Catch OOM errors from CUDA (RuntimeError) or CPU (MemoryError)
                error_str = str(e).lower()
                is_oom = ("out of memory" in error_str or 
                          "cuda" in error_str and "memory" in error_str or
                          "alloc" in error_str or
                          isinstance(e, MemoryError))
                
                if is_oom:
                    print(f"[OOM] Reference run (baseline+aer) hit OOM at nqubits={combo['nqubits']}: {e}")
                    ref_oom = True
                    _cleanup_gpu_memory()
                    
                    # Log OOM to CSV
                    if csv_path:
                        oom_row = _row_base("gpu_ref", ref_device, 0, 0.0, 0.0, notes=f"OOM: {str(e)[:100]}")
                        if "transpiler" in oom_row: oom_row["transpiler"] = "baseline"
                        if "backend" in oom_row:    oom_row["backend"] = "aer"
                        _append_csv(csv_path, _filtered(oom_row), fieldnames)
                else:
                    # Re-raise non-OOM errors
                    raise

        # ---- Candidate run (GPU custom/custom or whatever combo says) ----
        try:
            trans_elapsed, sim_elapsed, extra, stats, res_usage = run_once(
                sim=sim,
                qc=qc,
                task=combo["tasks"],
                shots=combo["shots"],
                measure=True,
                transpiler=combo["transpiler"],
                num_local_qubits=combo["nL"],
                backend=combo["backend"],
                device_used=device_used,
            )
        except (MemoryError, RuntimeError, Exception) as e:
            # Catch OOM errors from CUDA (RuntimeError) or CPU (MemoryError)
            error_str = str(e).lower()
            is_oom = ("out of memory" in error_str or 
                      "cuda" in error_str and "memory" in error_str or
                      "alloc" in error_str or
                      isinstance(e, MemoryError))
            
            if is_oom:
                print(f"[OOM] Candidate run ({combo['backend']}/{combo['transpiler']}) hit OOM at nqubits={combo['nqubits']}: {e}")
                _cleanup_gpu_memory()
                
                # Log OOM to CSV and continue to next repeat/combo
                if csv_path:
                    oom_row = _row_base("gpu", device_used, i + 1, 0.0, 0.0, notes=f"OOM: {str(e)[:100]}")
                    _append_csv(csv_path, _filtered(oom_row), fieldnames)
                
                # Skip remaining repeats for this combo - it will likely OOM again
                print(f"[INFO] Skipping remaining repeats for this combo due to OOM")
                return  # exit single_backend, move to next combo
            else:
                # Re-raise non-OOM errors
                raise
                
        total_s = trans_elapsed + sim_elapsed

        # ---- Correctness compare + print ----
        note = ""
        ov, l2, tvd, kl, passed = None, None, None, None, None

        if skip_comparison or ref_oom:
            # No comparison - just log runtime metrics
            skip_reason = "ref OOM" if ref_oom else "large qubit count"
            print(
                f"[GPU {combo['backend']}/{combo['transpiler']}] "
                f"rep {i+1}/{combo['repeats']} | (no comparison - {skip_reason}) "
                f"| t_transpile={trans_elapsed:.4f}s t_sim={sim_elapsed:.4f}s total={total_s:.4f}s"
            )
            note = f"skipped_comparison ({skip_reason})"
        elif combo["tasks"] == "statevector":
            cand_sv = extra.get("statevector")
            ov, l2 = statevector_overlap(ref_sv, cand_sv)
            passed = (ov > 1 - 1e-6)  # 6 nines of precision - reasonable for large qubit counts
            print(
                f"[GPU ref=aer/baseline vs cand={combo['backend']}/{combo['transpiler']}] "
                f"rep {i+1}/{combo['repeats']} | overlap={ov:.12f} l2={l2:.3e} "
                f"| t_transpile={trans_elapsed:.4f}s t_sim={sim_elapsed:.4f}s total={total_s:.4f}s PASS={passed}"
            )
            note = f"overlap={ov:.12f}; l2={l2:.3e}; pass={passed}"
        else:
            cand_counts = extra.get("counts", {})
            tvd = total_variation_distance(ref_counts, cand_counts)
            passed = (tvd < 0.02)
            kl = _kl_div(ref_counts, cand_counts) if "kl_div" in fieldnames else None
            print(
                f"[GPU ref=aer/baseline vs cand={combo['backend']}/{combo['transpiler']}] "
                f"rep {i+1}/{combo['repeats']} | TVD={tvd:.6f}"
                + (f" KL={kl:.6f}" if kl is not None else "")
                + f" | t_transpile={trans_elapsed:.4f}s t_sim={sim_elapsed:.4f}s total={total_s:.4f}s PASS={passed}"
            )
            note = f"TVD={tvd:.6f}; " + (f"KL={kl:.6f}; " if kl is not None else "") + f"pass={passed}"

        # ---- Candidate CSV row ----
        if csv_path:
            row = _row_base("gpu", device_used, i + 1, trans_elapsed, sim_elapsed, notes=note)
            _inject_stats(row, stats, res_usage)

            # correctness columns (only if comparison was done)
            if not skip_comparison:
                if combo["tasks"] == "statevector":
                    if "overlap" in fieldnames: row["overlap"] = f"{ov:.12f}"
                    if "l2" in fieldnames:      row["l2"]      = f"{l2:.3e}"
                    if "passed" in fieldnames:  row["passed"]  = str(passed)
                else:
                    if "tvd" in fieldnames:     row["tvd"]     = f"{tvd:.6f}"
                    if kl is not None:          row["kl_div"]  = f"{kl:.6f}"
                    if "passed" in fieldnames:  row["passed"]  = str(passed)

            _append_csv(csv_path, _filtered(row), fieldnames)

        # Cleanup to free GPU memory between runs
        del extra, stats, res_usage
        import gc; gc.collect()
        if HAS_CUPY:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()



def compare_both_backends(combo: Dict, qc: QuantumCircuit, fieldnames: List[str], csv_path: Optional[str]):
    # last arg might not be needed
    env = _env_info()

    # For large qubit counts, skip comparison to avoid holding two statevectors in memory
    skip_comparison = combo["nqubits"] >= 32
    if skip_comparison:
        print(f"[INFO] nqubits={combo['nqubits']} >= 32: skipping comparison to save memory (will run both but not compare)")

    def _filtered(row: Dict[str, object]) -> Dict[str, object]:
        # keep only keys that exist in current CSV header
        return {k: v for k,v in row.items() if k in fieldnames}
    
    def _row_base(backend: str, device: str, repeat_idx: int,
                  t_trans: float, t_sim: float, notes: str="") -> Dict[str, object]:
        row = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "host": env.get("host", ""),
            "os": env.get("os", ""),
            "python": env.get("python", ""),
            "mode": backend,
            "device": device,
            "task": combo["tasks"],
            "circuit": combo["circuit"],
            "nqubits": combo["nqubits"],
            "depth": combo["depth"],
            "shots": combo["shots"] if combo["tasks"] == "sampling" else 0,
            "repeat_idx": repeat_idx,
            "transpile_s": f"{t_trans:.6f}",
            "simulate_s": f"{t_sim:.6f}",
            "total_s": f"{t_trans + t_sim:.6f}",
            "notes": notes,
            "transpiler": combo["transpiler"],
            "nL": combo["nL"],
            "backend": combo["backend"]
        }
        return row

    def _inject_stats(row: Dict[str, object], stats: Dict[str, object], res_usage: Dict[str, object]) -> None:
        # Expected optional columns: nqubits_t, depth_t, total_gates, twoq_count, cx_count, cz_count, swap_count, rz_count, rx_count
        for k in ["nqubits_t","depth_t","total_gates","twoq_count","cx_count","cz_count","swap_count","rz_count","rx_count"]:
            if k in fieldnames and stats is not None:
                row[k] = stats.get(k, "")
        # Resource usage: rss_mb, gpu_mem_mb, gpu_util
        if res_usage:
            for k in ["rss_mb","gpu_mem_mb","gpu_util"]:
                if k in fieldnames:
                    row[k] = res_usage.get(k, "")
        # Inject kernel launch overhead metrics (custom backend)
        if stats is not None:
            for k in ["num_tiles","num_gates_custom","num_fused_ops","fusion_ratio","num_reorders",
                      "num_kernel_launches","num_graph_replays","avoided_launches",
                      "t_perm_s","t_apply_s","t_sync_s","t_launch_overhead_saved_us","tile_mode"]:
                if k in fieldnames:
                    row[k] = stats.get(k, "")

    def _kl_div(counts_p: Dict[str, int], counts_q: Dict[str, int], eps: float = 1e-12) -> float:
        keys = set(counts_p.keys()) | set(counts_q.keys())
        sp = sum(counts_p.values()) or 1
        sq = sum(counts_q.values()) or 1
        k = len(keys) or 1
        kl = 0.0
        for key in keys:
            p = (counts_p.get(key, 0) + eps) / (sp + eps * k)
            q = (counts_q.get(key, 0) + eps) / (sq + eps * k)
            kl += p * np.log(p / q)
        return float(kl)
        

    # build CPU ref sim - always use baseline transpiler + aer backend as gold standard
    cpu_sim, cpu_device = create_simulator("cpu", combo["tasks"])
    ref_sv = None
    ref_counts = None
    
    for i in range(combo["repeats"]):
        # reference run (ONE run for all repeats) ####CHANGED TO REPEATS NOT ONE RUN######
        # for i in range(args.repeats):
        ref_trans_sec, ref_sim_sec, ref_extra, ref_stats, ref_res = run_once(
            sim=cpu_sim, qc=qc,task=combo["tasks"], shots=combo["shots"], measure=True,
            transpiler=combo["transpiler"], num_local_qubits=combo["nL"], backend="aer", device_used=cpu_device
            )

        # extract ref artifact - only if we need comparison (to avoid holding large statevectors)
        if not skip_comparison:
            ref_sv = ref_extra.get("statevector") if combo["tasks"] == "statevector" else None
            ref_counts = ref_extra.get("counts") if combo["tasks"] != "statevector" else None
        
        # Free memory immediately if skipping comparison
        if skip_comparison and combo["tasks"] == "statevector":
            del ref_extra
            import gc; gc.collect()

        # if csv_path and combo["transpiler"] == "baseline" and combo["backend"] == "aer":
        if csv_path:
            note = "reference" if not skip_comparison else "reference (no comparison - large qubit count)"
            cpu_row = _row_base("cpu", cpu_device, 0, ref_trans_sec, ref_sim_sec, notes=note)
            _inject_stats(cpu_row, ref_stats, ref_res)
            # correctness cols stay blank for ref
            _append_csv(csv_path, _filtered(cpu_row), fieldnames)

    gpu_sim, gpu_device = create_simulator("gpu", combo["tasks"])
    if "GPU" not in gpu_device:
        raise RuntimeError("GPU not present/available")

    for i in range(combo["repeats"]):
        transpile_s, simulate_s, extra, g_stats, g_res = run_once(sim=gpu_sim, qc=qc,task=combo["tasks"], 
                                                                  shots=combo["shots"], measure=True, transpiler=combo["transpiler"], 
                                                                  num_local_qubits=combo["nL"], backend=combo["backend"], device_used=gpu_device)
        total_s = transpile_s + simulate_s

        note = ""
        row = _row_base("gpu", gpu_device, i + 1, transpile_s, simulate_s)
        _inject_stats(row, g_stats, g_res)

        # correctness - skip comparison for large qubit counts to avoid OOM
        if skip_comparison:
            print(
                f"[GPU {combo['backend']}/{combo['transpiler']}] rep {i+1}/{combo['repeats']} "
                f"| (no comparison - large qubit count) "
                f"| t_transpile={transpile_s:.4f}s t_sim={simulate_s:.4f}s total={total_s:.4f}s"
            )
            note = "skipped_comparison (large qubit count)"
        elif combo["tasks"] == "statevector":
            gpu_sv = extra.get("statevector")
            ov, l2 = statevector_overlap(ref_sv, gpu_sv)
            passed = (ov > 1 - 1e-6)  # 6 nines of precision - reasonable for large qubit counts
            print(
                f"[GPU sv] rep {i+1}/{combo['repeats']} | overlap={ov:.12f} l2={l2:.3e} "
                f"| t_transpile={transpile_s:.4f}s t_sim={simulate_s:.4f}s total={total_s:.4f}s PASS={passed}"
            )
            note = f"overlap={ov:.12f}; l2={l2:.3e}; pass={passed}"
            if "overlap" in fieldnames: row["overlap"] = f"{ov:.12f}"
            if "l2" in fieldnames:      row["l2"]      = f"{l2:.3e}"
            if "passed" in fieldnames:  row["passed"]  = str(passed)
        else:
            gpu_counts = extra.get("counts", {})
            tvd = total_variation_distance(ref_counts, gpu_counts)
            passed = (tvd < 0.02)
            kl = _kl_div(ref_counts, gpu_counts) if "kl_div" in fieldnames else None
            print(
                f"[GPU sampling] rep {i+1}/{combo['repeats']} | TVD={tvd:.6f}"
                + (f" KL={kl:.6f}" if kl is not None else "")
                + f" | t_transpile={transpile_s:.4f}s t_sim={simulate_s:.4f}s total={total_s:.4f}s PASS={passed}"
            )
            note = f"TVD={tvd:.6f}; " + (f"KL={kl:.6f}; " if kl is not None else "") + f"pass={passed}"
            if "tvd" in fieldnames:     row["tvd"]     = f"{tvd:.6f}"
            if kl is not None:          row["kl_div"]  = f"{kl:.6f}"
            if "passed" in fieldnames:  row["passed"]  = str(passed)

        # keep human-readable note regardless of column set
        row["notes"] = note
        if csv_path:
            _append_csv(csv_path, _filtered(row), fieldnames)


def main():
    # parameters to tweak
    # Note: transpiler and backend are paired - only valid combinations are:
    #   (baseline, aer) - standard Qiskit
    #   (custom, custom) - tiled approach with custom backend
    
    params = {
        "mode": ["gpu"],
        "tasks": ["statevector"],
        "circuit": ["random"],
        "nqubits": [32],
        "depth": [16],
        "shots": [1024],
        "repeats": [1],
        "seed": [42],
        "nL": [14],
        # "transpiler": ["baseline", "custom"],  # filled in later
        # "backend": ["aer", "custom"],     # filled in later
    }
    
    # # Valid (transpiler, backend) pairs
    valid_transpiler_backend_pairs = [
        ("baseline", "aer"),
        ("custom", "custom"),
    ]

    keys = list(params.keys())
    config_list = []
    for combination in product(*params.values()):
        base_config = dict(zip(keys, combination))
        # config_list.append(base_config)
        # Add each valid transpiler/backend pair
        for transpiler, backend in valid_transpiler_backend_pairs:
            config = base_config.copy()
            config["transpiler"] = transpiler
            config["backend"] = backend
            config_list.append(config)

    # print(config_list)
    # print(f"total number is {len(config_list)}")


    parser = argparse.ArgumentParser(description="Benchmarking Script")

    parser.add_argument("--backend", type=str, choices=["cpu", "gpu", "compare"], default="cpu")
    parser.add_argument("--tasks", type=str, choices=["statevector", "sampling"], default="statevector")
    parser.add_argument("--circuit", type=str, choices=["random", "ghz", "qft"], default="random")
    parser.add_argument("--nqubits", type=int, default=10)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    # default for csv is set; change later on
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    csv_path = PROJECT_ROOT / "results" / "result.csv"

    parser.add_argument("--csv", type=str, default=str(csv_path))

    #parser.add_argument("--csv", type=str, default="results/result.csv", help="Path to CSV log (e.g., results/bench.csv)")

    parser.add_argument("--transpiler", type=str, choices=["baseline", "custom"], default="baseline")

    args = parser.parse_args()
    
    # prep csv info
    fieldnames = [
    "timestamp","host","os","python",
    "mode","device","task","circuit","nqubits","depth","shots","repeat_idx",
    "transpile_s","simulate_s","total_s",
    # NEW compile/circuit stats
    "nqubits_t","depth_t","total_gates","twoq_count","cx_count","cz_count","swap_count","rz_count","rx_count",
    # NEW resource usage
    "rss_mb","gpu_mem_mb","gpu_util",
    # NEW correctness (will be blank if not applicable)
    "overlap","l2","tvd","kl_div","passed",
    # NEW kernel launch overhead metrics (custom backend only)
    "num_tiles","num_gates_custom","num_fused_ops","fusion_ratio","num_reorders",
    "num_kernel_launches","num_graph_replays","avoided_launches",
    "t_perm_s","t_apply_s","t_sync_s","t_launch_overhead_saved_us","tile_mode",
    "notes", "transpiler", "nL", "backend"
    ]
    if args.csv:
        _ensure_csv(args.csv, fieldnames)
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    env = _env_info()

    # start loop through combinations
    combo_num = 1
    for combo in config_list:
        print(f"\n=== Running combo {combo_num}/{len(config_list)}: {combo} ===")

        # prepare circuit
        qc = build_circuit(combo["circuit"], combo["nqubits"], combo["depth"], combo["seed"])


        # prepare backends
        if combo["mode"] == "compare":
            # # compare both cpu and gpu statevector
            # compare_both_backends(combo, qc, fieldnames, args.csv)
            try:
                compare_both_backends(combo, qc, fieldnames, args.csv)
            except RuntimeError as e:
                print(f"[WARN] {e} - no gpu present.")
            except (MemoryError, Exception) as e:
                error_str = str(e).lower()
                if "out of memory" in error_str or "alloc" in error_str or isinstance(e, MemoryError):
                    print(f"[OOM] compare_both_backends hit OOM at nqubits={combo['nqubits']}: {e}")
                    print(f"[INFO] Continuing to next combo...")
                else:
                    raise

        else:
            # only cpu or gpu+cpu
            try:
                single_backend(combo, fieldnames, env, qc, args.csv)
            except (MemoryError, RuntimeError, Exception) as e:
                error_str = str(e).lower()
                is_oom = ("out of memory" in error_str or 
                          "cuda" in error_str and "memory" in error_str or
                          "alloc" in error_str or
                          isinstance(e, MemoryError))
                if is_oom:
                    print(f"[OOM] single_backend hit OOM at nqubits={combo['nqubits']}: {e}")
                    print(f"[INFO] Continuing to next combo...")
                else:
                    raise
        combo_num += 1

if __name__ == "__main__":
    main()
