import argparse
from qiskit import QuantumCircuit, transpile    
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT
import time
import numpy as np
import random
import datetime
from pathlib import Path
import csv
from typing import Dict, List, Tuple, Optional


from custom_logging import _ensure_csv, _env_info, _append_csv

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



def build_circuit(circuit: str, n_qubits: int, depth: int, seed: int):
    """builds test circuit
    ghz: GHZ state
    random: random layers of H, RX, RZ, CX
    qft: quantum fourier transform
    """
    qc = QuantumCircuit(n_qubits)
    if circuit == "ghz":
        qc.h(0)
        for i in range(1, n_qubits):
            qc.cx(0, i)
    
    elif circuit == "random":
        rng = random.Random(seed)
        for _ in range(max(1, depth)):
            for i in range(n_qubits):
                # random single-qubit rotations
                qc.rz(2 * np.pi * rng.random(), i)
                qc.rx(2 * np.pi * rng.random(), i)
            # random nearest-neighbor CX chain
            for i in range(0, n_qubits - 1, 2):
                qc.cx(i, i + 1)
            for i in range(1, n_qubits - 1, 2):
                qc.cx(i, i + 1)
        return qc

    elif circuit == "qft":
        qc = QFT(num_qubits=n_qubits, do_swaps=False)
        return qc
    
    else:
        raise ValueError(f"Unknown circuit type: {circuit}")


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


def create_simulator(backend: str, task: str):
    """creates AerSimulator"""
    method = "automatic"
    if task == "statevector":
        method = "statevector"
    elif task == "sampling":
        method = "automatic"
    
    device_used = "CPU"
    if backend == "cpu":
        sim = AerSimulator(method=method)  # default is CPU
    elif backend == "gpu":
        try:
            sim = AerSimulator(method=method, device="GPU")
            device_used = "GPU"
        except TypeError:
            sim = AerSimulator(method=method)
            print("GPU backend not available, falling back to CPU.")
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return sim, device_used


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


def single_backend(args: argparse.Namespace, fieldnames: list, env: dict, qc: QuantumCircuit):
    # only cpu or gpu+cpu
    sim, device_used = create_simulator(backend=args.backend, task=args.tasks)
    #device_used = "GPU" if (args.backend == "gpu") else "CPU"
    for i in range(args.repeats):
        trans_elapsed, sim_elapsed, extra, stats, res_usage = run_once(sim=sim, qc=qc, task=args.tasks, shots=args.shots, measure=True)
        final_str = (
        f"Run {i+1}/{args.repeats} on {args.backend} took {trans_elapsed:.4f} sec "
        f"to transpile and {sim_elapsed:.4f} sec to simulate. "
        f"Total: {(trans_elapsed + sim_elapsed):.4f}. Extra: {extra}"
        )
        print(final_str)


        if args.csv:
            row = {
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "host": env["host"],
                "os": env["os"],
                "python": env["python"],
                "backend": args.backend,
                "device": device_used,
                "task": args.tasks,
                "circuit": args.circuit,
                "nqubits": args.nqubits,
                "depth": args.depth,
                "shots": args.shots if args.tasks == "sampling" else 0,
                "repeat_idx": i + 1,
                "transpile_s": f"{trans_elapsed:.6f}",
                "simulate_s": f"{sim_elapsed:.6f}",
                "total_s": f"{trans_elapsed + sim_elapsed:.6f}",
                 **{k: stats[k] for k in ["nqubits_t","depth_t","twoq_count","cx_count","cz_count","swap_count","rz_count","rx_count"]},
                "rss_mb": res_usage.get("rss_mb",""),
                "gpu_mem_mb": res_usage.get("gpu_mem_mb",""),
                "gpu_util": res_usage.get("gpu_util",""),
                "notes": "",  # e.g., layout/method variants later
            }
            _append_csv(args.csv, row, fieldnames)



def compare_both_backends(args, qc: QuantumCircuit, fieldnames: List[str], csv_path: Optional[str]):
    # last arg might not be needed
    env = _env_info()
    timestamp = lambda: datetime.datetime.now().isoformat(timespec="seconds")

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
            "backend": backend,
            "device": device,
            "task": args.tasks,
            "circuit": args.circuit,
            "nqubits": args.nqubits,
            "depth": args.depth,
            "shots": args.shots if args.tasks == "sampling" else 0,
            "repeat_idx": repeat_idx,
            "transpile_s": f"{t_trans:.6f}",
            "simulate_s": f"{t_sim:.6f}",
            "total_s": f"{t_trans + t_sim:.6f}",
            "notes": notes,
        }
        return row

    def _inject_stats(row: Dict[str, object], stats: Dict[str, object], res_usage: Dict[str, object]) -> None:
        # Expected optional columns: nqubits_t, depth_t, twoq_count, cx_count, cz_count, swap_count, rz_count, rx_count
        for k in ["nqubits_t","depth_t","twoq_count","cx_count","cz_count","swap_count","rz_count","rx_count"]:
            if k in fieldnames and stats is not None:
                row[k] = stats.get(k, "")
        # Resource usage: rss_mb, gpu_mem_mb, gpu_util
        if res_usage:
            for k in ["rss_mb","gpu_mem_mb","gpu_util"]:
                if k in fieldnames:
                    row[k] = res_usage.get(k, "")

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
        

    # build CPU ref sim
    cpu_sim, cpu_device = create_simulator("cpu", args.tasks)

    # reference run (ONE run for all repeats)
    ref_trans_sec, ref_sim_sec, ref_extra, ref_stats, ref_res = run_once(
        sim=cpu_sim, qc=qc,task=args.tasks, shots=args.shots, measure=True)

    # extract ref artifact
    ref_sv = ref_extra.get("statevector") if args.tasks == "statevector" else None
    ref_counts = ref_extra.get("counts") if args.tasks != "statevector" else None

    if csv_path:
        cpu_row = _row_base("cpu", cpu_device, 0, ref_trans_sec, ref_sim_sec, notes="reference")
        _inject_stats(cpu_row, ref_stats, ref_res)
        # correctness cols stay blank for ref
        _append_csv(csv_path, _filtered(cpu_row), fieldnames)

    # gpu part now; keep running until repeats end
    gpu_sim, gpu_device = create_simulator("gpu", args.tasks)
    if "GPU" not in gpu_device:
        raise RuntimeError("GPU not present/available")

    for i in range(args.repeats):
        transpile_s, simulate_s, extra, g_stats, g_res = run_once(sim=gpu_sim, qc=qc,task=args.tasks, shots=args.shots, measure=True)
        total_s = transpile_s + simulate_s

        # # metrics
        # sv_overlap = ""
        # sv_l2 = ""
        # tvd = ""
        # passed = ""

        note = ""
        row = _row_base("gpu", gpu_device, i + 1, transpile_s, simulate_s)
        _inject_stats(row, g_stats, g_res)

        # correctness
        if args.tasks == "statevector":
            gpu_sv = extra.get("statevector")
            ov, l2 = statevector_overlap(ref_sv, gpu_sv)
            passed = (ov > 1 - 1e-9)
            print(
                f"[GPU sv] rep {i+1}/{args.repeats} | overlap={ov:.12f} l2={l2:.3e} "
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
                f"[GPU sampling] rep {i+1}/{args.repeats} | TVD={tvd:.6f}"
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

    args = parser.parse_args()
    
    # prep csv info
    fieldnames = [
    "timestamp","host","os","python",
    "backend","device","task","circuit","nqubits","depth","shots","repeat_idx",
    "transpile_s","simulate_s","total_s",
    # NEW compile/circuit stats
    "nqubits_t","depth_t","twoq_count","cx_count","cz_count","swap_count","rz_count","rx_count",
    # NEW resource usage
    "rss_mb","gpu_mem_mb","gpu_util",
    # NEW correctness (will be blank if not applicable)
    "overlap","l2","tvd","kl_div","passed",
    "notes"
    ]
    if args.csv:
        _ensure_csv(args.csv, fieldnames)
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    env = _env_info()


    # prepare circuit
    qc = build_circuit(circuit=args.circuit, n_qubits=args.nqubits, depth=args.depth, seed=args.seed)

    # prepare backends
    if args.backend == "compare":
        # compare both cpu and gpu statevector
        compare_both_backends(args, qc, fieldnames, args.csv)
        return
    else:
        # only cpu or gpu+cpu
        single_backend(args, fieldnames, env, qc)




if __name__ == "__main__":
    main()
