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

    return transpile_s, simulate_s, extra


def single_backend(args: argparse.Namespace, fieldnames: list, env: dict, qc: QuantumCircuit):
    # only cpu or gpu+cpu
    sim, device_used = create_simulator(backend=args.backend, task=args.tasks)
    #device_used = "GPU" if (args.backend == "gpu") else "CPU"
    for i in range(args.repeats):
        trans_elapsed, sim_elapsed, extra = run_once(sim=sim, qc=qc, task=args.tasks, shots=args.shots, measure=True)
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
                "notes": "",  # e.g., layout/method variants later
            }
            _append_csv(args.csv, row, fieldnames)


def compare_both_backends(args, qc: QuantumCircuit, fieldnames: List[str], csv_path: Optional[str]):
    # last arg might not be needed
    env = _env_info()
    timestamp = lambda: datetime.datetime.now().isoformat(timespec="seconds")

    # build CPU ref sim
    cpu_sim, cpu_device = create_simulator("cpu", args.tasks)

    # reference run (ONE run for all repeats)
    ref_trans_sec, ref_sim_sec, ref_extra = run_once(sim=cpu_sim, qc=qc,task=args.tasks, shots=args.shots, measure=True)

    # extract ref artifact
    ref_sv = None
    ref_counts = None
    if args.tasks == "statevector":
        ref_sv = ref_extra["statevector"]
    else:
        ref_counts = ref_extra["counts"]

    # gpu part now; keep running until repeats end
    gpu_sim, gpu_device = create_simulator("gpu", args.tasks)
    if "GPU" not in gpu_device:
        raise RuntimeError("GPU not present/available")
    
    for i in range(args.repeats):
        transpile_s, simulate_s, extra = run_once(sim=gpu_sim, qc=qc,task=args.tasks, shots=args.shots, measure=True)
        total_s = transpile_s + simulate_s

        # metrics
        sv_overlap = ""
        sv_l2 = ""
        tvd = ""
        passed = ""

        if args.tasks == "statevector":
            gpu_sv = extra["statevector"]
            ov, l2 = statevector_overlap(ref_sv, gpu_sv)
            sv_overlap = f"{ov:.12f}"
            sv_l2 = f"{l2:.3e}"
            passed = str(ov > 1 - 1e-9)

            print(
                f"[compare sv] rep {i+1}/{args.repeats} "
                f"| overlap={sv_overlap} l2={sv_l2} "
                f"| t_transpile={transpile_s:.4f}s t_sim={simulate_s:.4f}s PASS={passed}"
            )

        else:
            gpu_counts = extra["counts"]
            tvd_val = total_variation_distance(ref_counts, gpu_counts)
            tvd = f"{tvd_val:.5f}"
            passed = str(tvd_val < 0.02)

            print(
                f"[compare sampling] rep {i+1}/{args.repeats} "
                f"| TVD={tvd} "
                f"| t_transpile={transpile_s:.4f}s t_sim={simulate_s:.4f}s PASS={passed}"
            )

    pass


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