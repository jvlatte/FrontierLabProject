import argparse
import csv
import datetime
import os
import platform
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT

import qiskit
import qiskit_aer
import random


# -------------------------
# Circuit builders
# -------------------------
def build_circuit(circuit: str, n_qubits: int, depth: int, seed: int):
    """builds test circuit
    ghz: GHZ state
    random: random layers of H, RX, RZ, CX
    qft: quantum fourier transform (no final swaps)
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

    elif circuit == "qft":
        qc = QFT(num_qubits=n_qubits, do_swaps=False).to_instruction().definition

    else:
        raise ValueError(f"Unknown circuit type: {circuit}")

    return qc


# -------------------------
# Env / CSV helpers
# -------------------------
def _env_info() -> Dict[str, str]:
    return {
        "host": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "qiskit": getattr(qiskit, "__version__", "unknown"),
        "qiskit_aer": getattr(qiskit_aer, "__version__", "unknown"),
    }


def _ensure_csv(path: str, fieldnames: List[str]):
    exists = os.path.exists(path)
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    if not exists:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def _append_csv(path: str, row: Dict[str, str], fieldnames: List[str]):
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


# -------------------------
# Metrics
# -------------------------
def statevector_overlap(cpu_vec: np.ndarray, gpu_vec: np.ndarray) -> Tuple[float, float]:
    """Return (|<cpu|gpu>|, L2 norm of difference)."""
    # Normalize just in case small numerical drift exists
    cpu = cpu_vec / np.linalg.norm(cpu_vec)
    gpu = gpu_vec / np.linalg.norm(gpu_vec)
    overlap = abs(np.vdot(cpu, gpu))
    l2 = np.linalg.norm(cpu - gpu)
    return float(overlap), float(l2)


def total_variation_distance(counts_a: Dict[str, int], counts_b: Dict[str, int]) -> float:
    """TVD between two empirical distributions from counts dicts."""
    keys = set(counts_a.keys()) | set(counts_b.keys())
    shots_a = sum(counts_a.values()) or 1
    shots_b = sum(counts_b.values()) or 1
    tvd = 0.0
    for k in keys:
        pa = counts_a.get(k, 0) / shots_a
        pb = counts_b.get(k, 0) / shots_b
        tvd += abs(pa - pb)
    return 0.5 * tvd


# -------------------------
# Simulator factory
# -------------------------
def create_simulator(backend: str, task: str, seed_sim: Optional[int] = None) -> Tuple[AerSimulator, str]:
    """creates AerSimulator and returns (sim, device_used_str)."""
    method = "automatic"
    if task == "statevector":
        method = "statevector"
    elif task == "sampling":
        method = "automatic"

    device_used = "CPU"
    if backend == "cpu":
        sim = AerSimulator(method=method)

    elif backend == "gpu":
        # Try GPU path
        try:
            sim = AerSimulator(method=method, device="GPU")
            device_used = "GPU"
        except TypeError:
            # Older versions may not accept device kw; try options()
            try:
                sim = AerSimulator(method=method)
                sim.set_options(device="GPU")
                device_used = "GPU"
            except Exception:
                sim = AerSimulator(method=method)
                device_used = "CPU (fallback GPU unavailable)"

    else:
        raise ValueError(f"Unknown backend: {backend}")

    if seed_sim is not None:
        try:
            sim.set_options(seed_simulator=seed_sim)
        except Exception:
            pass

    return sim, device_used


# -------------------------
# Single-run executor
# -------------------------
def run_once(
    sim: AerSimulator,
    qc: QuantumCircuit,
    task: str,
    shots: int,
    measure: bool,
    optimization_level: int,
    seed_transpiler: Optional[int] = None,
    layout_method: Optional[str] = None,
) -> Tuple[float, float, Dict]:
    """execute circuit and measure transpile & simulate time; returns (transpile_s, simulate_s, extra)"""
    if task == "sampling" and measure:
        qc_run = qc.copy()
        qc_run.measure_all()
    else:
        qc_run = qc

    # Prepare circuit for statevector readout
    if task == "statevector":
        qc_sv = qc_run.copy()
        qc_sv.save_statevector()
        circ_for_transpile = qc_sv
    else:
        circ_for_transpile = qc_run

    # TRANPILE
    t0 = time.perf_counter()
    tqc = transpile(
        circ_for_transpile,
        backend=sim,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
        layout_method=layout_method,
    )
    transpile_s = time.perf_counter() - t0

    # SIMULATE
    t1 = time.perf_counter()
    result = sim.run(tqc, shots=shots if task == "sampling" else None).result()
    simulate_s = time.perf_counter() - t1

    if task == "statevector":
        vec = result.get_statevector(tqc)
        extra = {"statevector": np.asarray(vec, dtype=np.complex128)}
    else:
        counts = result.get_counts(tqc)
        extra = {"counts": counts}

    return transpile_s, simulate_s, extra


# -------------------------
# Compare mode workflow
# -------------------------
def run_compare_mode(
    args,
    qc: QuantumCircuit,
    fieldnames: List[str],
    csv_path: Optional[str],
):
    env = _env_info()
    timestamp = lambda: datetime.datetime.now().isoformat(timespec="seconds")

    # --- Build CPU reference simulator ---
    cpu_sim, cpu_device = create_simulator("cpu", args.tasks, seed_sim=args.seed)

    # Reference run (ONE run for all repeats)
    ref_transpile_s, ref_simulate_s, ref_extra = run_once(
        sim=cpu_sim,
        qc=qc,
        task=args.tasks,
        shots=args.shots,
        measure=True,
        optimization_level=args.opt,
        seed_transpiler=args.seed,
        layout_method=args.layout,
    )

    # Extract the reference artifact
    ref_sv = None
    ref_counts = None
    if args.tasks == "statevector":
        ref_sv = ref_extra["statevector"]
    else:
        ref_counts = ref_extra["counts"]

    # --- Build GPU simulator (must be real GPU) ---
    gpu_sim, gpu_device = create_simulator("gpu", args.tasks, seed_sim=args.seed)
    if "GPU" not in gpu_device:
        raise RuntimeError("Compare mode requires a real GPU backend. GPU not available.")

    # Repeat GPU runs vs the CPU reference
    for i in range(args.repeats):
        transpile_s, simulate_s, extra = run_once(
            sim=gpu_sim,
            qc=qc,
            task=args.tasks,
            shots=args.shots,
            measure=True,
            optimization_level=args.opt,
            seed_transpiler=args.seed,
            layout_method=args.layout,
        )
        total_s = transpile_s + simulate_s

        # Metrics
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

        # Log row
        if csv_path:
            row = {
                "timestamp": timestamp(),
                "host": env["host"],
                "os": env["os"],
                "python": env["python"],
                "qiskit": env["qiskit"],
                "qiskit_aer": env["qiskit_aer"],
                "mode": "compare",
                "backend": "compare",
                "device": gpu_device,
                "ref_backend": "cpu",
                "compare_backend": "gpu",
                "task": args.tasks,
                "circuit": args.circuit,
                "nqubits": args.nqubits,
                "depth": args.depth,
                "shots": args.shots if args.tasks == "sampling" else 0,
                "repeat_idx": i + 1,
                "opt_level": args.opt,
                "layout": args.layout or "",
                "transpile_s": f"{transpile_s:.6f}",
                "simulate_s": f"{simulate_s:.6f}",
                "total_s": f"{total_s:.6f}",
                "ref_transpile_s": f"{ref_transpile_s:.6f}",
                "ref_simulate_s": f"{ref_simulate_s:.6f}",
                "sv_overlap": sv_overlap,
                "sv_l2": sv_l2,
                "tvd": tvd,
                "passed_correctness": passed,
                "notes": "",
            }
            _append_csv(csv_path, row, fieldnames)


# -------------------------
# Main
# -------------------------
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
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV log (e.g., results/bench.csv)")

    # Step 5 prep: let you pin transpile settings to keep compare-mode fair/reproducible
    parser.add_argument("--opt", type=int, default=1, choices=[0, 1, 2, 3], help="transpile optimization_level")
    parser.add_argument("--layout", type=str, default=None, choices=[None, "trivial", "dense", "sabre"], help="layout_method")

    args = parser.parse_args()

    # CSV schema
    fieldnames = [
        "timestamp", "host", "os", "python", "qiskit", "qiskit_aer",
        "mode", "backend", "device", "ref_backend", "compare_backend",
        "task", "circuit", "nqubits", "depth", "shots", "repeat_idx",
        "opt_level", "layout",
        "transpile_s", "simulate_s", "total_s",
        "ref_transpile_s", "ref_simulate_s",
        "sv_overlap", "sv_l2", "tvd",
        "passed_correctness", "notes"
    ]
    if args.csv:
        _ensure_csv(args.csv, fieldnames)

    # Prepare circuit
    qc = build_circuit(circuit=args.circuit, n_qubits=args.nqubits, depth=args.depth, seed=args.seed)

    if args.backend == "compare":
        run_compare_mode(args, qc, fieldnames, args.csv)
        return

    # Single-backend mode (CPU or GPU)
    env = _env_info()
    sim, device_used = create_simulator(backend=args.backend, task=args.tasks, seed_sim=args.seed)

    for i in range(args.repeats):
        transpile_s, simulate_s, extra = run_once(
            sim=sim,
            qc=qc,
            task=args.tasks,
            shots=args.shots,
            measure=True,
            optimization_level=args.opt,
            seed_transpiler=args.seed,
            layout_method=args.layout,
        )
        total_s = transpile_s + simulate_s

        print(
            f"[{args.backend}] rep {i+1}/{args.repeats} "
            f"| transpile={transpile_s:.4f}s simulate={simulate_s:.4f}s total={total_s:.4f}s"
        )

        if args.csv:
            row = {
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "host": env["host"],
                "os": env["os"],
                "python": env["python"],
                "qiskit": env["qiskit"],
                "qiskit_aer": env["qiskit_aer"],
                "mode": "single",
                "backend": args.backend,
                "device": device_used,
                "ref_backend": "",
                "compare_backend": "",
                "task": args.tasks,
                "circuit": args.circuit,
                "nqubits": args.nqubits,
                "depth": args.depth,
                "shots": args.shots if args.tasks == "sampling" else 0,
                "repeat_idx": i + 1,
                "opt_level": args.opt,
                "layout": args.layout or "",
                "transpile_s": f"{transpile_s:.6f}",
                "simulate_s": f"{simulate_s:.6f}",
                "total_s": f"{total_s:.6f}",
                "ref_transpile_s": "",
                "ref_simulate_s": "",
                "sv_overlap": "",
                "sv_l2": "",
                "tvd": "",
                "passed_correctness": "",
                "notes": "",
            }
            _append_csv(args.csv, row, fieldnames)


if __name__ == "__main__":
    main()
