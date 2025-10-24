# from qiskit import QuantumCircuit, transpile
# from qiskit_aer import AerSimulator
# from qiskit_aer.library import SaveStatevector

# qc = QuantumCircuit(2)
# qc.h(0)
# qc.cx(0, 1)

# qc.append(SaveStatevector(num_qubits=qc.num_qubits, label="psi"), qc.qubits)

# qc.measure_all()

# sim = AerSimulator(device="GPU")
# tqc = transpile(qc, backend=sim)
# res = sim.run(tqc, shots=1024).result()

# sv = res.data(0)["psi"]      # saved statevector (pre-measure)
# counts = res.get_counts()    # measurement results

# print("Statevector:", sv)
# print("Counts:", counts)



import time
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.library import SaveStatevector

# --------- knobs you can tweak ----------
N_QUBITS = 24     # try 20–26 if your GPU has the memory
DEPTH    = 8      # more layers => heavier sim
SHOTS    = 4096   # for the sampling run
# ----------------------------------------

def build_circuit(n_qubits: int, depth: int, do_measure: bool):
    
    """
  
    qc   Builds a deterministic, moderately entangling circuit.
    - Layer: H on all qubits
    - Ring of CXs
    - Parametrized RY/RZ per qubit per layer
    """= QuantumCircuit(n_qubits)
    # save statevector BEFORE measurement so both CPU/GPU should give the same vector
    qc.append(SaveStatevector(num_qubits=n_qubits, label="psi"), qc.qubits)

    for d in range(depth):
        for q in range(n_qubits):
            qc.h(q)
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
        qc.cx(n_qubits - 1, 0)  # close the ring
        for q in range(n_qubits):
            # lightweight deterministic params (no randomness for reproducibility)
            qc.ry(0.05 * (q + 1) * (d + 1), q)
            qc.rz(0.03 * (q + 2) * (d + 1), q)

    if do_measure:
        qc.measure_all()

    return qc

def time_run(simulator, qc, shots=None):
    t0 = time.perf_counter()
    tqc = transpile(qc, backend=simulator, optimization_level=3)
    t1 = time.perf_counter()
    if shots is None:
        # statevector run
        result = simulator.run(tqc).result()
    else:
        result = simulator.run(tqc, shots=shots).result()
    t2 = time.perf_counter()
    return (t1 - t0), (t2 - t1), result  # (transpile_time, simulate_time, result)

# --- set up simulators ---
sim_cpu = AerSimulator(method="statevector", device="CPU")

# Try to create a GPU simulator; fall back cleanly if not available
sim_gpu = None
try:
    sim_gpu = AerSimulator(method="statevector", device="GPU")
except Exception as e:
    print("[WARN] GPU simulator not available:", e)

# --- build circuits ---
qc_state = build_circuit(N_QUBITS, DEPTH, do_measure=False)
qc_sample = build_circuit(N_QUBITS, DEPTH, do_measure=True)

# --- CPU runs ---
print(f"\n=== CPU (statevector) N={N_QUBITS}, depth={DEPTH} ===")
t_tr_cpu_sv, t_sim_cpu_sv, res_cpu_sv = time_run(sim_cpu, qc_state, shots=None)
print(f"Transpile: {t_tr_cpu_sv:.3f}s | Simulate: {t_sim_cpu_sv:.3f}s")

print(f"\n=== CPU (sampling) shots={SHOTS} ===")
t_tr_cpu_samp, t_sim_cpu_samp, res_cpu_samp = time_run(sim_cpu, qc_sample, shots=SHOTS)
print(f"Transpile: {t_tr_cpu_samp:.3f}s | Simulate: {t_sim_cpu_samp:.3f}s")

# --- GPU runs (if available) ---
if sim_gpu is not None:
    print(f"\n=== GPU (statevector) N={N_QUBITS}, depth={DEPTH} ===")
    t_tr_gpu_sv, t_sim_gpu_sv, res_gpu_sv = time_run(sim_gpu, qc_state, shots=None)
    print(f"Transpile: {t_tr_gpu_sv:.3f}s | Simulate: {t_sim_gpu_sv:.3f}s")

    print(f"\n=== GPU (sampling) shots={SHOTS} ===")
    t_tr_gpu_samp, t_sim_gpu_samp, res_gpu_samp = time_run(sim_gpu, qc_sample, shots=SHOTS)
    print(f"Transpile: {t_tr_gpu_samp:.3f}s | Simulate: {t_sim_gpu_samp:.3f}s")

    # sanity check: saved statevectors should match closely
    sv_cpu = res_cpu_sv.data(0)["psi"]
    sv_gpu = res_gpu_sv.data(0)["psi"]
    # print a quick fidelity-ish check (dot product magnitude)
    overlap = abs(sv_cpu.data.conj().dot(sv_gpu.data))
    print(f"\nStatevector overlap |⟨cpu|gpu⟩| ≈ {overlap:.12f}")

else:
    print("\n[INFO] Skipping GPU runs because the GPU simulator wasn’t available.")

print("\nDone.")

