from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.library import SaveStatevector

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

qc.append(SaveStatevector(num_qubits=qc.num_qubits, label="psi"), qc.qubits)

qc.measure_all()

sim = AerSimulator(device="CPU")
tqc = transpile(qc, backend=sim)
res = sim.run(tqc, shots=1024).result()

sv = res.data(0)["psi"]      # saved statevector (pre-measure)
counts = res.get_counts()    # measurement results

print("Statevector:", sv)
print("Counts:", counts)
