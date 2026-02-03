from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CommutativeCancellation
from .passes import CancelSelfInversePairs, LazyQubitReordering

def make_baseline_pm():
    # will probably not use this
    pm = PassManager()
    pm.append([Optimize1qGates(), CommutativeCancellation()])
    return pm

def make_custom_pm(num_local_qubits: int) -> PassManager:
    lqr_pass = LazyQubitReordering(num_local_qubits)
    pm = PassManager()

    # extra local cleanups
    # pm.append(CancelSelfInversePairs())
    pm.append(lqr_pass)

    return pm, lqr_pass
