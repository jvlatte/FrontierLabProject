from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CommutativeCancellation
from .passes import CancelSelfInversePairs, QubitInteractionAnalysis, LazyQubitReordering

def make_baseline_pm():
    # will probably not use this
    pm = PassManager()
    pm.append([Optimize1qGates(), CommutativeCancellation()])
    return pm

# def make_custom_pm():
#     pm = make_baseline_pm()
#     # run our extra optimization after the standard ones
#     print("reached custom passes")
#     pm.append(CancelSelfInversePairs())
#     return pm


def make_custom_pm(num_local_qubits: int) -> PassManager:
    lqr_pass = LazyQubitReordering(num_local_qubits)
    pm = PassManager()

    # # 1) build interaction graph
    # pm.append(QubitInteractionAnalysis())

    # 2) do lazy reordering based on that graph
    # pm.append(lqr_pass)
    # 3) extra local cleanups
    pm.append(CancelSelfInversePairs())
    pm.append(lqr_pass)

    return pm, lqr_pass
