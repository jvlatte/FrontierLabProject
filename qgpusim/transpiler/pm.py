from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CommutativeCancellation
from .passes import CancelSelfInversePairs, QubitInteractionAnalysis

def make_baseline_pm():
    pm = PassManager()
    pm.append([Optimize1qGates(), CommutativeCancellation()])
    return pm

def make_custom_pm():
    pm = make_baseline_pm()
    # run our extra optimization after the standard ones
    print("reached custom passes")
    pm.append(CancelSelfInversePairs())
    return pm
def make_custom_pm() -> PassManager:
    pm = PassManager()

    # local optimizations by qiskit transpiler already
    pm.append([Optimize1qGates(), CommutativeCancellation()])

    # build qubit–qubit interaction graph
    pm.append(QubitInteractionAnalysis())

    pm.append(CancelSelfInversePairs())


    return pm
