from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CommutativeCancellation
from .passes import MyFirstPass

def make_baseline_pm():
    pm = PassManager()
    pm.append([Optimize1qGates(), CommutativeCancellation()])
    print("went through basline pass")
    return pm

def make_custom_pm():
    pm = make_baseline_pm()
    print("went through custom pass")
    pm.append(MyFirstPass())
    return pm
