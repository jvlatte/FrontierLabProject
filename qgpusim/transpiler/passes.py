from qiskit.transpiler.basepasses import TransformationPass

class MyFirstPass(TransformationPass):
    """dummy pass"""
    def run(self, dag):
        # TODO: implement actual optimization
        
        return dag
