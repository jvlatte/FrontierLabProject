from collections import defaultdict
from qiskit.transpiler.basepasses import TransformationPass, AnalysisPass
from qiskit.dagcircuit import DAGCircuit


class CancelSelfInversePairs(TransformationPass):
    """Cancel adjacent pairs of self-inverse gates (e.g., X X, CX CX) on same qubits."""

    def __init__(self):
        super().__init__()
        self.self_inverse_gates = {"x", "z", "h", "cx", "cz", "swap"}

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        op_nodes = list(dag.topological_op_nodes())
        to_remove = set()

        i = 0
        while i < len(op_nodes) - 1:
            n1 = op_nodes[i]

            # if we already decided to remove n1, skip it
            if n1 in to_remove:
                i += 1
                continue

            n2 = op_nodes[i + 1]

            if (
                n1.op.name == n2.op.name
                and n1.op.name in self.self_inverse_gates
                and n1.qargs == n2.qargs
                and n1.cargs == n2.cargs
            ):
                # Found U, U on same wires -> cancel both
                to_remove.add(n1)
                to_remove.add(n2)
                i += 2  # skip past the pair
            else:
                i += 1

        for node in to_remove:
            dag.remove_op_node(node)

        print("ran through cancelselfinversepairs")
        return dag
    

class QubitInteractionAnalysis(AnalysisPass):
    """
    Analysis pass that builds a weighted qubit–qubit interaction graph.

    For every multi-qubit gate (e.g. CX, CZ, SWAP, etc.), we:
      - find the indices of the involved qubits
      - for each pair of involved qubits (i, j), i < j:
            interactions[(i, j)] += 1

    The result is stored in:
        self.property_set["qubit_interaction_graph"]
    as a dict mapping (i, j) -> weight (int).
    """

    def run(self, dag: DAGCircuit):
        interactions = defaultdict(int)

        # go through all operation nodes in topological order
        for node in dag.topological_op_nodes():
            qargs = node.qargs
            if len(qargs) < 2:
                continue  # only care about 2+ qubit gates

            # get qubit indices in the circuit's global ordering
            indices = []
            for q in qargs:
                bit_index = dag.find_bit(q).index
                indices.append(bit_index)

            # sort so (i, j) is canonical with i < j
            indices.sort()

            # for k-qubit gates, increment all pairwise edges
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    pair = (indices[i], indices[j])
                    interactions[pair] += 1

        # store the graph in the property_set
        self.property_set["qubit_interaction_graph"] = dict(interactions)
        return dag


class LazyQubitReordering(TransformationPass):
    "lets pray"
    def __init__(self, metadata: dict):
        super().__init__()
        self.metadata = metadata

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        op_nodes = list(dag.topological_op_nodes())
        
        pass
