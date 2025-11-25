from collections import defaultdict
from qiskit.transpiler.basepasses import TransformationPass, AnalysisPass
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from typing import List, Set, Dict, Tuple


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
    Analysis pass that builds a weighted qubit-qubit interaction graph.

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
        print("ran through qubit interaction analysis")
        return dag



class LazyQubitReordering(TransformationPass):
    def __init__(self, nL: int):
        super().__init__()
        self.nL = nL

    def tile_construction(self):
        # algorithm 2
        pass

    def greedy_qubit_mapping(self):
        # algorithm 3
        pass

    def flat_tiling(self, Q: Set[int], C: 
                    Set[DAGOpNode], T_map: Dict[DAGOpNode, 
                    Set[int]], nL: int) -> Tuple[list[DAGOpNode], dict[DAGOpNode, set[int]]]:
        # algorithm 1

        # Q_L <- {0, 1, ..., n_L - 1}
        # (local qubit indices; we assume nL <= |Q|)
        QL: Set[int] = set(range(min(nL, len(Q))))

        Cremain = C

        g_schedule = []
        M = {}

        x = 0  # position in schedule g

        while Cremain:
            G = self.tile_construction(QL, Cremain, T_map, nL)

            # if tile construction returns empty, avoid infinite loop.
            if not G:
                # schedule the first remaining gate as a single-tile G.
                G = [Cremain[0]]

            # for j = 0 ... |G| - 1 do
            for gate in G:
                # g(x) <- j-th gate of tile G
                g_schedule.append(gate)

                # M(g(x)) <- Q_L
                # (Store a *copy* of the current local set so later QL changes
                #  won’t mutate past entries.)
                M[gate] = set(QL)
                x += 1

                Cremain = [g for g in Cremain if g is not gate]


            # # Cremain <- C_remain \ {g(x)}
            # G_set = set(G)
            # Cremain = [gate for gate in Cremain if gate not in G_set]

            if Cremain:
                # Q_L <- QubitMapping(Q, C_remain, T, nL)
                QL = self.qubit_mapping(Q, Cremain, T_map, nL)

        # return (g, M)
        return g_schedule, M


        pass


    def run(self, dag: DAGCircuit) -> DAGCircuit:
        # prepare inputs for algoritm
        num_qubits = len(dag.qubits)
        Q = set(range(num_qubits))

        # map daq qubit -> int index
        qindex = {q: i for i, q in enumerate(dag.qubits)}

        C = set(dag.topological_op_nodes())  # C : set[DAGOpNode]
        C = sorted(C, key=lambda x: dag.node_depth(x))
        # m = len(C)   # number of gates

        T_map = {node: {qindex[q] for q in node.qargs} for node in C} # T_map: Dict[DAGOpNode, Set[int]]

        g, M = self.flat_tiling(Q, C, T_map, self.nL)







