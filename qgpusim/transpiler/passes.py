from collections import defaultdict
from qiskit.transpiler.basepasses import TransformationPass, AnalysisPass
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from typing import List, Set, Dict, Tuple
from dataclasses import dataclass

# -----------HELPER CLASS-----------------

@dataclass
class Tile:
    tile_idx: int

    # Global qubit IDs that are local in this tile (Q_L)
    logical_qubits: List[int]

    # global qubit -> local bit index (0..|Q_L|-1)
    global_to_local_map: Dict[int, int]

    # Desired statevector layout for this tile:
    # layout[pos] = global qubit at bit-position pos
    layout: List[int]

    # Gates belonging to this tile (compile-time)
    gates: List[DAGOpNode]


def _build_tile_layout(logical_qubits: List[int], num_qubits: int) -> List[int]:
    """
    layout[pos] = global qubit at bit-position pos.
    We force tile's logical_qubits to occupy the lowest bit positions.
    """
    ql = list(logical_qubits)
    ql_set = set(ql)
    rest = [q for q in range(num_qubits) if q not in ql_set]
    return ql + rest

# ------------ PASSES -------------

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

    # ---------------- Algorithm 2: TileConstruction ----------------
    def tile_construction(
        self,
        QL: Set[int],
        Cremain: List[DAGOpNode],
        T_map: Dict[DAGOpNode, Set[int]],
    ) -> List[DAGOpNode]:
        """
        TILECONSTRUCTION(Q_L, C, T)
        Input:
            QL      - set of *local* qubits for the current tile
            Cremain - ordered list of remaining gates (by depth)
            T_map   - node -> set of (global) qubit indices

        Output:
            Tile G: ordered list of DAGOpNode
        """
        # 2: Q_avail <- Q_L
        Q_avail: Set[int] = set(QL)
        # 3: G <- ∅
        G: List[DAGOpNode] = []
        # 4: C' <- C   (we just iterate over Cremain in order)
        C_prime = list(Cremain)

        # 5: for i = 0 ... |C'| - 1 do
        for U in C_prime:
            # 7: if T(U) ⊆ Q_avail then
            targets = T_map[U]
            if targets.issubset(Q_avail):
                # 8: Append U to the tail of G;
                G.append(U)
                # 9: C' <- C' \ {U}   (not needed explicitly in Python loop)
            else:
                # 11: Q_avail <- Q_avail \ T(U)
                Q_avail -= targets
                # 12: if Q_avail = ∅ then break
                if not Q_avail:
                    break

        # 14: return G
        return G

    # ---------------- Algorithm 3: QubitMapping ----------------
    def greedy_qubit_mapping(
        self,
        Q: Set[int],
        Cremain: List[DAGOpNode],
        T_map: Dict[DAGOpNode, Set[int]],
        nL: int,
    ) -> Set[int]:
        """
        QUBITMAPPING(Q, C, T, n_L)
        Returns updated Q_L for the *next* tile.
        """
        # 2: Q_avail <- Q
        Q_avail: Set[int] = set(Q)
        # 3: Q_L <- ∅
        QL: Set[int] = set()
        # 4: C' <- C
        C_prime = list(Cremain)

        # 5: for i = 0 ... |C'| - 1 do
        for U in C_prime:
            targets = T_map[U]
            # 7: if T(U) ⊆ Q_avail ∧ |Q_L ∪ T(U)| ≤ n_L then
            if targets.issubset(Q_avail) and len(QL | targets) <= nL:
                # 8: Q_L <- Q_L ∪ T(U)
                QL |= targets
            else:
                # 10: Q_avail <- Q_avail \ T(U)
                Q_avail -= targets
                # 11–12: if Q_avail = ∅ then break
                if not Q_avail:
                    break

        # 13–14: if Q_L = ∅ then return ∅
        if not QL:
            return set()

        # 15–17: while |Q_L| < n_L, fill with lowest remaining qubits
        if len(QL) < nL:
            remaining = sorted(Q - QL)
            for q in remaining:
                if len(QL) >= nL:
                    break
                QL.add(q)

        # 18: return Q_L
        return QL

    # alias so your existing call self.qubit_mapping(...) still works
    qubit_mapping = greedy_qubit_mapping

    # ---------------- Algorithm 1: FlatTimeSpaceTiling ----------------
    def flat_tiling(
        self,
        Q: Set[int],
        C: List[DAGOpNode],
        T_map: Dict[DAGOpNode, Set[int]],
        nL: int,
    ) -> Tuple[List[DAGOpNode], Dict[DAGOpNode, Set[int]]]:
        """
        FLATTIMESPACETILING(Q, C, T, n_L)

        Returns:
            g_schedule: ordered list of gates g(0),...,g(m-1)
            M: mapping M[gate] = set of local qubits Q_L used for that gate
        """
        # 2: Q_L <- {0, 1, ..., n_L - 1}
        QL: Set[int] = set(range(min(nL, len(Q))))
        # 3: C_remain <- (Sort C in ascending order of depths)
        Cremain: List[DAGOpNode] = list(C)  # already sorted by caller

        g_schedule: List[DAGOpNode] = []
        M: Dict[DAGOpNode, Set[int]] = {}

        # 4: x <- 0   (we just use len(g_schedule) instead of explicit x)
        while Cremain:
            # 6: G <- TILECONSTRUCTION(Q_L, C_remain, T)
            G = self.tile_construction(QL, Cremain, T_map)

            # Safety: avoid infinite loop if TILECONSTRUCTION somehow
            # returns empty even though Cremain is non-empty.
            # This can happen with long-range gates whose targets span
            # more qubits than the current tile.
            if not G:
                # Force the first remaining gate into its own tile.
                # Update Q_L to include all qubits this gate needs.
                forced_gate = Cremain[0]
                gate_qubits = T_map[forced_gate]
                # Build a new Q_L that includes this gate's qubits
                QL = set(gate_qubits)
                # Fill remaining slots with lowest available qubits
                if len(QL) < nL:
                    remaining = sorted(Q - QL)
                    for q in remaining:
                        if len(QL) >= nL:
                            break
                        QL.add(q)
                G = [forced_gate]

            # 7–11: for every gate in the tile
            for gate in G:
                # g(x) <- j-th gate of tile G
                g_schedule.append(gate)
                # M(g(x)) <- Q_L    (store a copy!)
                M[gate] = set(QL)
                # C_remain <- C_remain \ {g(x)}
                Cremain = [g for g in Cremain if g is not gate]

            # 12: Q_L <- QubitMapping(Q, C_remain, T, n_L)
            if Cremain:
                QL = self.qubit_mapping(Q, Cremain, T_map, nL)

        # 13: return (g, M)
        return g_schedule, M
    


        # ------------- Helpers: extract tiles & summarize ----------------
    @staticmethod
    def extract_tiles(
        g_schedule: list[DAGOpNode],
        M: dict[DAGOpNode, set[int]],
    ):
        """
        Group gates into tiles where Q_L stays constant.
        Returns a list of (Q_L, [gate1, gate2, ...]) pairs.
        """
        tiles = []
        if not g_schedule:
            return tiles

        curr_ql = M[g_schedule[0]]
        curr_tile = [g_schedule[0]]

        for node in g_schedule[1:]:
            if M[node] == curr_ql:
                curr_tile.append(node)
            else:
                tiles.append((curr_ql, curr_tile))
                curr_ql = M[node]
                curr_tile = [node]

        tiles.append((curr_ql, curr_tile))
        return tiles

    @staticmethod
    def summarize_tiling(
        g_schedule: list[DAGOpNode],
        M: dict[DAGOpNode, set[int]],
    ):
        """
        Compute simple statistics about the tiling:
          - number of tiles
          - avg tile size (gates per tile)
          - max tile size
          - avg |Q_L|
          - max |Q_L|
        Returns a dict with these fields.
        """
        tiles = LazyQubitReordering.extract_tiles(g_schedule, M)
        num_tiles = len(tiles)
        if num_tiles == 0:
            return {
                "num_tiles": 0,
                "avg_tile_size": 0.0,
                "max_tile_size": 0,
                "avg_Q_L_size": 0.0,
                "max_Q_L_size": 0,
            }

        tile_sizes = [len(gates) for (_ql, gates) in tiles]
        ql_sizes = [len(ql) for (ql, _gates) in tiles]

        avg_tile_size = sum(tile_sizes) / num_tiles
        max_tile_size = max(tile_sizes)
        avg_ql_size = sum(ql_sizes) / num_tiles
        max_ql_size = max(ql_sizes)

        return {
            "num_tiles": num_tiles,
            "avg_tile_size": avg_tile_size,
            "max_tile_size": max_tile_size,
            "avg_Q_L_size": avg_ql_size,
            "max_Q_L_size": max_ql_size,
        }
    
    @staticmethod
    def build_tiling_plan(g_schedule: list[DAGOpNode], M: dict[DAGOpNode, set[int]], num_qubits: int) -> List[Tile]:
        # use g and M to build list of Tiles
        raw_tiles = LazyQubitReordering.extract_tiles(g_schedule, M)
        tile_plan = []

        # for tile_idx, (ql, gates) in enumerate(raw_tiles):
        #     global_to_local = {gq: lq for lq, gq in enumerate(sorted(ql))}
        #     tile = Tile(
        #         tile_idx=tile_idx,
        #         logical_qubits=sorted(ql),
        #         global_to_local_map=global_to_local,
        #         gates=gates
        #     )
        #     tile_plan.append(tile)



        # new changes
        for tile_idx, (ql_set, gates) in enumerate(raw_tiles):
            # Keep QL as a deterministic ordered list.
            # (You can choose sorted(...) or preserve insertion order if you want;
            # sorted is fine for now.)
            ql = sorted(ql_set)

            # Define desired layout: tile's QL occupy bit positions 0..|QL|-1
            layout = _build_tile_layout(ql, num_qubits)

            # Since we force QL into the lowest bit positions, local bit index = position in ql
            global_to_local = {gq: i for i, gq in enumerate(ql)}

            tile = Tile(
                tile_idx=tile_idx,
                logical_qubits=ql,
                global_to_local_map=global_to_local,
                layout=layout,
                gates=gates,
            )
            tile_plan.append(tile)



        return tile_plan

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        num_qubits = len(dag.qubits)
        Q = set(range(num_qubits))

        qindex = {q: i for i, q in enumerate(dag.qubits)}

        # 1) Get gates in topological order
        C_list: list[DAGOpNode] = list(dag.topological_op_nodes())

        # 2) Compute depths manually (no node_depth / level APIs)
        last_depth = {i: -1 for i in range(num_qubits)}
        node_depth: dict[DAGOpNode, int] = {}

        for node in C_list:
            qubit_indices = [qindex[q] for q in node.qargs]
            if qubit_indices:
                d = max(last_depth[i] for i in qubit_indices) + 1
            else:
                d = 0
            node_depth[node] = d
            for i in qubit_indices:
                last_depth[i] = d

        # 3) Sort by depth (ascending) for Algorithm 1
        C_list.sort(key=lambda n: node_depth[n])

        # 4) Build T_map
        T_map = {
            node: {qindex[q] for q in node.qargs}
            for node in C_list
        }

        # 5) Run flat tiling + LQR
        g_schedule, M = self.flat_tiling(Q, C_list, T_map, self.nL)

        # Keep metadata for later GPU work
        self.property_set["flat_tiling_schedule"] = g_schedule
        self.property_set["flat_tiling_mapping"] = M

        tiling_plan = self.build_tiling_plan(g_schedule, M, num_qubits)
        self.property_set["flat_tiling_plan"] = tiling_plan

        # 6) Create a new empty DAG with same structure
        try:
            # should exist in most Qiskit versions
            new_dag = dag.copy_empty_like()
        except AttributeError:
            # fallback: recreate manually if needed
            new_dag = DAGCircuit()
            for qreg in dag.qregs.values():
                new_dag.add_qreg(qreg)
            for creg in dag.cregs.values():
                new_dag.add_creg(creg)

        # 7) Apply operations in g_schedule order
        for node in g_schedule:
            # node.op, node.qargs, node.cargs are reused directly
            new_dag.apply_operation_back(node.op, qargs=node.qargs, cargs=node.cargs)

        # Now LQR is a real TransformationPass
        print("ran through lqr")
        return new_dag


