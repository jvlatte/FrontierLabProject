from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qgpusim.transpiler.passes import LazyQubitReordering  # adjust import

# qc = QuantumCircuit(4)
# qc.h(0)
# qc.cx(0, 1)
# qc.cx(1, 2)
# qc.cx(2, 3)
# qc.cx(0, 2)
# qc.cx(1, 3)

# # keep handle to pass instance
# lz = LazyQubitReordering(nL=2)

# pm = PassManager([lz])
# pm.run(qc)

# g = lz.property_set["flat_tiling_schedule"]
# M = lz.property_set["flat_tiling_mapping"]

# print("Schedule:")
# for i, node in enumerate(g):
#     print(
#         f"{i:2d}: {node.name} on {[q._index for q in node.qargs]} "
#         f"QL={sorted(M[node])}"
#     )



def make_test_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.barrier()
    qc.cx(0, 2)
    qc.cx(1, 3)
    return qc


def run_tiling_for_nL(nL: int):
    qc = make_test_circuit()

    lz = LazyQubitReordering(nL=nL)
    pm = PassManager([lz])
    pm.run(qc)

    g = lz.property_set["flat_tiling_schedule"]
    M = lz.property_set["flat_tiling_mapping"]

    stats = LazyQubitReordering.summarize_tiling(g, M)
    tiles = LazyQubitReordering.extract_tiles(g, M)

    print(f"\n=== nL = {nL} ===")
    print(f"num_tiles      = {stats['num_tiles']}")
    print(f"avg_tile_size  = {stats['avg_tile_size']:.2f}")
    print(f"max_tile_size  = {stats['max_tile_size']}")
    print(f"avg |Q_L|      = {stats['avg_Q_L_size']:.2f}")
    print(f"max |Q_L|      = {stats['max_Q_L_size']}")

    # Optional: print each tile for debugging
    for t_idx, (ql, gates) in enumerate(tiles):
        print(f"  Tile {t_idx}: QL={sorted(ql)}, gates={len(gates)}")
        for g_idx, node in enumerate(gates):
            print(f"    {g_idx}: {node.name} on {[q._index for q in node.qargs]}")


if __name__ == "__main__":
    for nL in [2, 3, 4]:
        run_tiling_for_nL(nL)
