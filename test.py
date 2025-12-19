# import numpy as np
# from qgpusim.backends.custom.native.qgpusim_cuda import Statevector

# def test_identity():
#     sv = Statevector(4)
#     U = np.eye(2, dtype=np.complex128)
#     sv.apply_1q(0, U)
#     psi = sv.to_numpy()
#     assert np.allclose(psi[0], 1.0)
#     assert np.allclose(psi[1:], 0.0)

# test_identity()
# print("OK")
