# Native CUDA extension module
# This module will contain the compiled qgpusim_cuda.so after building

try:
    from .qgpusim_cuda import Statevector, apply_1q_gate, apply_2q_gate
    __all__ = ["Statevector", "apply_1q_gate", "apply_2q_gate"]
except ImportError as e:
    import warnings
    warnings.warn(
        f"Could not import native CUDA module: {e}. "
        "Please build the extension with: cd qgpusim_cuda && pip install -e ."
    )
    raise
