from qiskit_aer import AerSimulator


def create_simulator(backend: str, task: str):
    """creates AerSimulator"""
    method = "automatic"
    if task == "statevector":
        method = "statevector"
    elif task == "sampling":
        method = "automatic"
    
    device_used = "CPU"
    if backend == "cpu":
        sim = AerSimulator(method=method)  # default is CPU
    elif backend == "gpu":
        try:
            sim = AerSimulator(method=method, device="GPU")
            device_used = "GPU"
        except TypeError:
            sim = AerSimulator(method=method)
            print("GPU backend not available, falling back to CPU.")
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return sim, device_used
