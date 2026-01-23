from qiskit_aer import AerSimulator
from typing import Optional, Tuple



def create_simulator(
    backend: str,
    task: str,
    precision: Optional[str] = None,   # default None -> let Aer decide
) -> Tuple[AerSimulator, str]:
    """
    Create a 'true baseline' AerSimulator: minimal options, no tuning knobs.
    Goal: represent what a normal user would do: pick method + device and go.
    """
    # Pick the method only (keep it simple and explicit)
    if task == "statevector":
        method = "statevector"
    elif task == "sampling":
        method = "automatic"
    else:
        raise ValueError(f"Unknown task: {task}")

    sim_options = {"method": method}

    # Let Aer defaults decide precision unless you need match precision exactly.
    # (If you want apples-to-apples vs your custom implementation, pass "single"/"double")
    if precision is not None:
        sim_options["precision"] = "single"

    device_used = "CPU"
    if backend == "cpu":
        sim = AerSimulator(**sim_options, precision="single")
    elif backend == "gpu":
        # Baseline GPU: only set device, no cuStateVec flag, no blocking, no parallel tweaks
        sim_options["device"] = "GPU"
        sim = AerSimulator(**sim_options, precision="single")
        device_used = "GPU"
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return sim, device_used
