import csv, os, platform

def _env_info():
    return {
        "host": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
        # # last two lines maybe not needed
        # "qiskit": getattr(qiskit, "__version__", "unknown"),
        # "qiskit_aer": getattr(qiskit_aer, "__version__", "unknown"),
    }

def _ensure_csv(path: str, fieldnames: list[str]):
    exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    if not exists:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

def _append_csv(path: str, row: dict, fieldnames: list[str]):
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)

