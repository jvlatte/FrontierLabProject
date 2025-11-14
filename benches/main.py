import argparse
from qiskit import QuantumCircuit, transpile    
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT
import time
import numpy as np
import random
import datetime
from pathlib import Path
import csv
from typing import Dict, List, Tuple, Optional


from custom_logging import _ensure_csv, _env_info, _append_csv

import os
try:
    import psutil
except Exception:
    psutil = None
try:
    import pynvml
    pynvml.nvmlInit()
except Exception:
    print("pynvml not available")
    pynvml = None


