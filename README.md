# GPU-Accelerated Quantum Circuit Simulator with Lazy Qubit Reordering

A high-performance, statevector-based quantum circuit simulator that combines **lazy qubit reordering at the transpilation level** with a **custom CUDA-accelerated GPU backend**.  
The project targets one of the main bottlenecks in large-scale quantum simulation: **global statevector permutations and CPU-side kernel launch overhead**.

This simulator is designed for **research and performance benchmarking**, with a focus on scalable execution on modern NVIDIA GPUs.

The codebase is structured for experimentation and extension:
- swap transpilation strategies
- swap execution backends
- run controlled benchmarks
- inspect low-level GPU behavior (launches, memory, replays)


At a high level, the simulator is split into two layers:

1. **Transpilation**
   - Tracks logical → physical qubit layouts
   - Groups gates under compatible layouts
   - Partitions circuits into tiles of local qubits

2. **Simulation/Backend**
   - Executes tiles on CPU or GPU
   - Uses persistent statevector buffers
   - I've implemented 3 different methods to run each tile in the backend

## How to run the program
Make sure python environment has required libraries to run program.
Full list of libraries explicitly installed are shown in the "environment.yml"

Typical command I use is:
```
python3 -B -m benches.run_benchmarks
```

-B make sures no pycache is created in order to fully analyze raw results and data
 
As program runs, results are appended to the csv file located in the results folder. To create graphs and plots, run the command:

```
python3 -B -m benches.plotting
```

To compile C++/CUDA file, go into `build` folder (/qgpusim/backends/custom/qgpusim_cuda/build) and run the MakeFile (`make` should work out)
```
cd .../qgpusim/backends/custom/qgpusim_cuda/build && make
```


## Code details
To test different parameter values, tweak the variable values `params` directly under `main()` function.
- `"mode": ["compare", "gpu"]`; "compare" will compare both CPU and GPU versions, whereas "GPU" only uses GPU
- `"tasks": ["statevector", "sampling"]`; project is mostly statevector-based, so other value is most likely not needed
- `"circuit": ["random", "ghz", "qft"]`; as of now, only these 3 circuits are supported, feel free to add and test more circuits out!
- `"nqubits": [int]`; number of qubits
- `"depth": [int]` depth of circuit
- `"shots"`: (don't worry about it, it's for sampling tasks)
- `"repeats": [int]`; number of repeats runs for that certain combination of parameters
- `"nL": [int]`; number of local qubits

The variable `valid_transpiler_backend_pairs` is created so that it only runs either the baseline transpiler and baseline backend (both provided by Qiskit's Aer library) or our custom transpiler and custom backend. Essentially it will not run a simulation like baseline tranpspiler and custom backend, as that does not work out nicely

Simulator is mainly split into two parts: transpilation process and simulation process.

### Transpiler
The transpiler utilizes **Lazy Qubit Reordering** and mostly follows the alogrithms listed on the paper (only Flat-tiling algorithm as the code was meant to run one GPU only)

After transpilation, it outputs a tile schedule, with details about the tile data structure shown in the passes.py file under transpiler folder (/qgpusum/transpiler/passes.py)

### Simulation
Custom backend utilizes CUDA to execute each tile.
There are 3 different strategies to execute the CUDA tiles, each of which can be used when changing the `tile_mode` parameter in the  `run_custom_backend()` function

`tile_mode` parameter can these values: `"cooperative", "graphed", "sequential"`

`"graphed"` uses CUDA Graphs, as of right now seems to give best performance udner single-precision

`"cooperative"` creates a true single kernel to run a tile; (tried to get this to work, but decided to use CUDA graphs for time purposes)

`"sequential"` each kernel is launched per gate (so gate-by-gate execution rather than tile-by-tile)

### Correctness
Once simulation ends and comparison is enabled, it compares the statevector with a reference statevector and compares the overlap

## Important notes
Under the bindings.cpp file, there are two different classes for the statevector; `Statevector` and `StatevectorF32`; as of right now, our implementation seems to perform better than Qiskit's Aer when running single-precision (`StatevectorF32`), but then performs worse when running double-precision. That's one thing to work on in the future.

---

