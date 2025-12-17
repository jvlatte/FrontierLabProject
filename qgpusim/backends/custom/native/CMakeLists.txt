cmake_minimum_required(VERSION 3.22)
project(qgpusim_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Build a Python extension with pybind11
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(qgpusim_cuda
  src/bindings.cpp
  src/statevector.cu
  src/kernels.cu
)

target_include_directories(qgpusim_cuda PRIVATE include)

set_target_properties(qgpusim_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# If you want a specific GPU arch, uncomment and set it (example: A100=80, V100=70)
# set_property(TARGET qgpusim_cuda PROPERTY CUDA_ARCHITECTURES 80)

# Some toolchains need this to compile .cu in a shared module cleanly
target_compile_definitions(qgpusim_cuda PRIVATE
  PYBIND11_DETAILED_ERROR_MESSAGES=1
)

# On some clusters, you might need:
# target_link_libraries(qgpusim_cuda PRIVATE cudart)
