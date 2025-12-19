cmake_minimum_required(VERSION 3.18)
project(qgpusim_cuda LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Get pybind11 cmake dir from Python
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED)

# Set CUDA architectures (adjust based on your GPU)
# Common architectures: 70 (V100), 75 (T4), 80 (A100), 86 (RTX 3000), 89 (RTX 4000), 90 (H100)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

# Collect source files
set(CUDA_SOURCES
    src/kernels.cu
)

set(CPP_SOURCES
    src/bindings.cpp
)

# Create the pybind11 module
pybind11_add_module(qgpusim_cuda
    ${CUDA_SOURCES}
    ${CPP_SOURCES}
)

# Include directories
target_include_directories(qgpusim_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link CUDA runtime
target_link_libraries(qgpusim_cuda PRIVATE
    CUDA::cudart
)

# Compiler flags
target_compile_options(qgpusim_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -O3
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -fPIC
    >
)

# Set the output name and install location
set_target_properties(qgpusim_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Install the module to the native subdirectory
install(TARGETS qgpusim_cuda
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../native
)
