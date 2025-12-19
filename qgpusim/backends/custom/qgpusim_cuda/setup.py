import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """A CMake-based extension module."""
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Build extension using CMake."""
    
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Ensure output directory exists
        os.makedirs(extdir, exist_ok=True)
        
        # CMake configuration arguments
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE=Release",
        ]
        
        # Build arguments
        build_args = ["--config", "Release"]
        
        # Use parallel build
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            build_args += ["--", "-j4"]
        
        # Create build directory
        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)
        
        # Run CMake configure
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=build_temp
        )
        
        # Run CMake build
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=build_temp
        )


setup(
    name="qgpusim_cuda",
    version="0.1.0",
    author="Your Name",
    description="CUDA-accelerated quantum gate simulation using pybind11",
    long_description="",
    ext_modules=[CMakeExtension("qgpusim_cuda", sourcedir=".")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        "numpy",
    ],
)
