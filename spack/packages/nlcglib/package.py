# Copyright 2013-2022 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack.package import *


class Nlcglib(CMakePackage, CudaPackage,  ROCmPackage):
    """Nonlinear CG methods for wave-function optimization in DFT."""

    homepage = "https://github.com/simonpintarelli/nlcglib"
    git = "https://github.com/simonpintarelli/nlcglib.git"
    url = "https://github.com/simonpintarelli/nlcglib/archive/v0.9.tar.gz"

    maintainers = ["simonpintarelli"]

    version("develop", branch="develop")

    variant("openmp", default=True)
    variant("tests", default=False)
    variant("build_type",
            default="Release",
            description="CMake build type",
            values=("Debug", "Release", "RelWithDebInfo"),
            )

    with when('+rocm'):
        variant("magma", default=True, description="Use magma eigenvalue solver (AMDGPU)")
        depends_on("magma+rocm", when="+magma+rocm")

    depends_on("cmake@3.18:", type="build")
    depends_on("mpi")
    depends_on("lapack")

    depends_on("kokkos")
    depends_on("kokkos+openmp", when="+openmp")

    depends_on("kokkos+cuda+cuda_lambda+wrapper", when="+cuda%gcc")
    depends_on("kokkos+cuda", when="+cuda")

    # rocm dependencies
    depends_on("kokkos+rocm", when="+rocm")
    depends_on("rocblas", when="+rocm")
    depends_on("rocsolver", when="+rocm")

    depends_on("googletest", type="build", when="+tests")
    depends_on("nlohmann-json")

    def cmake_args(self):
        options = [
            self.define_from_variant("USE_OPENMP", "openmp"),
            self.define_from_variant("BUILD_TESTS", "tests"),
            self.define_from_variant("USE_ROCM", "rocm"),
            self.define_from_variant("USE_MAGMA", "magma"),
            self.define_from_variant("USE_CUDA", "cuda"),
        ]

        if self.spec["blas"].name in ["intel-mkl", "intel-parallel-studio"]:
            options.append("-DLAPACK_VENDOR=MKL")
        elif self.spec["blas"].name in ["openblas"]:
            options.append("-DLAPACK_VENDOR=OpenBLAS")
        else:
            raise Exception("blas/lapack must be either openblas or mkl.")

        if "+cuda%gcc" in self.spec:
            options.append(
                "-DCMAKE_CXX_COMPILER=%s" % self.spec["kokkos-nvcc-wrapper"].kokkos_cxx
            )

        if "+cuda" in self.spec:
            cuda_arch = self.spec.variants["cuda_arch"].value
            if cuda_arch[0] != "none":
                options += ["-DCMAKE_CUDA_FLAGS=-arch=sm_{0}".format(cuda_arch[0])]

        if "+rocm" in self.spec:
            options.append(self.define(
                "CMAKE_CXX_COMPILER", self.spec["hip"].hipcc))

        return options
