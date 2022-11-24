# Copyright 2013-2022 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import os

from spack.package import *


class Sirius(CMakePackage, CudaPackage, ROCmPackage):
    """Domain specific library for electronic structure calculations"""

    homepage = "https://github.com/simonpintarelli/SIRIUS"
    url      = "https://github.com/simonpintarelli/SIRIUS/archive/v6.1.5.tar.gz"
    list_url = "https://github.com/simonpintarelli/SIRIUS/releases"
    git      = "https://github.com/simonpintarelli/SIRIUS.git"

    maintainers = ["simonpintarelli", "haampie", "dev-zero", "AdhocMan", "toxa81"]

    version("develop", branch="develop")
    variant("shared", default=True, description="Build shared libraries")
    variant("openmp", default=True, description="Build with OpenMP support")
    variant(
        "boost_filesystem",
        default=False,
        description="Use Boost filesystem for self-consistent field method "
        "mini-app. Only required when the compiler does not "
        "support std::experimental::filesystem nor std::filesystem",
    )
    variant("fortran", default=False, description="Build Fortran bindings")
    variant("python", default=False, description="Build Python bindings")
    variant("memory_pool", default=True, description="Build with memory pool")
    variant("elpa", default=False, description="Use ELPA")
    variant("vdwxc", default=False, description="Enable libvdwxc support")
    variant("scalapack", default=False, description="Enable scalapack support")
    variant("magma", default=False, description="Enable MAGMA support")
    variant("nlcglib", default=False, description="enable robust wave function optimization")
    variant(
        "build_type",
        default="Release",
        description="CMake build type",
        values=("Debug", "Release", "RelWithDebInfo"),
    )
    variant("apps", default=True, description="Build applications")
    variant("tests", default=False, description="Build tests")
    variant("single_precision", default=False, description="Use single precision arithmetics")
    variant(
        "profiler", default=True, description="Use internal profiler to measure execution time"
    )

    depends_on("mpi")
    depends_on("gsl")
    depends_on("lapack")
    depends_on("fftw-api@3")
    depends_on("libxc@3.0.0:")
    depends_on("libxc@4.0.0:", when="@7.2.0:")
    depends_on("spglib")
    depends_on("hdf5+hl")
    depends_on("pkgconfig", type="build")

    # Python module
    depends_on("python", when="+python", type=("build", "run"))
    depends_on("python", when="@:6", type=("build", "run"))
    depends_on("py-numpy", when="+python", type=("build", "run"))
    depends_on("py-scipy", when="+python", type=("build", "run"))
    depends_on("py-h5py", when="+python", type=("build", "run"))
    depends_on("py-mpi4py", when="+python", type=("build", "run"))
    depends_on("py-pyyaml", when="+python", type=("build", "run"))
    depends_on("py-mpi4py", when="+python", type=("build", "run"))
    depends_on("py-voluptuous", when="+python", type=("build", "run"))
    depends_on("py-pybind11", when="+python", type=("build", "run"))
    extends("python", when="+python")

    depends_on("magma", when="+magma")
    depends_on("boost cxxstd=14 +filesystem", when="+boost_filesystem")

    depends_on("spfft@0.9.6: +mpi", when="@6.4.0:")
    depends_on("spfft@0.9.13:", when="@7.0.1:")
    depends_on("spfft+single_precision", when="+single_precision ^spfft")
    depends_on("spfft+cuda", when="+cuda ^spfft")
    depends_on("spfft+rocm", when="+rocm ^spfft")
    depends_on("spfft+openmp", when="+openmp ^spfft")

    depends_on("spla@1.1.0:", when="@7.0.2:")
    depends_on("spla+cuda", when="+cuda ^spla")
    depends_on("spla+rocm", when="+rocm ^spla")
    depends_on("spla+openmp", when="+openmp ^spla")

    depends_on("nlcglib@develop", when="+nlcglib")
    depends_on("nlcglib+rocm@develop", when="+nlcglib+rocm")
    depends_on("nlcglib~cuda@develop", when="+nlcglib~cuda")

    depends_on("libvdwxc@0.3.0:+mpi", when="+vdwxc")

    depends_on("scalapack", when="+scalapack")

    depends_on("rocblas", when="+rocm")
    depends_on("rocsolver", when="+rocm")

    # FindHIP cmake script only works for < 4.1
    depends_on("hip@:4.0", when="@:7.2.0 +rocm")

    conflicts("+shared", when="@6.3.0:6.4")
    conflicts("+boost_filesystem", when="~apps")
    conflicts("^libxc@5.0.0")  # known to produce incorrect results
    conflicts("+single_precision", when="@:7.2.4")
    conflicts("+scalapack", when="^cray-libsci")

    # Propagate openmp to blas
    depends_on("openblas threads=openmp", when="+openmp ^openblas")
    depends_on("amdblis threads=openmp", when="+openmp ^amdblis")
    depends_on("blis threads=openmp", when="+openmp ^blis")
    depends_on("intel-mkl threads=openmp", when="+openmp ^intel-mkl")

    depends_on("elpa+openmp", when="+elpa+openmp")
    depends_on("elpa~openmp", when="+elpa~openmp")

    depends_on("eigen@3.4.0:", when="@7.3.2: +tests")

    depends_on("costa+shared", when="@7.3.2:")

    @property
    def libs(self):
        libraries = []

        if "@6.3.0:" in self.spec:
            libraries += ["libsirius"]

            return find_libraries(
                libraries, root=self.prefix, shared="+shared" in self.spec, recursive=True
            )
        else:
            if "+fortran" in self.spec:
                libraries += ["libsirius_f"]

            if "+cuda" in self.spec:
                libraries += ["libsirius_cu"]

            return find_libraries(
                libraries, root=self.prefix, shared="+shared" in self.spec, recursive=True
            )

    def cmake_args(self):
        spec = self.spec

        args = [
            self.define_from_variant("USE_OPENMP", "openmp"),
            self.define_from_variant("USE_ELPA", "elpa"),
            self.define_from_variant("USE_MAGMA", "magma"),
            self.define_from_variant("USE_NLCGLIB", "nlcglib"),
            self.define_from_variant("USE_VDWXC", "vdwxc"),
            self.define_from_variant("USE_MEMORY_POOL", "memory_pool"),
            self.define_from_variant("USE_SCALAPACK", "scalapack"),
            self.define_from_variant("CREATE_FORTRAN_BINDINGS", "fortran"),
            self.define_from_variant("CREATE_PYTHON_MODULE", "python"),
            self.define_from_variant("USE_CUDA", "cuda"),
            self.define_from_variant("USE_ROCM", "rocm"),
            self.define_from_variant("BUILD_TESTING", "tests"),
            self.define_from_variant("BUILD_APPS", "apps"),
            self.define_from_variant("BUILD_SHARED_LIBS", "shared"),
            self.define_from_variant("USE_FP32", "single_precision"),
            self.define_from_variant("USE_PROFILER", "profiler"),
        ]

        lapack = spec["lapack"]
        blas = spec["blas"]

        args.extend(
            [
                self.define("LAPACK_FOUND", "true"),
                self.define("LAPACK_LIBRARIES", lapack.libs.joined(";")),
                self.define("BLAS_FOUND", "true"),
                self.define("BLAS_LIBRARIES", blas.libs.joined(";")),
            ]
        )

        if "+scalapack" in spec and "^cray-libsci" not in spec:
            args.extend(
                [
                    self.define("SCALAPACK_FOUND", "true"),
                    self.define("SCALAPACK_INCLUDE_DIRS", spec["scalapack"].prefix.include),
                    self.define("SCALAPACK_LIBRARIES", spec["scalapack"].libs.joined(";")),
                ]
            )

        if "^cray-libsci" in spec:
            args.append(self.define("USE_CRAY_LIBSCI", "ON"))

        if spec["blas"].name in ["intel-mkl", "intel-parallel-studio"]:
            args.append(self.define("USE_MKL", "ON"))

        if "+elpa" in spec:
            elpa_incdir = os.path.join(spec["elpa"].headers.directories[0], "elpa")
            args.append(self.define("ELPA_INCLUDE_DIR", elpa_incdir))

        if "+cuda" in spec:
            cuda_arch = spec.variants["cuda_arch"].value
            if cuda_arch[0] != "none":
                # Specify a single arch directly
                if "@:6" in spec:
                    args.append(
                        self.define("CMAKE_CUDA_FLAGS", "-arch=sm_{0}".format(cuda_arch[0]))
                    )

                # Make SIRIUS handle it
                else:
                    args.append(self.define("CUDA_ARCH", ";".join(cuda_arch)))

        if "+rocm" in spec:
            archs = ",".join(self.spec.variants["amdgpu_target"].value)
            args.extend(
                [
                    self.define("HIP_ROOT_DIR", spec["hip"].prefix),
                    self.define("HIP_HCC_FLAGS", "--amdgpu-target={0}".format(archs)),
                    self.define("HIP_CXX_COMPILER", self.spec["hip"].hipcc),
                ]
            )

        return args
