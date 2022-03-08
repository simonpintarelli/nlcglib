# Copyright 2013-2022 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *


class Nlcglib(CMakePackage, CudaPackage):
    """Nonlinear CG methods for wave-function optimization in DFT."""

    homepage = "https://github.com/simonpintarelli/nlcglib"
    git = "https://github.com/simonpintarelli/nlcglib.git"
    url = "https://github.com/simonpintarelli/nlcglib/archive/v0.9.tar.gz"

    maintainers = ["simonpintarelli"]

    version('master', branch="master")
    version('develop', branch="develop")
    version('0.9', sha256="8d5bc6b85ee714fb3d6480f767e7f43e5e7d569116cf60e48f533a7f50a37a08")

    variant('wrapper', default=False, description="Use nvcc-wrapper for CUDA build")
    variant('openmp', default=True)
    variant('cuda', default=False)
    variant('tests', default=False)
    variant('build_type',
            default="Release",
            description="CMake build type",
            values=("Debug", "Release", "RelWithDebInfo"),
            )

    depends_on('mpi')
    depends_on('lapack')
    depends_on('kokkos')
    depends_on('kokkos+openmp', when='+openmp')
    depends_on('kokkos-nvcc-wrapper', when='+wrapper')
    depends_on('kokkos+cuda~cuda_relocatable_device_code+cuda_lambda+wrapper', when='+wrapper')
    depends_on('cmake@3.15:', type='build')
    depends_on('kokkos+cuda~cuda_relocatable_device_code+cuda_lambda+openmp+wrapper',
               when='+openmp+wrapper')
    depends_on('kokkos+cuda', when='+cuda')
    depends_on('googletest', type='build', when='+tests')
    depends_on('nlohmann-json')

    def cmake_args(self):
        options = []

        if '+openmp' in self.spec:
            options.append('-DUSE_OPENMP=On')
        else:
            options.append('-DUSE_OPENMP=Off')

        if self.spec['blas'].name in ['intel-mkl', 'intel-parallel-studio']:
            options.append('-DLAPACK_VENDOR=MKL')
        elif self.spec['blas'].name in ['openblas']:
            options.append('-DLAPACK_VENDOR=OpenBLAS')
        else:
            raise Exception('blas/lapack must be either openblas or mkl.')

        if '+tests' in self.spec:
            options.append('-DBUILD_TESTS=On')
        else:
            options.append('-DBUILD_TESTS=Off')

        if '+wrapper' in self.spec:
            options.append(
                '-DCMAKE_CXX_COMPILER=%s' % self.spec['kokkos-nvcc-wrapper'].kokkos_cxx
            )

        if '+cuda' in self.spec:
            options.append('-DUSE_CUDA=On')
            cuda_arch = self.spec.variants['cuda_arch'].value
            if cuda_arch[0] != 'none':
                options += ['-DCMAKE_CUDA_FLAGS=-arch=sm_{0}'.format(cuda_arch[0])]
        else:
            options.append('-DUSE_CUDA=Off')

        return options
