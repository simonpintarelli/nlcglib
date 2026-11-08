#!/bin/bash

set -xeuo pipefail

export SPACK_SYSTEM_CONFIG_PATH=/user-environment/config

# make sure we keep the stage directory
spack config --scope=spack add config:build_stage:/dev/shm/spack-stage
# we might need to install dependencies too, e.g. kokkos in case of API changes
spack config --scope=spack add config:install_tree:root:/dev/shm/spack-stage

spack env create -d ./spack-env
spack -e ./spack-env add cray-mpich@9.1.0+cuda

spack -e ./spack-env config add "packages:all:variants:[cuda_arch=${CUDA_ARCH},+cuda]"

# keep -Werror instead of spack stripping it (SPEC sets cxxflags for nlcglib).
# -Werror=all-warnings is routed to nvcc by nvcc_wrapper, so that device-code diagnostics
# (e.g. #20013 host function called from __host__ __device__) fail the build.
spack -e ./spack-env config add "config:flags:keep_werror:all"

spack -e ./spack-env add "$SPEC"

# build nlcglib from source
spack -e ./spack-env develop -p $PWD nlcglib@develop

# display spack.yaml
cat ./spack-env/spack.yaml

spack -e ./spack-env concretize
spack -e ./spack-env install

# the tar pipe below expects a relative path
tar -cf builddir.tar /dev/shm/spack-stage
