#!/bin/bash
# Run the GTest unit tests from the nlcglib build directory produced by
# ci/build-{cuda,rocm}.sh (expects ./spack-env to exist).

set -xeuo pipefail

builddir=$(spack -e ./spack-env location -b nlcglib)

"${builddir}/unit_tests/nlcglib_unit_tests" --gtest_color=yes
