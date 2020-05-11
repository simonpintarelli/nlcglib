NLCGLIB
=======

nlcglib is relying on kokkos_ and is thus needs to be compiled with clang or nvcc_wrapper from kokkos.

.. _kokkos: https://www.github.com/kokkos/kokkos

.. code:: bash

          git clone https://github.com/simonpintarelli/nlcglib.git
          mkdir -p build && cd build
          CXX=nvcc_wrapper cmake ../build

Input parameters
================

Parameters for the robust wave-function optimization are passed using the
sirius.json. The algorithm will first perform a couple of scf steps and then
start the direct solver (usually between 5 and 10).

.. code:: json

          {
              "nlcg": {
                  "T": 300,
                  "smearing": "FD",
                  "tol": 1e-9,
                  "restart": 10,
                  "maxiter": 100,
                  "processing_unit": "gpu"
              },
              "parameters": {"num_dft_iter": 5}
          }

`processing_unit` is either `gpu` or `cpu`, if not set, the SIRIUS default (`gpu` in QE-SIRIUS) will be used.

In the nlcg section, smearing can either be `FD` for Fermi-Dirac or `GS` for
Gaussian-Spline (the smearing set in the QE input will be ignored). `maxiter` is
for the difficult problems ~150 on average. CG will stop if the tolerance (tol) is
reached, e.g. when the modulus of the descent (slope) along the search direction is less
than tol. It's roughly proportional to the error in the total energy.

Current limitations
===================

Pseudo-potentials must be norm-conserving, wave-functions *must not* be distributed.
