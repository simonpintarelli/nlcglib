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
                  "tol": 1e-9,
                  "restart": 10,
                  "maxiter": 300,
                  "processing_unit": "gpu"
              },
              "parameters": {"num_dft_iter": 5}
          }

`processing_unit` is either `gpu` or `cpu`, if not set, the SIRIUS default (`gpu` in QE-SIRIUS) will be used.

The minimization will stop if the tolerance (tol) is reached, e.g. when the
modulus of the descent (slope) along the search direction is less than tol. It's
roughly proportional to the error in the total energy.


References
==========

- Marzari, N., Vanderbilt, D., & Payne, M. C., Ensemble Density-Functional
  Theory for Ab Initio Molecular Dynamics of Metals and Finite-Temperature
  Insulators. , 79(7), 1337–1340. http://dx.doi.org/10.1103/PhysRevLett.79.1337
- Freysoldt, C., Boeck, S., & Neugebauer, J. Direct minimization technique
  for metals in density functional theory. , 79(24), 241103.
  http://dx.doi.org/10.1103/PhysRevB.79.241103
