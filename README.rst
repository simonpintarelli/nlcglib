NLCGLIB
=======

nlcglib is relying on kokkos_ and is thus needs to be compiled with clang or nvcc_wrapper from kokkos.

.. _kokkos: https://www.github.com/kokkos/kokkos

.. code:: bash

          git clone https://github.com/simonpintarelli/nlcglib.git
          mkdir -p build && cd build
          CXX=nvcc_wrapper cmake ../build
