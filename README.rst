Nlcglib is a plugin for sirius providing robust wave-function optimization to q-e-sirius_.

.. _q-e-sirius: https://github.com/electronic-structure/q-e-sirius

Installation
============

Q-e-sirius with the nlcglib plugin can be installed via spack_. Please use the
package files provided in this repository (in ``./repo``), as described below.

.. _spack: https://github.com/spack/spack


.. code:: bash

   git clone -b develop https://github.com/simonpintarelli/nlcglib
   # add the spack-repo
   spack repo add nlcglib/repo
   # build sirius/nlcglib with cuda enabled
   spack install q-e-sirius@develop-ristretto%gcc ^sirius+cuda+openmp+nlcglib
   #  build sirius/nlcglib without cuda
   spack install q-e-sirius@develop-ristretto%gcc ^sirius~cuda+openmp+nlcglib


Input parameters
================

Nlcglib specific settings are specified in the namelist `DIRECT_MINIMIZATION` in the QuantumESPRESSO input
file. It must be specified after the ELECTRONS namelist. The robust
wave-function optimization is run after the SCF loop, taking the last iteration
as starting guess. It is recommended to do at least 10 scf (=electron_maxstep)
iterations to obtain a good initial guess.

.. code::

   &ELECTRONS
   ...
   /
   &DIRECT_MINIMIZATION
      nlcg_maxiter = 300
      nlcg_conv_thr = 1e-9
      nlcg_restart = 10
      nlcg_bt_step_length = 0.1
      nlcg_pseudo_precond = 0.3
      nlcg_processing_unit 'none' | 'cpu' | 'gpu' '# default=none, i.e. will run on gpu if there is cuda device
   /

In most cases, only `nlcg_maxiter`, `nlcg_restart` need to be set.
`nlcg_conv_thr` is 1e-9 by default, and is equivalent to the `conv_thr` in the Electrons section of QE.

In order to enable the robust optimization an empty namelist can be inserted in the QE input file after the ELECTRONS namemlist:

.. code::

   &ELECTRONS
   ...
   /
   &DIRECT_MINIMIZATION
   /

In this case the default settings will be used (300 iterations max, cg-restart 10, threshold 1e-9).

Currently Gaussian, Fermi-Dirac broadening is supported. The support for Methfessel-Paxton and Marzari-Vanderbilt smearing is experimental.

References
==========

- Marzari, N., Vanderbilt, D., & Payne, M. C., Ensemble Density-Functional
  Theory for Ab Initio Molecular Dynamics of Metals and Finite-Temperature
  Insulators. , 79(7), 1337–1340. http://dx.doi.org/10.1103/PhysRevLett.79.1337
- Freysoldt, C., Boeck, S., & Neugebauer, J. Direct minimization technique
  for metals in density functional theory. , 79(24), 241103.
  http://dx.doi.org/10.1103/PhysRevB.79.241103
