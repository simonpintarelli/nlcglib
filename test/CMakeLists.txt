add_executable(test test.cpp)
NLCGLIB_SETUP_TARGET(test)

add_executable(test3 test3.cpp)
NLCGLIB_SETUP_TARGET(test3)

add_executable(test_traits test_traits.cpp)
NLCGLIB_SETUP_TARGET(test_traits)

add_executable(test_kokkos test_kokkos.cpp)
NLCGLIB_SETUP_TARGET(test_kokkos)

add_executable(test_view test_view.cpp)
NLCGLIB_SETUP_TARGET(test_view)

add_executable(test_gpu test_gpu.cpp)
NLCGLIB_SETUP_TARGET(test_gpu)

add_executable(test_stride_dvector test_stride_dvector.cpp)
NLCGLIB_SETUP_TARGET(test_stride_dvector)

add_executable(test_mpi_wrapper test_mpi_wrapper.cpp)
NLCGLIB_SETUP_TARGET(test_mpi_wrapper)

add_executable(test_solver test_solver.cpp)
NLCGLIB_SETUP_TARGET(test_solver)

add_executable(test_mvector_cuda test_mvector_cuda.cpp)
NLCGLIB_SETUP_TARGET(test_mvector_cuda)

add_executable(test_deep_copy test_deep_copy.cpp)
NLCGLIB_SETUP_TARGET(test_deep_copy)

add_executable(test_smearing test_smearing.cpp)
NLCGLIB_SETUP_TARGET(test_smearing)

add_executable(test_nvcc test_nvcc.cpp)
NLCGLIB_SETUP_TARGET(test_nvcc)

add_executable(test_check_alignment test_check_alignment.cpp)
NLCGLIB_SETUP_TARGET(test_check_alignment)
