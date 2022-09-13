add_executable(test test.cpp)
NLCGLIB_SETUP_TARGET(test)

add_executable(test3 test3.cpp)
NLCGLIB_SETUP_TARGET(test3)

add_executable(test_traits test_traits.cpp)
NLCGLIB_SETUP_TARGET(test_traits)

add_executable(test_kokkos test_kokkos.cpp)
NLCGLIB_SETUP_TARGET(test_kokkos)

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

# add_executable(test_check_alignment test_check_alignment.cpp)
# NLCGLIB_SETUP_TARGET(test_check_alignment)

add_executable(test_buffer test_buffer.cpp)
NLCGLIB_SETUP_TARGET(test_buffer)

if(USE_CUDA OR USE_ROCM)
add_executable(test_dvector_mirror_view_and_copy test_dvector_mirror_view_and_copy.cpp)
NLCGLIB_SETUP_TARGET(test_dvector_mirror_view_and_copy)

add_executable(test_lambda_device test_lambda_device.cpp)
NLCGLIB_SETUP_TARGET(test_lambda_device)

add_executable(test_gpu test_gpu.cpp)
NLCGLIB_SETUP_TARGET(test_gpu)
endif()

add_executable(test_utils test_utils.cpp)
NLCGLIB_SETUP_TARGET(test_utils)
