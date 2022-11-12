add_executable(test test.cpp)
target_link_libraries(test PRIVATE nlcglib_core)

add_executable(test3 test3.cpp)
target_link_libraries(test3 PRIVATE nlcglib_core)

add_executable(test_traits test_traits.cpp)
target_link_libraries(test_traits PRIVATE nlcglib_core)

add_executable(test_kokkos test_kokkos.cpp)
target_link_libraries(test_kokkos PRIVATE nlcglib_core)

add_executable(test_stride_dvector test_stride_dvector.cpp)
target_link_libraries(test_stride_dvector PRIVATE nlcglib_core)

add_executable(test_mpi_wrapper test_mpi_wrapper.cpp)
target_link_libraries(test_mpi_wrapper PRIVATE nlcglib_core)

add_executable(test_solver test_solver.cpp)
target_link_libraries(test_solver PRIVATE nlcglib_core)

add_executable(test_mvector_cuda test_mvector_cuda.cpp)
target_link_libraries(test_mvector_cuda PRIVATE nlcglib_core)

add_executable(test_deep_copy test_deep_copy.cpp)
target_link_libraries(test_deep_copy PRIVATE nlcglib_core)

add_executable(test_smearing test_smearing.cpp)
target_link_libraries(test_smearing PRIVATE nlcglib_core)

# add_executable(test_check_alignment test_check_alignment.cpp)
# target_link_libraries(test_check_alignment PRIVATE nlcglib_core)

add_executable(test_buffer test_buffer.cpp)
target_link_libraries(test_buffer PRIVATE nlcglib_core)

if(USE_CUDA OR USE_ROCM)
add_executable(test_dvector_mirror_view_and_copy test_dvector_mirror_view_and_copy.cpp)
target_link_libraries(test_dvector_mirror_view_and_copy PRIVATE nlcglib_core)

add_executable(test_lambda_device test_lambda_device.cpp)
target_link_libraries(test_lambda_device PRIVATE nlcglib_core)

add_executable(test_gpu test_gpu.cpp)
target_link_libraries(test_gpu PRIVATE nlcglib_core)
endif()

add_executable(test_utils test_utils.cpp)
target_link_libraries(test_utils PRIVATE nlcglib_core)
