add_executable(test_lapack test.cpp)
target_link_libraries(test_lapack PRIVATE nlcglib_test_support)

set(nlcglib_tests
    traits
    kokkos
    stride_dvector
    mpi_wrapper
    deep_copy
    smearing
    buffer
    utils)
if(USE_CUDA)
  list(APPEND nlcglib_tests mvector_cuda)
endif()
if(USE_CUDA OR USE_ROCM)
  list(APPEND nlcglib_tests dvector_mirror_view_and_copy lambda_device gpu
       solver)
endif()
foreach(nlcglib_test IN LISTS nlcglib_tests)
  add_executable(test_${nlcglib_test} test_${nlcglib_test}.cpp)
  target_link_libraries(test_${nlcglib_test} PRIVATE nlcglib_test_support)
endforeach()
