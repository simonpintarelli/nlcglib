set(nlcglib_private_libraries Kokkos::kokkos fmt::fmt
                              $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>)
set(nlcglib_private_definitions)

if(_lapack_vendor STREQUAL "OpenBLAS")
  list(APPEND nlcglib_private_libraries OpenBLAS::LAPACKE)
elseif(_lapack_vendor STREQUAL "MKL")
  list(APPEND nlcglib_private_libraries MKL::MKL)
  list(APPEND nlcglib_private_definitions __USE_MKL)
else()
  list(APPEND nlcglib_private_libraries SCI::sci)
endif()

if(USE_OPENMP)
  list(APPEND nlcglib_private_libraries OpenMP::OpenMP_CXX)
  list(APPEND nlcglib_private_definitions __USE_OPENMP)
endif()
if(USE_GPU_DIRECT)
  list(APPEND nlcglib_private_definitions __NLCGLIB__GPU_DIRECT)
endif()
if(USE_TX)
  list(APPEND nlcglib_private_definitions __NLCGLIB__TX)
endif()
if(USE_CUDA)
  list(APPEND nlcglib_private_libraries CUDA::cudart CUDA::cublas
       CUDA::cusolver)
  list(APPEND nlcglib_private_definitions __NLCGLIB__CUDA)
  if(USE_TX)
    list(APPEND nlcglib_private_libraries CUDA::nvtx3)
  endif()
elseif(USE_ROCM)
  list(APPEND nlcglib_private_libraries hip::host roc::rocblas roc::rocsolver)
  list(APPEND nlcglib_private_definitions __NLCGLIB__ROCM)
  if(USE_TX)
    list(APPEND nlcglib_private_libraries
         rocprofiler-sdk-roctx::rocprofiler-sdk-roctx)
  endif()
endif()
if(USE_MAGMA)
  list(APPEND nlcglib_private_libraries MAGMA::MAGMA roc::hipblas
       roc::hipsparse)
  list(APPEND nlcglib_private_definitions __NLCGLIB__MAGMA)
endif()

add_library(nlcglib nlcglib.cpp utils/step_logger.cpp)
add_library(nlcglib::nlcglib ALIAS nlcglib)
if(USE_CUDA)
  target_sources(nlcglib PRIVATE la/cusolver.cpp)
endif()
if(USE_CUDA OR USE_ROCM)
  target_sources(nlcglib PRIVATE gpu/acc.cpp)
endif()
if(USE_MAGMA)
  target_sources(nlcglib PRIVATE la/magma.cpp)
endif()

target_compile_features(
  nlcglib
  PUBLIC cxx_std_11
  PRIVATE cxx_std_20)
target_compile_options(
  nlcglib
  PRIVATE "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra>")
target_compile_definitions(nlcglib PRIVATE ${nlcglib_private_definitions})
target_include_directories(
  nlcglib
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/nlcglib>
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(
  nlcglib
  PUBLIC MPI::MPI_CXX
  PRIVATE ${nlcglib_private_libraries})
set_target_properties(
  nlcglib
  PROPERTIES POSITION_INDEPENDENT_CODE ON
             VERSION ${PROJECT_VERSION}
             SOVERSION ${PROJECT_VERSION_MAJOR})

# Internal headers contain templates, so tests need the implementation dependencies too.
# This build-only target is never part of the installed link interface.
if(BUILD_TESTS)
  add_library(nlcglib_test_support INTERFACE)
  target_compile_features(nlcglib_test_support INTERFACE cxx_std_20)
  target_link_libraries(nlcglib_test_support
                        INTERFACE nlcglib ${nlcglib_private_libraries})
  target_include_directories(nlcglib_test_support
                             INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_definitions(nlcglib_test_support
                             INTERFACE ${nlcglib_private_definitions})
endif()

install(TARGETS nlcglib EXPORT nlcglib_targets)
install(FILES "${PROJECT_SOURCE_DIR}/include/nlcglib.hpp"
              "${PROJECT_SOURCE_DIR}/include/interface.hpp"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/nlcglib")
