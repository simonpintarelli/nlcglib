cmake_minimum_required(VERSION 3.21)
project(
  nlcglib
  LANGUAGES CXX
  VERSION 1.4.3)

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(BUILD_TESTS "Build tests" OFF)
option(USE_OPENMP "Use OpenMP" ON)
option(USE_CUDA "Use CUDA" OFF)
option(USE_ROCM "Use AMD GPUs" OFF)
option(USE_MAGMA "Use the MAGMA eigensolver for AMD GPUs" OFF)
option(USE_GPU_DIRECT "Use GPU-aware MPI" OFF)
option(USE_TX "Enable vendor profiling markers (NVTX for CUDA, roctx for ROCm)"
       OFF)

set(LAPACK_VENDOR
    "AUTO"
    CACHE STRING "LAPACK provider")
set_property(CACHE LAPACK_VENDOR PROPERTY STRINGS AUTO OpenBLAS MKLONEAPI MKL
                                          CRAY_LIBSCI)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      "Release"
      CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release
                                               RelWithDebInfo)
endif()

if(USE_CUDA AND USE_ROCM)
  message(FATAL_ERROR "USE_CUDA and USE_ROCM are mutually exclusive")
endif()
if(USE_MAGMA AND NOT USE_ROCM)
  message(FATAL_ERROR "USE_MAGMA requires USE_ROCM")
endif()
if((USE_GPU_DIRECT OR USE_TX) AND NOT (USE_CUDA OR USE_ROCM))
  message(FATAL_ERROR "USE_GPU_DIRECT and USE_TX require a GPU backend")
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
list(PREPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(Kokkos REQUIRED CONFIG)
find_package(nlohmann_json 3.2.0 REQUIRED CONFIG)
find_package(fmt REQUIRED CONFIG)

if(USE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  kokkos_check(DEVICES OpenMP)
else()
  kokkos_check(DEVICES Serial)
endif()

if(LAPACK_VENDOR STREQUAL "AUTO")
  if(DEFINED MKLROOT OR DEFINED ENV{MKLROOT})
    set(_lapack_vendor "MKLONEAPI")
  else()
    set(_lapack_vendor "OpenBLAS")
  endif()
elseif(LAPACK_VENDOR STREQUAL "MKL")
  # Alias: both spellings select oneAPI MKL via MKLConfig.cmake.
  set(_lapack_vendor "MKLONEAPI")
else()
  set(_lapack_vendor "${LAPACK_VENDOR}")
endif()

if(_lapack_vendor STREQUAL "OpenBLAS")
  find_package(OpenBLAS REQUIRED MODULE)
elseif(_lapack_vendor STREQUAL "MKLONEAPI")
  set(MKL_INTERFACE
      "lp64"
      CACHE STRING "MKL integer interface")
  set(MKL_THREADING
      "sequential"
      CACHE STRING "MKL threading layer")
  set(MKL_LINK
      "dynamic"
      CACHE STRING "MKL linkage (dynamic or static)")
  if(NOT MKL_INTERFACE STREQUAL "lp64")
    message(FATAL_ERROR "nlcglib requires the MKL LP64 interface")
  endif()
  find_package(MKL REQUIRED CONFIG)
elseif(_lapack_vendor STREQUAL "CRAY_LIBSCI")
  set(SCI_USE_OPENMP ${USE_OPENMP})
  find_package(SCI REQUIRED MODULE)
else()
  message(FATAL_ERROR "Unsupported LAPACK_VENDOR: ${LAPACK_VENDOR}")
endif()

if(USE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  kokkos_check(DEVICES Cuda)
  if(USE_TX AND NOT TARGET CUDA::nvtx3)
    message(
      FATAL_ERROR
        "USE_TX with CUDA requires the CUDA::nvtx3 target (CMake 3.25 or newer)"
    )
  endif()
elseif(USE_ROCM)
  find_package(hip REQUIRED CONFIG)
  find_package(rocblas REQUIRED CONFIG)
  find_package(rocsolver REQUIRED CONFIG)
  kokkos_check(DEVICES HIP)
  if(USE_TX)
    find_package(rocprofiler-sdk-roctx REQUIRED CONFIG)
  endif()
endif()
if(USE_MAGMA)
  find_package(hipblas REQUIRED CONFIG)
  find_package(hipsparse REQUIRED CONFIG)
  find_package(MAGMA REQUIRED MODULE)
endif()

add_subdirectory(src)

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
  add_subdirectory(unit_tests)
endif()

set(nlcglib_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/nlcglib")
configure_package_config_file(
  cmake/nlcglib-config.cmake.in "${PROJECT_BINARY_DIR}/nlcglib-config.cmake"
  INSTALL_DESTINATION "${nlcglib_cmake_dir}")
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/nlcglib-config-version.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

install(
  EXPORT nlcglib_targets
  FILE nlcglib-targets.cmake
  NAMESPACE nlcglib::
  DESTINATION "${nlcglib_cmake_dir}")
install(FILES "${PROJECT_BINARY_DIR}/nlcglib-config.cmake"
              "${PROJECT_BINARY_DIR}/nlcglib-config-version.cmake"
        DESTINATION "${nlcglib_cmake_dir}")
if(NOT BUILD_SHARED_LIBS)
  if(LAPACK_VENDOR STREQUAL "OpenBLAS")
    install(FILES cmake/modules/FindOpenBLAS.cmake
            DESTINATION "${nlcglib_cmake_dir}/modules")
  elseif(LAPACK_VENDOR STREQUAL "CRAY_LIBSCI")
    install(FILES cmake/modules/FindSCI.cmake
            DESTINATION "${nlcglib_cmake_dir}/modules")
  endif()
  if(USE_MAGMA)
    install(FILES cmake/modules/FindMAGMA.cmake
            DESTINATION "${nlcglib_cmake_dir}/modules")
  endif()
endif()
