cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(nlcglib LANGUAGES CXX VERSION 0.8)

set(CMAKE_CXX_STANDARD 17)
# user variables
set(USE_OPENMP On CACHE BOOL "use OpenMP")
set(USE_CUDA Off CACHE BOOL "use cuda")
set(USE_ROCM Off CACHE BOOL "use amd gpus")
set(USE_MAGMA Off CACHE BOOL "use magma eigensolver for amd gpus")

set(BUILD_TESTS OFF CACHE BOOL "build tests")
set(LAPACK_VENDOR "OpenBLAS" CACHE STRING "lapack vendor")
set_property(CACHE LAPACK_VENDOR PROPERTY STRINGS "OpenBLAS" "MKL" "MKLONEAPI")

find_package(MPI REQUIRED COMPONENTS CXX)

if(USE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS "YES")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

include(cmake/nlcglib_macros.cmake)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

find_package(Kokkos)
if(USE_CUDA)
  find_package(CUDA REQUIRED)
  enable_language(CUDA)
  include(cmake/cudalibs_target.cmake)
endif()

if(USE_ROCM)
  # find_package(hip REQUIRED)
  # enable_language(HIP)
  include(cmake/rocmlibs_target.cmake)
endif()

if(USE_MAGMA)
  find_package(MAGMA REQUIRED)
endif()

find_package(MPI REQUIRED)

if(LAPACK_VENDOR MATCHES OpenBLAS)
  message("find package openblas")
  find_package(OpenBLAS REQUIRED)
<<<<<<< variant A
  add_library(my_lapack INTERFACE IMPORTED)
  set_target_properties(my_lapack PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${OpenBLAS_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${OpenBLAS_LIBRARIES}")
elseif(LAPACK_VENDOR STREQUAL MKL)
>>>>>>> variant B
  message("oblas includes: ${OpenBLAS_INCLUDE_DIRS}")
  if(NOT TARGET nlcg::cpu_lapack)
    add_library(nlcg::cpu_lapack INTERFACE IMPORTED)
    set_target_properties(nlcg::cpu_lapack PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${OpenBLAS_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${OpenBLAS_LIBRARIES}")
  endif()
elseif(LAPACK_VENDOR MATCHES MKL)
======= end
  message("LAPACK VENDOR MKL")
  find_package(MKL REQUIRED)
elseif(LAPACK_VENDOR STREQUAL MKLONEAPI)
  # set(MKL_THREADING gnu_thread)
  find_package(MKL REQUIRED NO_MODULE)
elseif(LAPACK_VENDOR STREQUAL CRAY_LIBSCI)
  message("LAPACK VENDOR Cray Libsci")
  find_package(SCI REQUIRED)
  if(NOT TARGET nlcg::cpu_lapack)
    add_library(nlcg::cpu_lapack INTERFACE IMPORTED)
    set_target_properties(my_lapack PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${SCI_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${SCI_LIBRARIES}")
  endif()
else()
  message(FATAL_ERROR "must specify a LAPACK_VENDOR")
endif()

find_package(nlohmann_json 3.2.0 REQUIRED)

add_subdirectory(src)

if(BUILD_TESTS)
  add_subdirectory(test)
  add_subdirectory(unit_tests)
endif()

# preserve rpaths when installing and make the install folder relocatable
# use `CMAKE_SKIP_INSTALL_RPATH` to skip this
# https://spack.readthedocs.io/en/latest/workflows.html#write-the-cmake-build
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
          "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" isSystemDir)
# skip RPATH if nlcglib is installed to system directories
if(isSystemDir STREQUAL "-1")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  if(APPLE)
    set(basePoint @loader_path)
  else()
    set(basePoint $ORIGIN)
  endif()
  file(RELATIVE_PATH relDir ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}
                            ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
  set(CMAKE_INSTALL_RPATH ${basePoint} ${basePoint}/${relDir})
endif()

include(CMakePackageConfigHelpers)
install(EXPORT nlcglib_targets
  FILE nlcglib-targets.cmake
  NAMESPACE nlcglib::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/nlcglib")

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/nlcglib/nlcglibConfigVersion.cmake"
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY AnyNewerVersion
  )

configure_package_config_file(
  cmake/nlcglib-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/nlcglib-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nlcglib
  )

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/nlcglib-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/nlcglib/nlcglibConfigVersion.cmake
  DESTINATION
  ${CMAKE_INSTALL_LIBDIR}/cmake/nlcglib)
install(
  FILES
  ${PROJECT_SOURCE_DIR}/include/nlcglib.hpp
  ${PROJECT_SOURCE_DIR}/include/interface.hpp
  DESTINATION "${CMAKE_INSTALL_PREFIX}/include/nlcglib"
  )
