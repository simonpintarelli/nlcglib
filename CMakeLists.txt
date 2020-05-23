cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(nlcglib LANGUAGES CXX CUDA VERSION 0.8)

set(CMAKE_CXX_STANDARD 14)

# user variables
set(USE_OPENMP On CACHE BOOL "use OpenMP")

set(BUILD_TESTS OFF CACHE BOOL "build tests")
set(LAPACK_VENDOR "OpenBLAS" CACHE STRING "lapack vendor")
set_property(CACHE LAPACK_VENDOR PROPERTY STRINGS "OpenBLAS" "MKL")

find_package(MPI REQUIRED)

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
find_package(CUDA REQUIRED)
find_package(MPI REQUIRED)

if(LAPACK_VENDOR MATCHES OpenBLAS)
  find_package(OpenBLAS REQUIRED)
  add_library(my_lapack INTERFACE IMPORTED)
  set_target_properties(my_lapack PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${OpenBLAS_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${OpenBLAS_LIBRARIES}")
elseif(LAPACK_VENDOR MATCHES MKL)
  message("LAPACK VENDOR MKL")
  find_package(MKL REQUIRED)
else()
  message(FATAL_ERROR "must specify a LAPACK_VENDOR")
endif()

add_subdirectory(src)
set(nlcglib_internal_location $<TARGET_FILE:nlcglib_internal>)

if(BUILD_TESTS)
  add_subdirectory(test)
endif()

include(CMakePackageConfigHelpers)
install(EXPORT nlcglib_targets
  FILE nlcglib-targets.cmake
  NAMESPACE nlcglib::
  DESTINATION "${CMAKE_INSTALL_PREFIX}/cmake/nlcglib")


write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/nlcglib/nlcglibConfigVersion.cmake"
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY AnyNewerVersion
  )

configure_package_config_file(
  cmake/nlcglib-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/nlcglib-config.cmake
  INSTALL_DESTINATION "${CMAKE_INSTALL_PREFIX}/cmake/nlcglib"
  )

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/nlcglib-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/nlcglib/nlcglibConfigVersion.cmake
  DESTINATION
  ${CMAKE_INSTALL_PREFIX}/cmake/nlcglib)
install(FILES ${PROJECT_SOURCE_DIR}/include/nlcglib.hpp
  ${PROJECT_SOURCE_DIR}/include/interface.hpp
  DESTINATION "${CMAKE_INSTALL_PREFIX}/include/nlcglib")
