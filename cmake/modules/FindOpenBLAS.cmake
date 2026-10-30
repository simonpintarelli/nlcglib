# OpenBLAS packages may provide either an imported target or only variables.
# Provide OpenBLAS::LAPACKE with the CBLAS/LAPACKE interfaces
# used by nlcglib. Some distributions ship LAPACKE in a separate library.
find_package(OpenBLAS QUIET CONFIG)

set(_OpenBLAS_libraries)
foreach(_OpenBLAS_target OpenBLAS::OpenBLAS OpenBLAS::openblas openblas)
  if(TARGET ${_OpenBLAS_target})
    set(_OpenBLAS_libraries ${_OpenBLAS_target})
    get_target_property(_OpenBLAS_include_dirs ${_OpenBLAS_target}
                        INTERFACE_INCLUDE_DIRECTORIES)
    break()
  endif()
endforeach()
if(NOT _OpenBLAS_libraries)
  if(OpenBLAS_LIBRARIES)
    set(_OpenBLAS_libraries ${OpenBLAS_LIBRARIES})
  else()
    find_library(OpenBLAS_LIBRARY NAMES openblas)
    set(_OpenBLAS_libraries ${OpenBLAS_LIBRARY})
  endif()
endif()
find_path(OpenBLAS_CBLAS_INCLUDE_DIR cblas.h HINTS ${OpenBLAS_INCLUDE_DIRS}
                                                   ${_OpenBLAS_include_dirs})
find_path(OpenBLAS_LAPACKE_INCLUDE_DIR lapacke.h
          HINTS ${OpenBLAS_INCLUDE_DIRS} ${_OpenBLAS_include_dirs})

if(_OpenBLAS_libraries)
  include(CMakePushCheckState)
  include(CheckCXXSourceCompiles)
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_LIBRARIES ${_OpenBLAS_libraries})
  check_cxx_source_compiles("extern \"C\" void LAPACKE_zheevd();
     int main() { LAPACKE_zheevd(); }" OpenBLAS_HAS_LAPACKE)
  cmake_pop_check_state()
  if(NOT OpenBLAS_HAS_LAPACKE)
    find_library(OpenBLAS_LAPACKE_LIBRARY NAMES lapacke)
  endif()
endif()

include(FindPackageHandleStandardArgs)
set(_OpenBLAS_required_vars _OpenBLAS_libraries OpenBLAS_CBLAS_INCLUDE_DIR
                            OpenBLAS_LAPACKE_INCLUDE_DIR)
if(NOT OpenBLAS_HAS_LAPACKE)
  list(APPEND _OpenBLAS_required_vars OpenBLAS_LAPACKE_LIBRARY)
endif()
find_package_handle_standard_args(OpenBLAS
                                  REQUIRED_VARS ${_OpenBLAS_required_vars})

if(OpenBLAS_FOUND)
  if(NOT TARGET OpenBLAS::OpenBLAS)
    add_library(OpenBLAS::OpenBLAS INTERFACE IMPORTED)
    set_target_properties(
      OpenBLAS::OpenBLAS PROPERTIES INTERFACE_LINK_LIBRARIES
                                    "${_OpenBLAS_libraries}")
  endif()
  if(NOT TARGET OpenBLAS::LAPACKE)
    add_library(OpenBLAS::LAPACKE INTERFACE IMPORTED)
    set_target_properties(
      OpenBLAS::LAPACKE
      PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                 "${OpenBLAS_CBLAS_INCLUDE_DIR};${OpenBLAS_LAPACKE_INCLUDE_DIR}"
    )
    if(OpenBLAS_HAS_LAPACKE)
      target_link_libraries(OpenBLAS::LAPACKE INTERFACE OpenBLAS::OpenBLAS)
    else()
      # LAPACKE must precede BLAS/LAPACK for static linkers.
      target_link_libraries(
        OpenBLAS::LAPACKE INTERFACE "${OpenBLAS_LAPACKE_LIBRARY}"
                                    OpenBLAS::OpenBLAS)
    endif()
  endif()
endif()
mark_as_advanced(OpenBLAS_LIBRARY OpenBLAS_CBLAS_INCLUDE_DIR
                 OpenBLAS_LAPACKE_INCLUDE_DIR OpenBLAS_LAPACKE_LIBRARY)
