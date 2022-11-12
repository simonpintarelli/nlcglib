include(FindPackageHandleStandardArgs)

find_path(MAGMA_INCLUDE_DIR magmablas.h
  HINTS
  ENV EBROOTMAGMA
  ENV MAGMA_DIR
  ENV MAGMAROOT
  PATH_SUFFIXES include magma/include
  )

find_library(MAGMA_LIBRARIES NAMES magma magma_sparse
  HINTS
  ENV EBROOTMAGMA
  ENV MAGMA_DIR
  ENV MAGMAROOT
  PATH_SUFFIXES lib magma/lib
  )

find_package_handle_standard_args(MAGMA DEFAULT_MSG MAGMA_INCLUDE_DIR MAGMA_LIBRARIES)
mark_as_advanced(MAGMA_FOUND MAGMA_INCLUDE_DIR MAGMA_LIBRARIES)
find_package(hipblas REQUIRED)
find_package(hipsparse REQUIRED)

if(MAGMA_FOUND AND NOT TARGET nlcglib::magma)
  add_library(nlcglib::magma INTERFACE IMPORTED)
  set_target_properties(nlcglib::magma PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${MAGMA_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${MAGMA_LIBRARIES}")
endif()
