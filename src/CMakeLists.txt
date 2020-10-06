add_library(nlcglib_internal STATIC la/cusolver.cpp gpu/acc.cpp)
target_link_libraries(nlcglib_internal PUBLIC Kokkos::kokkoscore)
target_link_libraries(nlcglib_internal INTERFACE "${LAPACK_LIBRARIES}")
target_link_libraries(nlcglib_internal INTERFACE MPI::MPI_CXX)
target_link_libraries(nlcglib_internal INTERFACE my_lapack)
target_include_directories(nlcglib_internal PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_include_directories(nlcglib_internal PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(nlcglib_internal PUBLIC ${CUDA_INCLUDE_DIRS})
set(nlcglib_internal_location $<TARGET_FILE:nlcglib_internal>)


add_library(nlcglib SHARED nlcglib.cpp)
target_include_directories(nlcglib
  PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

nlcglib_setup_target(nlcglib)
set_target_properties(nlcglib PROPERTIES
  PUBLIC_HEADER ${CMAKE_SOURCE_DIR}/include/nlcglib.hpp)
target_compile_definitions(nlcglib PUBLIC $<$<BOOL:${USE_CUDA}>:__CUDA>)

include(GNUInstallDirs)

install(TARGETS nlcglib
  EXPORT nlcglib_targets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
