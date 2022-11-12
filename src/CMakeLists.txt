set(SOURCES
  la/cusolver.cpp
  gpu/acc.cpp
  la/magma.cpp
)

add_library(nlcglib SHARED nlcglib.cpp ${SOURCES})
target_include_directories(nlcglib PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)

nlcglib_setup_target(nlcglib)
set_target_properties(nlcglib PROPERTIES PUBLIC_HEADER ${CMAKE_SOURCE_DIR}/include/nlcglib.hpp)

include(GNUInstallDirs)

install(TARGETS nlcglib
  EXPORT nlcglib_targets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
