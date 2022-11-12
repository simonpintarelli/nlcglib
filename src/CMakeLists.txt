set(SOURCES
  la/cusolver.cpp
  gpu/acc.cpp
  la/magma.cpp
)

add_library(nlcglib_core ${SOURCES})
nlcglib_setup_target(nlcglib_core)

add_library(nlcglib SHARED nlcglib.cpp)
target_include_directories(nlcglib PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries(nlcglib PRIVATE nlcglib_core)

set_target_properties(nlcglib PROPERTIES PUBLIC_HEADER ${CMAKE_SOURCE_DIR}/include/nlcglib.hpp)

include(GNUInstallDirs)

install(TARGETS nlcglib
  EXPORT nlcglib_targets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
