find_package(rocsolver REQUIRED)
find_package(rocblas REQUIRED)
find_package(hip REQUIRED)

if (NOT TARGET nlcglib::rocmlibs)
  add_library(nlcglib::rocmlibs INTERFACE IMPORTED)
  # target_link_libraries(nlcglib::rcomlibs INTERFACE roc::rocblas roc::rocsolver hip::device)
  target_link_libraries(nlcglib::rocmlibs INTERFACE roc::rocblas roc::rocsolver)
endif()
