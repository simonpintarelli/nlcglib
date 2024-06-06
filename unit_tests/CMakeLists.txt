find_package(GTest REQUIRED)

if(BUILD_TESTS AND NOT GTEST_FOUND)
  message(FATAL_ERROR "Could not find googletest.")
endif()

if(BUILD_TESTS)
  add_executable(gtest local/test_la_wrappers.cpp local/test_solver_wrappers.cpp)
  target_link_libraries(gtest PUBLIC nlcglib_core)
  target_link_libraries(gtest PRIVATE GTest::gtest GTest::gtest_main)
endif()
