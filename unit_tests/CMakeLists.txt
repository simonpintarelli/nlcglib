find_package(GTest OPTIONAL)

if(BUILD_TESTS AND NOT GTEST_FOUND)
  message(FATAL_ERROR "Could not find googletest.")
endif()

if(BUILD_TESTS)
  add_executable(gtest local/test_la_wrappers.cpp local/test_solver_wrappers.cpp)
  nlcglib_setup_target(gtest)
  target_link_libraries(gtest PRIVATE GTest::GTest GTest::Main)
endif()
