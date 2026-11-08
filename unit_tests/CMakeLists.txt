find_package(GTest REQUIRED)

add_executable(nlcglib_unit_tests local/test_la_wrappers.cpp
                                  local/test_solver_wrappers.cpp
                                  local/test_smearing.cpp)
target_link_libraries(nlcglib_unit_tests PRIVATE nlcglib_test_support
                                                 GTest::gtest)
add_test(NAME nlcglib.unit_tests COMMAND nlcglib_unit_tests)
