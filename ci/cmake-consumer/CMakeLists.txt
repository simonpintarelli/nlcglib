cmake_minimum_required(VERSION 3.21)
project(nlcglib_consumer LANGUAGES CXX)

find_package(nlcglib 1.4 CONFIG REQUIRED)

# A shared consumer also verifies that a static nlcglib was built with PIC.
add_library(nlcglib_consumer SHARED consumer.cpp)
target_link_libraries(nlcglib_consumer PUBLIC nlcglib::nlcglib)
set_target_properties(nlcglib_consumer PROPERTIES CXX_STANDARD 11
                                                  CXX_STANDARD_REQUIRED ON)

add_executable(nlcglib_consumer_test main.cpp)
target_link_libraries(nlcglib_consumer_test PRIVATE nlcglib_consumer)

enable_testing()
add_test(NAME nlcglib.consumer COMMAND nlcglib_consumer_test)
