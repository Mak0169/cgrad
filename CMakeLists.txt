cmake_minimum_required(VERSION 3.20)
project(cgrad LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CGRAD_BUILD_TESTS "Build cgrad tests" ON)

# The library
add_library(cgrad src/tensor.cpp)
target_include_directories(cgrad PUBLIC include)

# Warnings (and sanitizers in Debug) for everything that links cgrad
target_compile_options(cgrad PRIVATE -Wall -Wextra -Wpedantic)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(cgrad PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(cgrad PUBLIC -fsanitize=address,undefined)
endif()

# Tests
if(CGRAD_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
