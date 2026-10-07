add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE cgrad)
add_test(NAME test_tensor COMMAND test_tensor)
