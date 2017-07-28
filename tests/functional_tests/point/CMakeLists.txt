#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Test_point test_point.cpp)
target_link_libraries(Test_point htool)
add_dependencies(build-tests Test_point)
add_test(Test_point Test_point)
