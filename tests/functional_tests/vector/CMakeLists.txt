#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Test_vector test_vector.cpp)
target_link_libraries(Test_vector htool)
add_dependencies(build-tests Test_vector)
add_test(Test_vector Test_vector)

add_executable(Test_vector_file test_vector_file.cpp)
target_link_libraries(Test_vector_file htool)
add_dependencies(build-tests Test_vector_file)
add_test(Test_vector_file Test_vector_file)
