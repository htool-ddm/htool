#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Test_basic_types_vector test_basic_types_vector.cpp)
target_link_libraries(Test_basic_types_vector htool)
add_dependencies(build-tests-basic-types Test_basic_types_vector)
add_test(Test_basic_types_vector Test_basic_types_vector)

add_executable(Test_basic_types_vector_file test_basic_types_vector_file.cpp)
target_link_libraries(Test_basic_types_vector_file htool)
add_dependencies(build-tests-basic-types Test_basic_types_vector_file)
add_test(Test_basic_types_vector_file Test_basic_types_vector_file)
