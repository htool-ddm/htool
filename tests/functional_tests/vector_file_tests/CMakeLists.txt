#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Vector_file_tests vector_file_tests.cpp)
target_link_libraries(Vector_file_tests htool)
add_dependencies(build-tests Vector_file_tests)
add_test(Vector_file_tests Vector_file_tests)
