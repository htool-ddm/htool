#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Vector_tests vector_tests.cpp)
target_link_libraries(Vector_tests htool)
add_dependencies(build-tests Vector_tests)
add_test(Vector_tests Vector_tests)
