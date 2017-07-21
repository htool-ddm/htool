#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Matrix_tests matrix_tests.cpp)
target_link_libraries(Matrix_tests htool)
add_dependencies(build-tests Matrix_tests)
add_test(Matrix_tests Matrix_tests)
