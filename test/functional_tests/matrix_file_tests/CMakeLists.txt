#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Matrix_file_tests matrix_file_tests.cpp)
target_link_libraries(Matrix_file_tests htool)
add_dependencies(build-tests Matrix_file_tests)
add_test(Matrix_file_tests Matrix_file_tests)
SET_TESTS_PROPERTIES(Matrix_file_tests PROPERTIES WILL_FAIL false)
