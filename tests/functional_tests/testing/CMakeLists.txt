#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Test_testing_local_dense_matrix test_testing_local_dense_matrix.cpp)
target_link_libraries(Test_testing_local_dense_matrix htool)
add_dependencies(build-tests-testing Test_testing_local_dense_matrix)
add_test(Test_testing_local_dense_matrix Test_testing_local_dense_matrix)
