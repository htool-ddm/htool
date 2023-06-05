#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Test_local_operator_dense_matrix test_local_operator_dense_matrix.cpp)
target_link_libraries(Test_local_operator_dense_matrix htool)
add_dependencies(build-tests-local-operators Test_local_operator_dense_matrix)
add_test(Test_local_operator_dense_matrix Test_local_operator_dense_matrix)

add_custom_target(build-tests-local-operators-dense-product)
add_subdirectory(local_dense_operator_product)
