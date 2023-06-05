#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Test_local_operator_dense_matrix_product test_local_operator_dense_matrix_product.cpp)
target_link_libraries(Test_local_operator_dense_matrix_product htool)
add_dependencies(build-tests-local-operators-dense-product Test_local_operator_dense_matrix_product)
add_test(Test_local_operator_dense_matrix_product Test_local_operator_dense_matrix_product)
