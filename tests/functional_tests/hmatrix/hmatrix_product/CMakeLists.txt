#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

#=== lrmat_SVD

add_executable(Test_hmatrix_product test_hmatrix_product.cpp)
target_link_libraries(Test_hmatrix_product htool)
add_dependencies(build-tests-hmatrix-product Test_hmatrix_product)
add_test(Test_hmatrix_product Test_hmatrix_product)
