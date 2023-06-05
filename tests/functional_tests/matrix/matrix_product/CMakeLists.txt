#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

set(types "double")
list(APPEND types "complex_double")

foreach(type ${types})
    add_executable(Test_matrix_product_${type} test_matrix_product_${type}.cpp)
    target_link_libraries(Test_matrix_product_${type} htool)
    add_dependencies(build-tests-matrix-product Test_matrix_product_${type})
    add_test(Test_matrix_product_${type} Test_matrix_product_${type})
endforeach()
