#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

#=== lrmat_SVD
set(types "double")
list(APPEND types "complex_double")

foreach(type ${types})
    add_executable(Test_lrmat_product_${type} test_lrmat_product_${type}.cpp)
    target_link_libraries(Test_lrmat_product_${type} htool)
    add_dependencies(build-tests-lrmat-product Test_lrmat_product_${type})
    add_test(Test_lrmat_product_${type} Test_lrmat_product_${type})
endforeach()
