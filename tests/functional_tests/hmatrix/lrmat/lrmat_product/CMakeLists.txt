#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

#=== lrmat_SVD
set(compressions "fullACA")
list(APPEND compressions "partialACA")
list(APPEND compressions "sympartialACA")
list(APPEND compressions "SVD")
foreach(compression ${compressions})
    add_executable(Test_lrmat_product_${compression} test_lrmat_product_${compression}.cpp)
    target_link_libraries(Test_lrmat_product_${compression} htool)
    add_dependencies(build-tests-lrmat-product Test_lrmat_product_${compression})
    add_test(Test_lrmat_product_${compression} Test_lrmat_product_${compression})
endforeach()
