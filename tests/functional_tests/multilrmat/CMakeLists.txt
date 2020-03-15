#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

#=== lrmat_SVD
set(compressions "partialACA")
# list(APPEND compressions "partialACA")
# list(APPEND compressions "sympartialACA")
# list(APPEND compressions "SVD")
foreach(compression ${compressions})
    add_executable(Test_multi_lrmat_${compression} test_multi_lrmat_${compression}.cpp)
    target_link_libraries(Test_multi_lrmat_${compression} htool)
    add_dependencies(build-tests Test_multi_lrmat_${compression})
    add_test(Test_multi_lrmat_${compression} Test_multi_lrmat_${compression})
endforeach()
