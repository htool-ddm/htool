#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

#=== lrmat_SVD
set(compressions "fullACA")
list(APPEND compressions "partialACA")
list(APPEND compressions "sympartialACA")
list(APPEND compressions "SVD")
foreach(compression ${compressions})
    add_executable(Test_lrmat_${compression} test_lrmat_${compression}.cpp)
    target_link_libraries(Test_lrmat_${compression} htool)
    add_dependencies(build-tests Test_lrmat_${compression})
    add_test(Test_lrmat_${compression} Test_lrmat_${compression})
endforeach()
