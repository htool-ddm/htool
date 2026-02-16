#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

#=== lrmat_SVD
set(compressions "fullACA")
list(APPEND compressions "partialACA")
list(APPEND compressions "recompressed_partialACA")
list(APPEND compressions "sympartialACA")
list(APPEND compressions "SVD")
list(APPEND compressions "HCA")
foreach(compression ${compressions})
    add_executable(Test_lrmat_build_${compression} test_lrmat_build_${compression}.cpp)
    target_link_libraries(Test_lrmat_build_${compression} htool)
    add_dependencies(build-tests-lrmat-build Test_lrmat_build_${compression})
    add_test(Test_lrmat_build_${compression} Test_lrmat_build_${compression})
endforeach()
