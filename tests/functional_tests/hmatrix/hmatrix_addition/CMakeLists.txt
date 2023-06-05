#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

set(types "double")
list(APPEND types "complex_double")

foreach(type ${types})
    add_executable(Test_hmatrix_addition_${type} test_hmatrix_addition_${type}.cpp)
    target_link_libraries(Test_hmatrix_addition_${type} htool)
    add_dependencies(build-tests-hmatrix-addition Test_hmatrix_addition_${type})
    add_test(Test_hmatrix_addition_${type}_1 Test_hmatrix_addition_${type})
endforeach()
