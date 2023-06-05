#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#
set(types "double")
list(APPEND types "complex_double")

foreach(type ${types})
    add_executable(Test_lrmat_addition_${type} test_lrmat_addition_${type}.cpp)
    target_link_libraries(Test_lrmat_addition_${type} htool)
    add_dependencies(build-tests-lrmat Test_lrmat_addition_${type})
    add_test(Test_lrmat_addition_${type} Test_lrmat_addition_${type})
endforeach()
