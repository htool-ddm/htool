#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

set(types "double")
list(APPEND types "complex_double")

foreach(type ${types})
    add_executable(Test_matrix_triangular_solve_${type} test_matrix_triangular_solve_${type}.cpp)
    target_link_libraries(Test_matrix_triangular_solve_${type} htool)
    add_dependencies(build-tests-matrix-triangular-solve Test_matrix_triangular_solve_${type})
    add_test(Test_matrix_triangular_solve_${type} Test_matrix_triangular_solve_${type})
endforeach()
