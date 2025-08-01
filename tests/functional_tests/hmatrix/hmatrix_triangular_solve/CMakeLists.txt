#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

set(types "double")
list(APPEND types "complex_double")

foreach(type ${types})
    add_executable(Test_hmatrix_triangular_solve_${type} test_hmatrix_triangular_solve_${type}.cpp)
    target_link_libraries(Test_hmatrix_triangular_solve_${type} htool)
    add_dependencies(build-tests-hmatrix-triangular-solve Test_hmatrix_triangular_solve_${type})
    add_test(Test_hmatrix_triangular_solve_${type} Test_hmatrix_triangular_solve_${type})
endforeach()

foreach(type ${types})
    add_executable(Test_task_based_hmatrix_triangular_solve_${type} test_task_based_hmatrix_triangular_solve_${type}.cpp)
    target_link_libraries(Test_task_based_hmatrix_triangular_solve_${type} htool)
    add_dependencies(build-tests-hmatrix-triangular-solve Test_task_based_hmatrix_triangular_solve_${type})

    add_test(Test_task_based_hmatrix_triangular_solve_${type}_1 Test_task_based_hmatrix_triangular_solve_${type})
    set_tests_properties(Test_task_based_hmatrix_triangular_solve_${type}_1 PROPERTIES ENVIRONMENT OMP_NUM_THREADS=1)

    add_test(Test_task_based_hmatrix_triangular_solve_${type}_2 Test_task_based_hmatrix_triangular_solve_${type})
    set_tests_properties(Test_task_based_hmatrix_triangular_solve_${type}_2 PROPERTIES ENVIRONMENT OMP_NUM_THREADS=2)

    add_test(Test_task_based_hmatrix_triangular_solve_${type}_4 Test_task_based_hmatrix_triangular_solve_${type})
    set_tests_properties(Test_task_based_hmatrix_triangular_solve_${type}_4 PROPERTIES ENVIRONMENT OMP_NUM_THREADS=4)
endforeach()
