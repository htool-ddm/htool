#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

set(types "double")
list(APPEND types "complex_double")

foreach(type ${types})
    add_executable(Test_hmatrix_task_based_${type} test_hmatrix_task_based_${type}.cpp)
    target_link_libraries(Test_hmatrix_task_based_${type} htool)
    add_dependencies(build-tests-hmatrix-task-based Test_hmatrix_task_based_${type})
    add_test(NAME Test_hmatrix_task_based_${type}_1 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmatrix_task_based_${type})
    set_tests_properties(Test_hmatrix_task_based_${type}_1 PROPERTIES ENVIRONMENT OMP_NUM_THREADS=4)

    add_test(NAME Test_hmatrix_task_based_${type}_2 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmatrix_task_based_${type})
    set_tests_properties(Test_hmatrix_task_based_${type}_2 PROPERTIES ENVIRONMENT OMP_NUM_THREADS=2)

    add_test(NAME Test_hmatrix_task_based_${type}_4 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmatrix_task_based_${type})
    set_tests_properties(Test_hmatrix_task_based_${type}_4 PROPERTIES ENVIRONMENT OMP_NUM_THREADS=1)

endforeach()
