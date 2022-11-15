#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

set(types "double")
list(APPEND types "complex_double")

set(operations "vec_prod")
list(APPEND operations "mat_prod")
list(APPEND operations "vec_prod_transp")
list(APPEND operations "mat_prod_transp")

foreach(type ${types})
    foreach(operation ${operations})
        add_executable(Test_distributed_operator_${operation}_${type} test_distributed_operator_${operation}_${type}.cpp)
        target_link_libraries(Test_distributed_operator_${operation}_${type} htool)
        add_dependencies(build-tests-distributed-operator Test_distributed_operator_${operation}_${type})
        add_test(NAME Test_distributed_operator_${operation}_${type}_1 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_distributed_operator_${operation}_${type})
        set_tests_properties(Test_distributed_operator_${operation}_${type}_1 PROPERTIES ENVIRONMENT OMP_NUM_THREADS=4)
        add_test(NAME Test_distributed_operator_${operation}_${type}_2 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_distributed_operator_${operation}_${type})
        set_tests_properties(Test_distributed_operator_${operation}_${type}_2 PROPERTIES ENVIRONMENT OMP_NUM_THREADS=2)
        add_test(NAME Test_distributed_operator_${operation}_${type}_4 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_distributed_operator_${operation}_${type})
        set_tests_properties(Test_distributed_operator_${operation}_${type}_4 PROPERTIES ENVIRONMENT OMP_NUM_THREADS=1)
    endforeach()
endforeach()

# add_executable(Test_distributed_operator_mat_prod_complex_double test_distributed_operator_mat_prod_complex_double.cpp)
# target_link_libraries(Test_distributed_operator_mat_prod_complex_double htool)
# add_dependencies(build-tests-distributed-operator Test_distributed_operator_mat_prod_complex_double)

# add_test(Test_distributed_operator_mat_prod_complex_double Test_distributed_operator_mat_prod_complex_double)
