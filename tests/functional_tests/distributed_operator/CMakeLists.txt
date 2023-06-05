#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

set(data_types "matrix")
list(APPEND data_types "hmatrix")

set(types "double")
list(APPEND types "complex_double")

# foreach(data_type ${data_types})
foreach(type ${types})
    add_executable(Test_distributed_operator_product_${type} test_distributed_operator_product_${type}.cpp)
    target_link_libraries(Test_distributed_operator_product_${type} htool)
    add_dependencies(build-tests-distributed-operator Test_distributed_operator_product_${type})
    add_test(NAME Test_distributed_operator_product_${type}_1 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_distributed_operator_product_${type})
    set_tests_properties(Test_distributed_operator_product_${type}_1 PROPERTIES ENVIRONMENT OMP_NUM_THREADS=4)
    set_tests_properties(Test_distributed_operator_product_${type}_1 PROPERTIES LABELS "mpi")
    add_test(NAME Test_distributed_operator_product_${type}_2 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_distributed_operator_product_${type})
    set_tests_properties(Test_distributed_operator_product_${type}_2 PROPERTIES ENVIRONMENT OMP_NUM_THREADS=2)
    set_tests_properties(Test_distributed_operator_product_${type}_2 PROPERTIES LABELS "mpi")
    add_test(NAME Test_distributed_operator_product_${type}_4 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_distributed_operator_product_${type})
    set_tests_properties(Test_distributed_operator_product_${type}_4 PROPERTIES ENVIRONMENT OMP_NUM_THREADS=1)
    set_tests_properties(Test_distributed_operator_product_${type}_4 PROPERTIES LABELS "mpi")
endforeach()
# endforeach()

# add_executable(Test_distributed_operator_mat_prod_complex_double test_distributed_operator_mat_prod_complex_double.cpp)
# target_link_libraries(Test_distributed_operator_mat_prod_complex_double htool)
# add_dependencies(build-tests-distributed-operator Test_distributed_operator_mat_prod_complex_double)

# add_test(Test_distributed_operator_mat_prod_complex_double Test_distributed_operator_mat_prod_complex_double)
