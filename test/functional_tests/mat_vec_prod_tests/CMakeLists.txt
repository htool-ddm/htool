#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Mat_vec_prod_tests mat_vec_prod_tests.cpp)
target_link_libraries(Mat_vec_prod_tests htool)
add_dependencies(build-tests Mat_vec_prod_tests)
add_test(NAME Mat_vec_prod_tests_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${CMAKE_CURRENT_BINARY_DIR}/Mat_vec_prod_tests)
add_test(NAME Mat_vec_prod_tests_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${CMAKE_CURRENT_BINARY_DIR}/Mat_vec_prod_tests)
add_test(NAME Mat_vec_prod_tests_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${CMAKE_CURRENT_BINARY_DIR}/Mat_vec_prod_tests)
