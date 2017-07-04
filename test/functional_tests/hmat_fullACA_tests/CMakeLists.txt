#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Hmat_fullACA_tests hmat_fullACA_tests.cpp)
target_link_libraries(Hmat_fullACA_tests htool)
add_dependencies(build-tests Hmat_fullACA_tests)
add_test(NAME Hmat_fullACA_tests_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${CMAKE_CURRENT_BINARY_DIR}/Hmat_fullACA_tests)
add_test(NAME Hmat_fullACA_tests_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${CMAKE_CURRENT_BINARY_DIR}/Hmat_fullACA_tests)
add_test(NAME Hmat_fullACA_tests_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${CMAKE_CURRENT_BINARY_DIR}/Hmat_fullACA_tests)
