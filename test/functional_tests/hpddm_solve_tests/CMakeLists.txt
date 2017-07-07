#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Hpddm_solve_tests hpddm_solve_tests.cpp)
target_link_libraries(Hpddm_solve_tests htool)
add_dependencies(build-tests Hpddm_solve_tests)
# add_test(NAME Hmat_fullACA_tests_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${CMAKE_CURRENT_BINARY_DIR}/Hmat_fullACA_tests)
# add_test(NAME Hmat_fullACA_tests_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${CMAKE_CURRENT_BINARY_DIR}/Hmat_fullACA_tests)
# add_test(NAME Hmat_fullACA_tests_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${CMAKE_CURRENT_BINARY_DIR}/Hmat_fullACA_tests)
