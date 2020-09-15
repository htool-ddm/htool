#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

#=== hmat_partialACA
add_executable(Test_multi_hmat_partialACA test_multi_hmat_partialACA.cpp)
target_link_libraries(Test_multi_hmat_partialACA htool)
add_dependencies(build-tests Test_multi_hmat_partialACA)
add_test(NAME Test_multi_hmat_partialACA_1 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_multi_hmat_partialACA)
add_test(NAME Test_multi_hmat_partialACA_2 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_multi_hmat_partialACA)
add_test(NAME Test_multi_hmat_partialACA_3 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_multi_hmat_partialACA)
add_test(NAME Test_multi_hmat_partialACA_4 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_multi_hmat_partialACA)
