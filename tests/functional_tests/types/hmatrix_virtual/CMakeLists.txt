#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Test_hmat_virtual_cluster_geometric_splitting_fullACA test_hmat_virtual_cluster_geometric_splitting_fullACA.cpp)
target_link_libraries(Test_hmat_virtual_cluster_geometric_splitting_fullACA htool)
add_dependencies(build-tests Test_hmat_virtual_cluster_geometric_splitting_fullACA)
add_test(NAME Test_hmat_virtual_cluster_geometric_splitting_fullACA_1 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_virtual_cluster_geometric_splitting_fullACA)
add_test(NAME Test_hmat_virtual_cluster_geometric_splitting_fullACA_2 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_virtual_cluster_geometric_splitting_fullACA)
add_test(NAME Test_hmat_virtual_cluster_geometric_splitting_fullACA_3 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_virtual_cluster_geometric_splitting_fullACA)
add_test(NAME Test_hmat_virtual_cluster_geometric_splitting_fullACA_4 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_virtual_cluster_geometric_splitting_fullACA)
