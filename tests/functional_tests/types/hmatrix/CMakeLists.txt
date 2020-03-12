#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#


#=== hmat_auto_SVD
add_executable(Test_hmat_auto_SVD test_hmat_auto_SVD.cpp)
target_link_libraries(Test_hmat_auto_SVD htool)
add_dependencies(build-tests Test_hmat_auto_SVD)
add_test(NAME Test_hmat_auto_SVD_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_auto_SVD)
add_test(NAME Test_hmat_auto_SVD_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_auto_SVD)
add_test(NAME Test_hmat_auto_SVD_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_auto_SVD)

#=== hmat_cluster_SVD
add_executable(Test_hmat_cluster_SVD test_hmat_cluster_SVD.cpp)
target_link_libraries(Test_hmat_cluster_SVD htool)
add_dependencies(build-tests Test_hmat_cluster_SVD)
add_test(NAME Test_hmat_cluster_SVD_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_cluster_SVD)
add_test(NAME Test_hmat_cluster_SVD_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_cluster_SVD)
add_test(NAME Test_hmat_cluster_SVD_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_cluster_SVD)


#=== hmat_auto_fullACA
add_executable(Test_hmat_auto_fullACA test_hmat_auto_fullACA.cpp)
target_link_libraries(Test_hmat_auto_fullACA htool)
add_dependencies(build-tests Test_hmat_auto_fullACA)
add_test(NAME Test_hmat_auto_fullACA_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_auto_fullACA)
add_test(NAME Test_hmat_auto_fullACA_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_auto_fullACA)
add_test(NAME Test_hmat_auto_fullACA_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_auto_fullACA)

#=== hmat_cluster_fullACA
add_executable(Test_hmat_cluster_fullACA test_hmat_cluster_fullACA.cpp)
target_link_libraries(Test_hmat_cluster_fullACA htool)
add_dependencies(build-tests Test_hmat_cluster_fullACA)
add_test(NAME Test_hmat_cluster_fullACA_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_cluster_fullACA)
add_test(NAME Test_hmat_cluster_fullACA_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_cluster_fullACA)
add_test(NAME Test_hmat_cluster_fullACA_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_cluster_fullACA)

#=== hmat_auto_partialACA
add_executable(Test_hmat_auto_partialACA test_hmat_auto_partialACA.cpp)
target_link_libraries(Test_hmat_auto_partialACA htool)
add_dependencies(build-tests Test_hmat_auto_partialACA)
add_test(NAME Test_hmat_auto_partialACA_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_auto_partialACA)
add_test(NAME Test_hmat_auto_partialACA_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_auto_partialACA)
add_test(NAME Test_hmat_auto_partialACA_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_auto_partialACA)


#=== hmat_cluster_partialACA
add_executable(Test_hmat_cluster_partialACA test_hmat_cluster_partialACA.cpp)
target_link_libraries(Test_hmat_cluster_partialACA htool)
add_dependencies(build-tests Test_hmat_cluster_partialACA)
add_test(NAME Test_hmat_cluster_partialACA_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_cluster_partialACA)
add_test(NAME Test_hmat_cluster_partialACA_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_cluster_partialACA)
add_test(NAME Test_hmat_cluster_partialACA_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_cluster_partialACA)

#=== hmat_auto_sympartialACA
add_executable(Test_hmat_auto_sympartialACA test_hmat_auto_sympartialACA.cpp)
target_link_libraries(Test_hmat_auto_sympartialACA htool)
add_dependencies(build-tests Test_hmat_auto_sympartialACA)
add_test(NAME Test_hmat_auto_sympartialACA_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_auto_sympartialACA)
add_test(NAME Test_hmat_auto_sympartialACA_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_auto_sympartialACA)
add_test(NAME Test_hmat_auto_sympartialACA_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_auto_sympartialACA)


#=== hmat_cluster_sympartialACA
add_executable(Test_hmat_cluster_sympartialACA test_hmat_cluster_sympartialACA.cpp)
target_link_libraries(Test_hmat_cluster_sympartialACA htool)
add_dependencies(build-tests Test_hmat_cluster_sympartialACA)
add_test(NAME Test_hmat_cluster_sympartialACA_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_cluster_sympartialACA)
add_test(NAME Test_hmat_cluster_sympartialACA_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_cluster_sympartialACA)
add_test(NAME Test_hmat_cluster_sympartialACA_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_cluster_sympartialACA)

#=== hmat_vec_prod
add_executable(Test_hmat_vec_prod test_hmat_vec_prod.cpp)
target_link_libraries(Test_hmat_vec_prod htool)
add_dependencies(build-tests Test_hmat_vec_prod)
add_test(NAME Test_hmat_vec_prod_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_vec_prod)
add_test(NAME Test_hmat_vec_prod_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_vec_prod)
add_test(NAME Test_hmat_vec_prod_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_vec_prod)

#=== hmat_mat_prod
add_executable(Test_hmat_mat_prod test_hmat_mat_prod.cpp)
target_link_libraries(Test_hmat_mat_prod htool)
add_dependencies(build-tests Test_hmat_mat_prod)
add_test(NAME Test_hmat_mat_prod_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_mat_prod)
add_test(NAME Test_hmat_mat_prod_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_mat_prod)
add_test(NAME Test_hmat_mat_prod_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_mat_prod)

#=== hmat_to_dense
add_executable(Test_hmat_to_dense test_hmat_to_dense.cpp)
target_link_libraries(Test_hmat_to_dense htool)
add_dependencies(build-tests Test_hmat_to_dense)
add_test(NAME Test_hmat_to_dense COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_to_dense)

#=== hmat_save
add_executable(Test_hmat_save test_hmat_save.cpp)
target_link_libraries(Test_hmat_save htool)
add_dependencies(build-tests Test_hmat_save)
add_test(NAME Test_hmat_save COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_save)
