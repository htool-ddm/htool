#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#


if (EIGEN3_FOUND)
  #=== hmat_SVD
  add_executable(Test_hmat_SVD test_hmat_SVD.cpp)
  target_link_libraries(Test_hmat_SVD htool
                                ${EIGEN3_LIBRARIES})
  target_include_directories(Test_hmat_SVD PUBLIC
                              ${EIGEN3_INCLUDE_DIR})
  add_dependencies(build-tests Test_hmat_SVD)
  add_test(NAME Test_hmat_SVD_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_SVD)
  add_test(NAME Test_hmat_SVD_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_SVD)
  add_test(NAME Test_hmat_SVD_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_SVD)

  #=== hmat_cluster_SVD
  add_executable(Test_hmat_cluster_SVD test_hmat_cluster_SVD.cpp)
  target_link_libraries(Test_hmat_cluster_SVD htool
                                ${EIGEN3_LIBRARIES})
  target_include_directories(Test_hmat_cluster_SVD PUBLIC
                              ${EIGEN3_INCLUDE_DIR})
  add_dependencies(build-tests Test_hmat_cluster_SVD)
  add_test(NAME Test_hmat_cluster_SVD_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_cluster_SVD)
  add_test(NAME Test_hmat_cluster_SVD_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_cluster_SVD)
  add_test(NAME Test_hmat_cluster_SVD_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_cluster_SVD)

endif()
message("coucou")
message("${MPIEXEC_PREFLAGS}")
#=== hmat_fullACA
add_executable(Test_hmat_fullACA test_hmat_fullACA.cpp)
target_link_libraries(Test_hmat_fullACA htool)
add_dependencies(build-tests Test_hmat_fullACA)
add_test(NAME Test_hmat_fullACA_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_fullACA)
add_test(NAME Test_hmat_fullACA_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_fullACA)
add_test(NAME Test_hmat_fullACA_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_fullACA)

#=== hmat_cluster_fullACA
add_executable(Test_hmat_cluster_fullACA test_hmat_cluster_fullACA.cpp)
target_link_libraries(Test_hmat_cluster_fullACA htool)
add_dependencies(build-tests Test_hmat_cluster_fullACA)
add_test(NAME Test_hmat_cluster_fullACA_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_cluster_fullACA)
add_test(NAME Test_hmat_cluster_fullACA_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_cluster_fullACA)
add_test(NAME Test_hmat_cluster_fullACA_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_cluster_fullACA)

#=== hmat_partialACA
add_executable(Test_hmat_partialACA test_hmat_partialACA.cpp)
target_link_libraries(Test_hmat_partialACA htool)
add_dependencies(build-tests Test_hmat_partialACA)
add_test(NAME Test_hmat_partialACA_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_partialACA)
add_test(NAME Test_hmat_partialACA_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_partialACA)
add_test(NAME Test_hmat_partialACA_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_partialACA)


#=== hmat_cluster_partialACA
add_executable(Test_hmat_cluster_partialACA test_hmat_cluster_partialACA.cpp)
target_link_libraries(Test_hmat_cluster_partialACA htool)
add_dependencies(build-tests Test_hmat_cluster_partialACA)
add_test(NAME Test_hmat_cluster_partialACA_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_cluster_partialACA)
add_test(NAME Test_hmat_cluster_partialACA_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_cluster_partialACA)
add_test(NAME Test_hmat_cluster_partialACA_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_cluster_partialACA)


#=== hmat_vec_prod
add_executable(Test_hmat_vec_prod test_hmat_vec_prod.cpp)
target_link_libraries(Test_hmat_vec_prod htool)
add_dependencies(build-tests Test_hmat_vec_prod)
add_test(NAME Test_hmat_vec_prod_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_vec_prod)
add_test(NAME Test_hmat_vec_prod_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_vec_prod)
add_test(NAME Test_hmat_vec_prod_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_vec_prod)
