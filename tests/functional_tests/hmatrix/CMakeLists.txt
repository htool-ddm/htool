#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

#=== hmat_SVD
if (EIGEN3_FOUND)
  add_executable(Test_hmat_SVD test_hmat_SVD.cpp)
  target_link_libraries(Test_hmat_SVD htool
                                ${EIGEN3_LIBRARIES})
  target_include_directories(Test_hmat_SVD PUBLIC
                              ${EIGEN3_INCLUDE_DIR})
  add_dependencies(build-tests Test_hmat_SVD)
  add_test(NAME Test_hmat_SVD_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_SVD)
  add_test(NAME Test_hmat_SVD_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_SVD)
  add_test(NAME Test_hmat_SVD_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_SVD)

endif()

#=== hmat_fullACA
add_executable(Test_hmat_fullACA test_hmat_fullACA.cpp)
target_link_libraries(Test_hmat_fullACA htool)
add_dependencies(build-tests Test_hmat_fullACA)
add_test(NAME Test_hmat_fullACA_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_fullACA)
add_test(NAME Test_hmat_fullACA_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_fullACA)
add_test(NAME Test_hmat_fullACA_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_fullACA)

#=== hmat_partialACA
add_executable(Test_hmat_partialACA test_hmat_partialACA.cpp)
target_link_libraries(Test_hmat_partialACA htool)
add_dependencies(build-tests Test_hmat_partialACA)
add_test(NAME Test_hmat_partialACA_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_partialACA)
add_test(NAME Test_hmat_partialACA_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_partialACA)
add_test(NAME Test_hmat_partialACA_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_partialACA)


#=== hmat_vec_prod
add_executable(Test_mat_vec_prod test_hmat_vec_prod.cpp)
target_link_libraries(Test_mat_vec_prod htool)
add_dependencies(build-tests Test_mat_vec_prod)
add_test(NAME Test_mat_vec_prod_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${CMAKE_CURRENT_BINARY_DIR}/Test_mat_vec_prod)
add_test(NAME Test_mat_vec_prod_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${CMAKE_CURRENT_BINARY_DIR}/Test_mat_vec_prod)
add_test(NAME Test_mat_vec_prod_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${CMAKE_CURRENT_BINARY_DIR}/Test_mat_vec_prod)
