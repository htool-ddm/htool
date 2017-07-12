#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

#=== hmat_SVD
if (EIGEN3_FOUND)
  add_executable(Hmat_SVD_tests hmat_SVD_tests.cpp)
  target_link_libraries(Hmat_SVD_tests htool
                                ${EIGEN3_LIBRARIES})
  target_include_directories(Hmat_SVD_tests PUBLIC
                              ${EIGEN3_INCLUDE_DIR})
  add_dependencies(build-tests Hmat_SVD_tests)
  add_test(NAME Hmat_SVD_tests_1 COMMAND  mpirun -np 1 ${CMAKE_CURRENT_BINARY_DIR}/Hmat_SVD_tests)
  add_test(NAME Hmat_SVD_tests_2 COMMAND  mpirun -np 2 ${CMAKE_CURRENT_BINARY_DIR}/Hmat_SVD_tests)
  add_test(NAME Hmat_SVD_tests_4 COMMAND  mpirun -np 4 ${CMAKE_CURRENT_BINARY_DIR}/Hmat_SVD_tests)

endif()

#=== hmat_fullACA
add_executable(Hmat_fullACA_tests hmat_fullACA_tests.cpp)
target_link_libraries(Hmat_fullACA_tests htool)
add_dependencies(build-tests Hmat_fullACA_tests)
add_test(NAME Hmat_fullACA_tests_1 COMMAND  mpirun -np 1 ${CMAKE_CURRENT_BINARY_DIR}/Hmat_fullACA_tests)
add_test(NAME Hmat_fullACA_tests_2 COMMAND  mpirun -np 2 ${CMAKE_CURRENT_BINARY_DIR}/Hmat_fullACA_tests)
add_test(NAME Hmat_fullACA_tests_4 COMMAND  mpirun -np 4 ${CMAKE_CURRENT_BINARY_DIR}/Hmat_fullACA_tests)

#=== hmat_partialACA
add_executable(Hmat_partialACA_tests hmat_partialACA_tests.cpp)
target_link_libraries(Hmat_partialACA_tests htool)
add_dependencies(build-tests Hmat_partialACA_tests)
add_test(NAME Hmat_partialACA_tests_1 COMMAND  mpirun -np 1 ${CMAKE_CURRENT_BINARY_DIR}/Hmat_partialACA_tests)
add_test(NAME Hmat_partialACA_tests_2 COMMAND  mpirun -np 2 ${CMAKE_CURRENT_BINARY_DIR}/Hmat_partialACA_tests)
add_test(NAME Hmat_partialACA_tests_4 COMMAND  mpirun -np 4 ${CMAKE_CURRENT_BINARY_DIR}/Hmat_partialACA_tests)


#=== hmat_vec_prod
add_executable(Hmat_vec_prod_tests hmat_vec_prod_tests.cpp)
target_link_libraries(Hmat_vec_prod_tests htool)
add_dependencies(build-tests Hmat_vec_prod_tests)
add_test(NAME Hmat_vec_prod_tests_1 COMMAND  mpirun -np 1 ${CMAKE_CURRENT_BINARY_DIR}/Hmat_vec_prod_tests)
add_test(NAME Hmat_vec_prod_tests_2 COMMAND  mpirun -np 2 ${CMAKE_CURRENT_BINARY_DIR}/Hmat_vec_prod_tests)
add_test(NAME Hmat_vec_prod_tests_4 COMMAND  mpirun -np 4 ${CMAKE_CURRENT_BINARY_DIR}/Hmat_vec_prod_tests)
