#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

if (EIGEN3_FOUND)
  add_executable(Hmat_SVD_tests hmat_SVD_tests.cpp)
  target_link_libraries(Hmat_SVD_tests htool
                                ${EIGEN3_LIBRARIES})
  target_include_directories(Hmat_SVD_tests PUBLIC
                              ${EIGEN3_INCLUDE_DIR})
  add_dependencies(build-tests Hmat_SVD_tests)
  add_test(NAME Hmat_SVD_tests_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${CMAKE_CURRENT_BINARY_DIR}/Hmat_SVD_tests)
  add_test(NAME Hmat_SVD_tests_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${CMAKE_CURRENT_BINARY_DIR}/Hmat_SVD_tests)
  add_test(NAME Hmat_SVD_tests_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${CMAKE_CURRENT_BINARY_DIR}/Hmat_SVD_tests)

endif()
