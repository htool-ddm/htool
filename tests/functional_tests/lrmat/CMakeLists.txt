#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

#=== lrmat_SVD
if (EIGEN3_FOUND)
  add_executable(Lrmat_SVD_tests lrmat_SVD_tests.cpp)
  target_link_libraries(Lrmat_SVD_tests htool
                            ${EIGEN3_LIBRARIES})
  target_include_directories(Lrmat_SVD_tests PUBLIC
                            ${EIGEN3_INCLUDE_DIR})
  add_dependencies(build-tests Lrmat_SVD_tests)
  add_test(Lrmat_SVD_tests Lrmat_SVD_tests)
endif()

#=== lrmat_fullACA
add_executable(Lrmat_fullACA_tests lrmat_fullACA_tests.cpp)
target_link_libraries(Lrmat_fullACA_tests htool)
add_dependencies(build-tests Lrmat_fullACA_tests)
add_test(Lrmat_fullACA_tests Lrmat_fullACA_tests)

#=== lrmat_partialACA
add_executable(Lrmat_partialACA_tests lrmat_partialACA_tests.cpp)
target_link_libraries(Lrmat_partialACA_tests htool)
add_dependencies(build-tests Lrmat_partialACA_tests)
add_test(Lrmat_partialACA_tests Lrmat_partialACA_tests)
