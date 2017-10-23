#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

#=== lrmat_SVD
if (EIGEN3_FOUND)
  add_executable(Test_lrmat_SVD test_lrmat_SVD.cpp)
  target_link_libraries(Test_lrmat_SVD htool
                            ${EIGEN3_LIBRARIES})
  target_include_directories(Test_lrmat_SVD PUBLIC
                            ${EIGEN3_INCLUDE_DIR})
  add_dependencies(build-tests Test_lrmat_SVD)
  add_test(Test_lrmat_SVD Test_lrmat_SVD)
endif()

#=== lrmat_fullACA
add_executable(Test_lrmat_fullACA test_lrmat_fullACA.cpp)
target_link_libraries(Test_lrmat_fullACA htool)
add_dependencies(build-tests Test_lrmat_fullACA)
add_test(Test_lrmat_fullACA Test_lrmat_fullACA)

#=== lrmat_partialACA
add_executable(Test_lrmat_partialACA test_lrmat_partialACA.cpp)
target_link_libraries(Test_lrmat_partialACA htool)
add_dependencies(build-tests Test_lrmat_partialACA)
add_test(Test_lrmat_partialACA Test_lrmat_partialACA)
