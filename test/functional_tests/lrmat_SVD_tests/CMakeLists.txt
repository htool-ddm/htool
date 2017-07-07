#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#


if (EIGEN3_FOUND)

  add_executable(Lrmat_SVD_tests lrmat_SVD_tests.cpp)
  target_link_libraries(Lrmat_SVD_tests htool
                            ${EIGEN3_LIBRARIES})
  target_include_directories(Lrmat_SVD_tests PUBLIC
                            ${EIGEN3_INCLUDE_DIR})
  add_dependencies(build-tests Lrmat_SVD_tests)
  add_test(Lrmat_SVD_tests Lrmat_SVD_tests)
endif()
