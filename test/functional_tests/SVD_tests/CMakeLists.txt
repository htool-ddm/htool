#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#


if (EIGEN3_FOUND)

  add_executable(SVD_tests SVD_tests.cpp)
  target_link_libraries(SVD_tests htool
                            ${EIGEN3_LIBRARIES})
  target_include_directories(SVD_tests PUBLIC
                            ${EIGEN3_INCLUDE_DIR})
  add_dependencies(build-tests SVD_tests)
  add_test(SVD_tests SVD_tests)
endif()
