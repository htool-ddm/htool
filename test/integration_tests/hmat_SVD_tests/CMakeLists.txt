#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

if (EIGEN3_FOUND)
  add_executable(Hmat_SVD_tests hmat_SVD_tests.cpp)
  target_link_libraries(Hmat_SVD_tests htool
                                ${EIGEN3_LIBRARIES})
  target_include_directories(Hmat_SVD_tests PUBLIC
                              ${EIGEN3_INCLUDE_DIR})
endif()
