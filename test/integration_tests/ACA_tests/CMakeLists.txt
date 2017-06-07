#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#


if (EIGEN3_FOUND)

  add_executable(ACA_tests ACA_tests.cpp)
  target_link_libraries(ACA_tests htool
                            ${EIGEN3_LIBRARIES})
  target_include_directories(ACA_tests PUBLIC
                            ${EIGEN3_INCLUDE_DIR})
endif()
