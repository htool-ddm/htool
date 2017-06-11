#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#


if (EIGEN3_FOUND)

  add_executable(FullACA_tests fullACA_tests.cpp)
  target_link_libraries(FullACA_tests htool
                            ${EIGEN3_LIBRARIES})
  target_include_directories(FullACA_tests PUBLIC
                            ${EIGEN3_INCLUDE_DIR})
endif()
