#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Test_logger test_logger.cpp)
target_link_libraries(Test_logger htool)
add_dependencies(build-tests-misc Test_logger)
add_test(Test_logger Test_logger)
