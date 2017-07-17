#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Point_tests point_tests.cpp)
target_link_libraries(Point_tests htool)
add_dependencies(build-tests Point_tests)
add_test(Point_tests Point_tests)
