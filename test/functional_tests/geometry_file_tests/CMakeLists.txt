#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Geometry_file_tests geometry_file_tests.cpp)
target_link_libraries(Geometry_file_tests htool)
add_dependencies(build-tests Geometry_file_tests)
add_test(NAME Geometry_file_tests
COMMAND Geometry_file_tests ${CMAKE_SOURCE_DIR}/data/data_test/gmsh_test.msh)
