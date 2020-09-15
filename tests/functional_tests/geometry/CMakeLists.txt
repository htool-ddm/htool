#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Test_geometry_file test_geometry_file.cpp)
target_link_libraries(Test_geometry_file htool)
add_dependencies(build-tests Test_geometry_file)
add_test(NAME Test_geometry_file COMMAND Test_geometry_file ${CMAKE_SOURCE_DIR}/data/data_test/gmsh_test.msh)
