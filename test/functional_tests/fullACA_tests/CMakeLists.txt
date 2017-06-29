#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(FullACA_tests fullACA_tests.cpp)
target_link_libraries(FullACA_tests htool)
add_dependencies(build-tests FullACA_tests)
add_test(FullACA_tests FullACA_tests)
