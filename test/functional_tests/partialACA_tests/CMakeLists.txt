#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(PartialACA_tests partialACA_tests.cpp)
target_link_libraries(PartialACA_tests htool)
add_dependencies(build-tests PartialACA_tests)
add_test(PartialACA_tests PartialACA_tests)
