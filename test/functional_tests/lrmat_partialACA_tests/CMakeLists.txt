#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Lrmat_partialACA_tests lrmat_partialACA_tests.cpp)
target_link_libraries(Lrmat_partialACA_tests htool)
add_dependencies(build-tests Lrmat_partialACA_tests)
add_test(Lrmat_partialACA_tests Lrmat_partialACA_tests)
