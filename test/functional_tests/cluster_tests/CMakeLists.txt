#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Cluster_tests cluster_tests.cpp)
target_link_libraries(Cluster_tests htool)
add_dependencies(build-tests Cluster_tests)
add_test(Cluster_tests Cluster_tests)
