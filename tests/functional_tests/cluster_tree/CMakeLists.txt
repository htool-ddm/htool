#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Test_cluster_tree test_cluster_tree.cpp)
target_link_libraries(Test_cluster_tree htool)
add_dependencies(build-tests Test_cluster_tree)
add_test(Test_cluster_tree Test_cluster_tree)
