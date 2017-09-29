#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Test_cluster test_cluster.cpp)
target_link_libraries(Test_cluster htool)
add_dependencies(build-tests Test_cluster)
add_test(Test_cluster Test_cluster)
