#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

set(cluster_types "geometric_splitting")
list(APPEND cluster_types "regular_splitting")
# list(APPEND cluster_type "sympartialACA")
# list(APPEND cluster_type "SVD")

foreach(cluster_type ${cluster_types})
    add_executable(Test_cluster_${cluster_type} test_cluster_${cluster_type}.cpp)
    target_link_libraries(Test_cluster_${cluster_type} htool)
    add_dependencies(build-tests Test_cluster_${cluster_type})

    add_test(NAME Test_cluster_${cluster_type}_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_cluster_${cluster_type})

    add_test(NAME Test_cluster_${cluster_type}_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_cluster_${cluster_type})
    # Expect failure because 2 % 4 != 0
    set_tests_properties(Test_cluster_${cluster_type}_2 PROPERTIES WILL_FAIL TRUE) 

    add_test(NAME Test_cluster_${cluster_type}_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_cluster_${cluster_type})

endforeach()



add_executable(Test_cluster_tree test_cluster_tree.cpp)
target_link_libraries(Test_cluster_tree htool)
add_dependencies(build-tests Test_cluster_tree)
add_test(Test_cluster_tree Test_cluster_tree)
