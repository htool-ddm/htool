#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Test_cluster test_cluster.cpp)
target_link_libraries(Test_cluster htool)
add_dependencies(build-tests-clustering Test_cluster)

add_test(NAME Test_cluster_1 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_cluster)
set_tests_properties(Test_cluster_1 PROPERTIES LABELS "mpi")
add_test(NAME Test_cluster_2 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_cluster)
set_tests_properties(Test_cluster_2 PROPERTIES LABELS "mpi")
add_test(NAME Test_cluster_3 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_cluster)
set_tests_properties(Test_cluster_3 PROPERTIES LABELS "mpi")
add_test(NAME Test_cluster_4 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_cluster)
set_tests_properties(Test_cluster_4 PROPERTIES LABELS "mpi")

# set(cluster_types "pca")
# list(APPEND cluster_types "bounding_box")

# set(splitting_types "geometric_splitting")
# list(APPEND splitting_types "regular_splitting")

# foreach(cluster_type ${cluster_types})
#     foreach(splitting_type ${splitting_types})

#         add_executable(Test_cluster_${cluster_type}_${splitting_type} test_cluster_${cluster_type}_${splitting_type}.cpp)
#         target_link_libraries(Test_cluster_${cluster_type}_${splitting_type} htool)
#         add_dependencies(build-tests-clustering Test_cluster_${cluster_type}_${splitting_type})

#         add_test(NAME Test_cluster_${cluster_type}_${splitting_type}_1 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_cluster_${cluster_type}_${splitting_type})

#         add_test(NAME Test_cluster_${cluster_type}_${splitting_type}_2 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_cluster_${cluster_type}_${splitting_type})

#         add_test(NAME Test_cluster_${cluster_type}_${splitting_type}_4 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_cluster_${cluster_type}_${splitting_type})

#     endforeach()
# endforeach()
