#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

set(cluster_types "geometric_splitting")
list(APPEND cluster_types "regular_splitting")

set(compression_types "SVD")
list(APPEND compression_types "partialACA")
list(APPEND compression_types "fullACA")
list(APPEND compression_types "sympartialACA")



foreach(compression_type ${compression_types})

    foreach(cluster_type ${cluster_types})

        add_executable(Test_hmat_auto_${cluster_type}_${compression_type} test_hmat_auto_${cluster_type}_${compression_type}.cpp)
        target_link_libraries(Test_hmat_auto_${cluster_type}_${compression_type} htool)
        add_dependencies(build-tests Test_hmat_auto_${cluster_type}_${compression_type})
        add_test(NAME Test_hmat_auto_${cluster_type}_${compression_type}_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_auto_${cluster_type}_${compression_type})
        add_test(NAME Test_hmat_auto_${cluster_type}_${compression_type}_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_auto_${cluster_type}_${compression_type})
        add_test(NAME Test_hmat_auto_${cluster_type}_${compression_type}_3 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_auto_${cluster_type}_${compression_type})
        add_test(NAME Test_hmat_auto_${cluster_type}_${compression_type}_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_auto_${cluster_type}_${compression_type})
        
        add_executable(Test_hmat_${cluster_type}_${compression_type} test_hmat_${cluster_type}_${compression_type}.cpp)
        target_link_libraries(Test_hmat_${cluster_type}_${compression_type} htool)
        add_dependencies(build-tests Test_hmat_${cluster_type}_${compression_type})
        add_test(NAME Test_hmat_${cluster_type}_${compression_type}_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_${cluster_type}_${compression_type})
        add_test(NAME Test_hmat_${cluster_type}_${compression_type}_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_${cluster_type}_${compression_type})
        add_test(NAME Test_hmat_${cluster_type}_${compression_type}_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_${cluster_type}_${compression_type})

    endforeach()
endforeach()

#=== hmat_vec_prod
add_executable(Test_hmat_vec_prod test_hmat_vec_prod.cpp)
target_link_libraries(Test_hmat_vec_prod htool)
add_dependencies(build-tests Test_hmat_vec_prod)
add_test(NAME Test_hmat_vec_prod_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_vec_prod)
add_test(NAME Test_hmat_vec_prod_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_vec_prod)
add_test(NAME Test_hmat_vec_prod_3 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_vec_prod)
add_test(NAME Test_hmat_vec_prod_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_vec_prod)

add_executable(Test_hmat_symmetry_vec_prod test_hmat_symmetry_vec_prod.cpp)
target_link_libraries(Test_hmat_symmetry_vec_prod htool)
add_dependencies(build-tests Test_hmat_symmetry_vec_prod)
add_test(NAME Test_hmat_symmetry_vec_prod_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_vec_prod)
add_test(NAME Test_hmat_symmetry_vec_prod_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_vec_prod)
add_test(NAME Test_hmat_symmetry_vec_prod_3 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_vec_prod)
add_test(NAME Test_hmat_symmetry_vec_prod_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_vec_prod)

#=== hmat_mat_prod
add_executable(Test_hmat_mat_prod test_hmat_mat_prod.cpp)
target_link_libraries(Test_hmat_mat_prod htool)
add_dependencies(build-tests Test_hmat_mat_prod)
add_test(NAME Test_hmat_mat_prod_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_mat_prod)
add_test(NAME Test_hmat_mat_prod_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_mat_prod)
add_test(NAME Test_hmat_mat_prod_3 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_mat_prod)
add_test(NAME Test_hmat_mat_prod_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_mat_prod)

add_executable(Test_hmat_symmetry_mat_prod test_hmat_symmetry_mat_prod.cpp)
target_link_libraries(Test_hmat_symmetry_mat_prod htool)
add_dependencies(build-tests Test_hmat_symmetry_mat_prod)
add_test(NAME Test_hmat_symmetry_mat_prod_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_symmetry_mat_prod)
add_test(NAME Test_hmat_symmetry_mat_prod_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_symmetry_mat_prod)
add_test(NAME Test_hmat_symmetry_mat_prod_3 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_symmetry_mat_prod)
add_test(NAME Test_hmat_symmetry_mat_prod_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_symmetry_mat_prod)

#=== hmat_to_dense
add_executable(Test_hmat_to_dense test_hmat_to_dense.cpp)
target_link_libraries(Test_hmat_to_dense htool)
add_dependencies(build-tests Test_hmat_to_dense)
add_test(NAME Test_hmat_to_dense COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_to_dense)

#=== hmat_save
add_executable(Test_hmat_save test_hmat_save.cpp)
target_link_libraries(Test_hmat_save htool)
add_dependencies(build-tests Test_hmat_save)
add_test(NAME Test_hmat_save COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmat_save)
