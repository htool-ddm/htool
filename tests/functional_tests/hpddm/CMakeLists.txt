#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Test_hpddm_solve test_hpddm_solve.cpp)
target_link_libraries(Test_hpddm_solve htool)
add_dependencies(build-tests Test_hpddm_solve)

set(Test_hpddm_solve_ARGS ${CMAKE_SOURCE_DIR}/data/data_test/SPD_mat_example.bin
              ${CMAKE_SOURCE_DIR}/data/data_test/rhs.bin
              ${CMAKE_SOURCE_DIR}/data/data_test/gmsh_mesh.msh
              ${CMAKE_SOURCE_DIR}/data/data_test/sol.bin)

add_test(NAME Test_hpddm_solve_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${CMAKE_CURRENT_BINARY_DIR}/Test_hpddm_solve ${Test_hpddm_solve_ARGS} )
add_test(NAME Test_hpddm_solve_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${CMAKE_CURRENT_BINARY_DIR}/Test_hpddm_solve ${Test_hpddm_solve_ARGS})
add_test(NAME Test_hpddm_solve_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${CMAKE_CURRENT_BINARY_DIR}/Test_hpddm_solve ${Test_hpddm_solve_ARGS})


add_executable(Test_hpddm_solve_multi_rhs test_hpddm_solve_multi_rhs.cpp)
target_link_libraries(Test_hpddm_solve_multi_rhs htool)
add_dependencies(build-tests Test_hpddm_solve_multi_rhs)

set(Test_hpddm_solve_ARGS ${CMAKE_SOURCE_DIR}/data/data_test/SPD_mat_example.bin
              ${CMAKE_SOURCE_DIR}/data/data_test/rhs.bin
              ${CMAKE_SOURCE_DIR}/data/data_test/gmsh_mesh.msh
          ${CMAKE_SOURCE_DIR}/data/data_test/sol.bin 10)

add_test(NAME Test_hpddm_solve_multi_rhs_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${CMAKE_CURRENT_BINARY_DIR}/Test_hpddm_solve_multi_rhs ${Test_hpddm_solve_ARGS} )
add_test(NAME Test_hpddm_solve_multi_rhs_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${CMAKE_CURRENT_BINARY_DIR}/Test_hpddm_solve_multi_rhs ${Test_hpddm_solve_ARGS})
add_test(NAME Test_hpddm_solve_multi_rhs_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${CMAKE_CURRENT_BINARY_DIR}/Test_hpddm_solve_multi_rhs ${Test_hpddm_solve_ARGS})
