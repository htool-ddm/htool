#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Hpddm_solve_tests hpddm_solve_tests.cpp)
target_link_libraries(Hpddm_solve_tests htool)
add_dependencies(build-tests Hpddm_solve_tests)

set(HPDDM_SOLVE_TESTS_ARGS ${CMAKE_SOURCE_DIR}/data/data_test/SPD_mat_example.bin
              ${CMAKE_SOURCE_DIR}/data/data_test/rhs.bin
              ${CMAKE_SOURCE_DIR}/data/data_test/gmsh_mesh.msh
              ${CMAKE_SOURCE_DIR}/data/data_test/sol.bin)
message(${HPDDM_SOLVE_TESTS_ARGS})
add_test(NAME Hpddm_solve_tests_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${CMAKE_CURRENT_BINARY_DIR}/Hpddm_solve_tests ${HPDDM_SOLVE_TESTS_ARGS} )
add_test(NAME Hpddm_solve_tests_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${CMAKE_CURRENT_BINARY_DIR}/Hpddm_solve_tests ${HPDDM_SOLVE_TESTS_ARGS})
add_test(NAME Hpddm_solve_tests_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${CMAKE_CURRENT_BINARY_DIR}/Hpddm_solve_tests ${HPDDM_SOLVE_TESTS_ARGS})
