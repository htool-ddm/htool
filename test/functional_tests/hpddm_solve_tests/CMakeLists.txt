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

add_test(NAME Hpddm_solve_tests_1 COMMAND  mpirun -np 1 ${CMAKE_CURRENT_BINARY_DIR}/Hpddm_solve_tests ${HPDDM_SOLVE_TESTS_ARGS} )
add_test(NAME Hpddm_solve_tests_2 COMMAND  mpirun -np 2 ${CMAKE_CURRENT_BINARY_DIR}/Hpddm_solve_tests ${HPDDM_SOLVE_TESTS_ARGS})
add_test(NAME Hpddm_solve_tests_4 COMMAND  mpirun -np 4 ${CMAKE_CURRENT_BINARY_DIR}/Hpddm_solve_tests ${HPDDM_SOLVE_TESTS_ARGS})
