#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Test_blocks_RjasanowSteinbach test_blocks_RjasanowSteinbach.cpp)
target_link_libraries(Test_blocks_RjasanowSteinbach htool)
add_dependencies(build-tests Test_blocks_RjasanowSteinbach)

add_test(NAME Test_blocks_RjasanowSteinbach_1 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_blocks_RjasanowSteinbach)

add_test(NAME Test_blocks_RjasanowSteinbach_2 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_blocks_RjasanowSteinbach)

add_test(NAME Test_blocks_RjasanowSteinbach_3 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_blocks_RjasanowSteinbach)

add_test(NAME Test_blocks_RjasanowSteinbach_4 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_blocks_RjasanowSteinbach)

add_executable(Test_blocks_RjasanowSteinbach_symmetric test_blocks_RjasanowSteinbach_symmetric.cpp)
target_link_libraries(Test_blocks_RjasanowSteinbach_symmetric htool)
add_dependencies(build-tests Test_blocks_RjasanowSteinbach_symmetric)

add_test(NAME Test_blocks_RjasanowSteinbach_symmetric_1 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_blocks_RjasanowSteinbach_symmetric)

add_test(NAME Test_blocks_RjasanowSteinbach_symmetric_2 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_blocks_RjasanowSteinbach_symmetric)

add_test(NAME Test_blocks_RjasanowSteinbach_symmetric_3 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_blocks_RjasanowSteinbach_symmetric)

add_test(NAME Test_blocks_RjasanowSteinbach_symmetric_4 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_blocks_RjasanowSteinbach_symmetric)
