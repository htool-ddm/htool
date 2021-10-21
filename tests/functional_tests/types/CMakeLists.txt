#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_custom_target(build-tests-virtual-hmatrix)
add_subdirectory(virtual_hmatrix)
add_dependencies(build-tests build-tests-virtual-hmatrix)

# add_subdirectory(multihmatrix)

add_executable(Test_matrix test_matrix.cpp)
target_link_libraries(Test_matrix htool)
add_dependencies(build-tests Test_matrix)
add_test(Test_matrix Test_matrix)

add_executable(Test_matrix_file test_matrix_file.cpp)
target_link_libraries(Test_matrix_file htool)
add_dependencies(build-tests Test_matrix_file)
add_test(Test_matrix_file Test_matrix_file)

add_executable(Test_point test_point.cpp)
target_link_libraries(Test_point htool)
add_dependencies(build-tests Test_point)
add_test(Test_point Test_point)

add_executable(Test_vector test_vector.cpp)
target_link_libraries(Test_vector htool)
add_dependencies(build-tests Test_vector)
add_test(Test_vector Test_vector)

add_executable(Test_vector_file test_vector_file.cpp)
target_link_libraries(Test_vector_file htool)
add_dependencies(build-tests Test_vector_file)
add_test(Test_vector_file Test_vector_file)
