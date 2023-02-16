#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_custom_target(build-tests-basic-types-matrix-product)
add_dependencies(build-tests-basic-types build-tests-basic-types-matrix-product)
add_subdirectory(matrix_product)

add_executable(Test_basic_types_matrix test_basic_types_matrix.cpp)
target_link_libraries(Test_basic_types_matrix htool)
add_dependencies(build-tests-basic-types Test_basic_types_matrix)
add_test(Test_basic_types_matrix Test_basic_types_matrix)

add_executable(Test_basic_types_matrix_file test_basic_types_matrix_file.cpp)
target_link_libraries(Test_basic_types_matrix_file htool)
add_dependencies(build-tests-basic-types Test_basic_types_matrix_file)
add_test(Test_basic_types_matrix_file Test_basic_types_matrix_file)

add_executable(Test_basic_types_point test_basic_types_point.cpp)
target_link_libraries(Test_basic_types_point htool)
add_dependencies(build-tests-basic-types Test_basic_types_point)
add_test(Test_basic_types_point Test_basic_types_point)

add_executable(Test_basic_types_vector test_basic_types_vector.cpp)
target_link_libraries(Test_basic_types_vector htool)
add_dependencies(build-tests-basic-types Test_basic_types_vector)
add_test(Test_basic_types_vector Test_basic_types_vector)

add_executable(Test_basic_types_vector_file test_basic_types_vector_file.cpp)
target_link_libraries(Test_basic_types_vector_file htool)
add_dependencies(build-tests-basic-types Test_basic_types_vector_file)
add_test(Test_basic_types_vector_file Test_basic_types_vector_file)
