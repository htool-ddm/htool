#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Test_lrmat_addition test_lrmat_addition.cpp)
target_link_libraries(Test_lrmat_addition htool)
add_dependencies(build-tests-lrmat-build Test_lrmat_addition)
add_test(Test_lrmat_addition Test_lrmat_addition)
