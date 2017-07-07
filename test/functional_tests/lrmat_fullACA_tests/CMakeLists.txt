#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Lrmat_fullACA_tests lrmat_fullACA_tests.cpp)
target_link_libraries(Lrmat_fullACA_tests htool)
add_dependencies(build-tests Lrmat_fullACA_tests)
add_test(Lrmat_fullACA_tests Lrmat_fullACA_tests)
