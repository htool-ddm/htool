#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Hmat_fullACA_vec_prod_scaling hmat_fullACA_vec_prod_scaling.cpp)
target_link_libraries(Hmat_fullACA_vec_prod_scaling htool)

add_executable(Hmat_partialACA_vec_prod_scaling hmat_partialACA_vec_prod_scaling.cpp)
target_link_libraries(Hmat_partialACA_vec_prod_scaling htool)
