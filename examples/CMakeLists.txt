#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Use_clustering use_clustering.cpp)
target_link_libraries(Use_clustering htool)
add_dependencies(build-examples Use_clustering)

add_executable(Use_hmatrix use_hmatrix.cpp)
target_link_libraries(Use_hmatrix htool)
add_dependencies(build-examples Use_hmatrix)

add_executable(Use_distributed_operator use_distributed_operator.cpp)
target_link_libraries(Use_distributed_operator htool)
add_dependencies(build-examples Use_distributed_operator)

add_executable(Use_ddm_solver use_ddm_solver.cpp)
target_link_libraries(Use_ddm_solver htool)
add_dependencies(build-examples Use_ddm_solver)

add_executable(Compression_comparison compression_comparison.cpp)
target_link_libraries(Compression_comparison htool)
add_dependencies(build-examples Compression_comparison)
