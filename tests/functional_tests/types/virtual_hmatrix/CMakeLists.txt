#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_custom_target(build-tests-virtual-hmatrix-hmatrix)
add_subdirectory(hmatrix)
add_dependencies(build-tests-virtual-hmatrix build-tests-virtual-hmatrix-hmatrix)
