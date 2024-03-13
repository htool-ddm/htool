#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_custom_target(build-tests-lrmat-build)
add_dependencies(build-tests-lrmat build-tests-lrmat-build)
add_subdirectory(lrmat_build)

add_custom_target(build-tests-lrmat-product)
add_dependencies(build-tests-lrmat build-tests-lrmat-product)
add_subdirectory(lrmat_product)

add_custom_target(build-tests-lrmat-addition)
add_dependencies(build-tests-lrmat build-tests-lrmat-addition)
add_subdirectory(lrmat_addition)
