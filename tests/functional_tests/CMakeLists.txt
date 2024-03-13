add_custom_target(build-tests-basic-types)
add_dependencies(build-tests build-tests-basic-types)
add_subdirectory(basic_types)

add_custom_target(build-tests-misc)
add_dependencies(build-tests build-tests-misc)
add_subdirectory(misc)

add_custom_target(build-tests-clustering)
add_dependencies(build-tests build-tests-clustering)
add_subdirectory(clustering)

add_custom_target(build-tests-matrix)
add_dependencies(build-tests build-tests-matrix)
add_subdirectory(matrix)

add_custom_target(build-tests-hmatrix)
add_dependencies(build-tests build-tests-hmatrix)
add_subdirectory(hmatrix)
# add_subdirectory(multilrmat)

add_custom_target(build-tests-distributed-operator)
add_dependencies(build-tests build-tests-distributed-operator)
add_subdirectory(distributed_operator)

add_custom_target(build-tests-solvers)
add_dependencies(build-tests build-tests-solvers)
add_subdirectory(solvers)

# add_subdirectory(types)
