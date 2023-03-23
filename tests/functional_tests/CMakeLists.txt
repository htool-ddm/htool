add_custom_target(build-tests-basic-types)
add_dependencies(build-tests build-tests-basic-types)
add_subdirectory(basic_types)

add_custom_target(build-tests-clustering)
add_dependencies(build-tests build-tests-clustering)
add_subdirectory(clustering)

# add_custom_target(build-tests-local-operators)
# add_dependencies(build-tests build-tests-local-operators)
# add_subdirectory(local_operators)

# add_subdirectory(blocks)

add_custom_target(build-tests-hmatrix)
add_dependencies(build-tests build-tests-hmatrix)
add_subdirectory(hmatrix)
# add_subdirectory(multilrmat)

add_custom_target(build-tests-distributed-operator)
add_dependencies(build-tests build-tests-distributed-operator)
add_subdirectory(distributed_operator)

# add_custom_target(build-tests-solvers)
# add_dependencies(build-tests build-tests-solvers)
# add_subdirectory(solvers)

# add_subdirectory(types)
