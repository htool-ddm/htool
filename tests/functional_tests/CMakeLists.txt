add_custom_target(build-tests-basic-types)
add_dependencies(build-tests build-tests-basic-types)
add_subdirectory(basic_types)

add_custom_target(build-tests-clustering)
add_dependencies(build-tests build-tests-clustering)
add_subdirectory(clustering)

add_custom_target(build-tests-testing)
add_dependencies(build-tests build-tests-testing)
add_subdirectory(testing)

add_subdirectory(blocks)
add_subdirectory(hmatrix)
# add_subdirectory(multilrmat)

add_custom_target(build-tests-distributed-operator)
add_dependencies(build-tests build-tests-distributed-operator)
add_subdirectory(distributed_operator)

add_subdirectory(solvers)
add_subdirectory(types)
