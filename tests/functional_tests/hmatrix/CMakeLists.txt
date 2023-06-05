add_custom_target(build-tests-lrmat)
add_dependencies(build-tests build-tests-lrmat)
add_subdirectory(lrmat)

add_custom_target(build-tests-hmatrix-build)
add_dependencies(build-tests-hmatrix build-tests-hmatrix-build)
add_subdirectory(hmatrix_build)

add_custom_target(build-tests-hmatrix-product)
add_dependencies(build-tests-hmatrix build-tests-hmatrix-product)
add_subdirectory(hmatrix_product)

add_custom_target(build-tests-hmatrix-addition)
add_dependencies(build-tests-hmatrix build-tests-hmatrix-addition)
add_subdirectory(hmatrix_addition)

add_custom_target(build-tests-hmatrix-triangular-solve)
add_dependencies(build-tests-hmatrix build-tests-hmatrix-triangular-solve)
add_subdirectory(hmatrix_triangular_solve)

add_custom_target(build-tests-hmatrix-factorization)
add_dependencies(build-tests-hmatrix build-tests-hmatrix-factorization)
add_subdirectory(hmatrix_factorization)
