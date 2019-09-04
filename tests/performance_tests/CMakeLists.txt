add_custom_target(build-Hmat)

add_executable(Hmat_fullACA hmat_fullACA.cpp)
target_link_libraries(Hmat_fullACA htool)
add_dependencies(build-Hmat Hmat_fullACA)

add_executable(Hmat_partialACA hmat_partialACA.cpp)
target_link_libraries(Hmat_partialACA htool)
add_dependencies(build-Hmat Hmat_partialACA)
