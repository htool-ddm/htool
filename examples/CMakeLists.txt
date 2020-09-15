#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Smallest_example smallest_example.cpp)
target_link_libraries(Smallest_example htool)
add_dependencies(build-examples Smallest_example)

add_executable(VisuCluster visucluster.cpp)
target_link_libraries(VisuCluster htool)
add_dependencies(build-examples VisuCluster)

add_executable(Compression_comparison compression_comparison.cpp)
target_link_libraries(Compression_comparison htool)
add_dependencies(build-examples Compression_comparison)
