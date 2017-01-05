cmake_minimum_required(VERSION 2.6)
Project(ElastoPhi)

if ("${CMAKE_VERSION}" VERSION_LESS 3.1)
	add_compile_options(-std=c++11)
else ()
	set(CMAKE_CXX_STANDARD 11)
	set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif ()

# List the headers and add them into the project
FILE(GLOB_RECURSE MY_HEADERS "include/*.hpp")
include_directories(include)

# Eigen package
find_package (Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package (MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# BLAS
find_package(BLAS REQUIRED)
	
# Add sources file
add_subdirectory(src)

# # Add test files
# add_subdirectory(test)
# 
# 
# # Testing project
# enable_testing()
# add_test(NAME lrmatrix_test COMMAND Test_lrmatrix)
# add_test(NAME hmatrix_test COMMAND Test_hmatrix)
# add_test(NAME cluster_test COMMAND Test_cluster) 
# # Add documentation
# add_subdirectory(doc)
