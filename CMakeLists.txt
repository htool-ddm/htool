#=============================================================================#
#============================== Project ======================================#
#=============================================================================#
cmake_minimum_required(VERSION 2.6)
Project(ElastoPhi)

if ("${CMAKE_VERSION}" VERSION_LESS 3.1)
	add_compile_options(-std=c++11)
else ()
	set(CMAKE_CXX_STANDARD 11)
	set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif ()

# Files to do find_package for some module
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake_modules")

# Useful variables
set(htool_include_dir "${CMAKE_CURRENT_SOURCE_DIR}/include")




#=============================================================================#
#=========================== Libraries =======================================#
#=============================================================================#
# Eigen package
find_package (Eigen3 REQUIRED)

# MPI
find_package (MPI REQUIRED)

# BLAS
find_package(BLAS REQUIRED)

#OpenGL
find_package(OpenGL)
if(NOT OPENGL_FOUND)
    message(ERROR " OPENGL not found!")
endif(NOT OPENGL_FOUND)

#FreeGlut
find_package(GLUT)
if(NOT GLUT_FOUND)
    message(ERROR " GLUT not found!")
endif(NOT GLUT_FOUND)

# #Glui
# find_package(GLUI)
# if(NOT GLUI_FOUND)
#     message(ERROR " GLUI not found!")
# endif(NOT GLUI_FOUND)

configure_file(${htool_include_dir}/config.h.in ${CMAKE_BINARY_DIR}/generated/config.h )
include_directories( ${CMAKE_BINARY_DIR}/generated/ )

#=============================================================================#
#========================== Repertories ======================================#
#=============================================================================#
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
