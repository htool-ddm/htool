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

#Glui
find_package(GLUI)
# if(NOT GLUT_FOUND)
#     message(ERROR " GLUT not found!")
# endif(NOT GLUT_FOUND)



#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#
add_executable(Compression compress.cpp)
target_include_directories(Compression PRIVATE
                            ${htool_include_dir}
                            ${EIGEN3_INCLUDE_DIR}
                            ${MPI_INCLUDE_PATH})
target_link_libraries (Compression
                            ${EIGEN3_LIBRARIES}
                            ${MPI_LIBRARIES}
                            ${BLAS_LIBRARIES})

add_executable(matrix_to_bytes matrix_to_bytes.cpp)
target_include_directories(matrix_to_bytes PRIVATE
                            ${htool_include_dir}
                            ${EIGEN3_INCLUDE_DIR}
                            ${MPI_INCLUDE_PATH})
target_link_libraries (matrix_to_bytes
                            ${MPI_LIBRARIES}
                            ${EIGEN3_LIBRARIES}
                            ${BLAS_LIBRARIES})


if(OPENGL_FOUND AND GLUT_FOUND)
add_executable(Test_freeGLUT test_freeglut.cpp)
target_link_libraries(Test_freeGLUT ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
endif(OPENGL_FOUND AND GLUT_FOUND)


# # Creation de l executable main a partir de prog
# add_executable(main prog.cpp ${MY_HEADERS})
# target_link_libraries (main  ${EIGEN3_LIBRARIES})
#
# # Creation de l executable CompaSparse a partir de comparison_sparse
# add_executable(CompaSparse comparison_sparse.cpp)
# target_link_libraries (CompaSparse ${EIGEN3_LIBRARIES})
#
# # Creation de l executable MultiCompression a partir de multi_compress.cpp
# add_executable(MultiCompression multi_compress.cpp)
# target_link_libraries (MultiCompression ${EIGEN3_LIBRARIES})
#
# # Creation de l executable Test_ACA a partir de test_ACA
# add_executable(Test_ACA test_ACA.cpp)
# target_link_libraries (Test_ACA  ${EIGEN3_LIBRARIES})
#
# # Creation de l executable ErrDecrease a partir de err_decrease
# add_executable(ErrDecrease err_decrease.cpp)
# target_link_libraries (ErrDecrease ${EIGEN3_LIBRARIES})
#
# # Creation de l executable VisuCluster a partir de visucluster
# add_executable(VisuCluster visucluster.cpp)
# target_link_libraries (VisuCluster ${EIGEN3_LIBRARIES})
#
# # Creation de l executable VisuMatrix a partir de visumatrix
# add_executable(VisuMatrix visumatrix.cpp)
# target_link_libraries (VisuMatrix ${EIGEN3_LIBRARIES})
#
# # Creation de l executable VisuMesh a partir de visumesh
# add_executable(VisuMesh visumesh.cpp)
# target_link_libraries (VisuMesh ${EIGEN3_LIBRARIES})

# add_executable(Test main_test.cpp test.cpp)
# target_include_directories(Test PRIVATE ${MPI_INCLUDE_PATH})
#
# target_link_libraries (Test ${MPI_LIBRARIES})
