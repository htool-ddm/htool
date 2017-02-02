//#include <OpenGL/gl.h>
//#include <OpenGL/glu.h>

//#define GLEW_STATIC
//#include <GL/glew.h>

#include <nanogui/nanogui.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <fstream>

#include "mpi.h"
#include "htool.hpp"
#include "view.hpp"

using namespace std;
using namespace htool;
using namespace nanogui;

int main(int argc, char **argv) {
	
	MPI_Init(&argc, &argv);
    /*# Init #*/
    int rankWorld, sizeWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    
    //string str = "../matrices/maillage3600FracsV1DN1.txt";
    //string strm = "../matrices/matrice3600FracsV1DN1.bin";
 
 	/*
    string str = "../matrices/maillage450Fracs.txt";
    string strm = "../matrices/matrice450Fracs.bin"; 
    
    bytes_to_matrix(strm,A);
    LoadPoints(str.c_str(),x,r);
    vectInt tab(nb_rows(A));
    for (int j=0;j<x.size();j++){
    	tab[3*j]  = j;
        tab[3*j+1]= j;
        tab[3*j+2]= j;
    }   

	Cluster t(x,r,tab);
	
	LoadMesh(str,X,Elts,NbPts);
	GLMesh m(X,Elts,NbPts);
	m.set_cluster(&t);
	*/
    
    Scene s;
	
	//s.add_mesh(m);
	
	s.init(&argc, argv);
	
	s.run();
    
    MPI_Finalize();
    return 0;
}
