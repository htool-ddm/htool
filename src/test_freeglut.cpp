//#include <OpenGL/gl.h>
//#include <OpenGL/glu.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include <GL/glui.h>

#include <iostream>
#include <fstream>

#include "mpi.h"
#include "htool.hpp"
#include "view.hpp"

using namespace std;
using namespace htool;

//Main program

int main(int argc, char **argv) {
	
	MPI_Init(&argc, &argv);
    /*# Init #*/
    int rankWorld, sizeWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
	
	std::vector<R3>  X;
	std::vector<N4>  Elts;
	std::vector<int> NbPts;
	
    vectReal r;
    vectR3   x;  
    Matrix   A;
    
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
