#include <iostream>
#include <complex>
#include <vector>
#include <cassert>
#include "point.hpp"
#include "matrix.hpp"
#include "lrmat.hpp"
#include "cluster.hpp"
#include "hmatrix.hpp"
#include "loading.hpp"
#include "export.hpp"
#include "user.hpp"
#include "parametres.hpp"


#include <stdio.h>
#include <stdlib.h>
#include <time.h>


using namespace std;

/**************************************************************************//**
* It reads the mesh file from the format given by Ibtihel to gmsh format (.msh)
* and adds a tag to color the different clusters at level 1, 2 and 3 of the 
* cluster tree.
 *****************************************************************************/

int main(int argc, char* argv[]){
	
	
	////////////////========================================================////////////////
	////////////////////////////////========  Input ========////////////////////////////////
	
	// Check the number of parameters
	if (argc < 2) {
		// Tell the user how to run the program
		cerr << "Usage: " << argv[0] << " input name" << endl;
		/* "Usage messages" are a conventional way of telling the user
		 * how to run a program if they enter the command incorrectly.
		 */
		return 1;
	}
	
	// Load the inputs
	string inputname = argv[1];
	LoadParamIO(inputname);
 
	cout<<"############# Inputs #############"<<endl;
	cout<<"Output path : "+GetOutputPath()<<endl;
	cout<<"Mesh path : "+GetMeshPath()<<endl;
	cout<<"##################################"<<endl;
 
	////////////////========================================================////////////////
	
	vectReal r;
	vectR3   x;
	tic();
	LoadPoints((GetMeshPath()).c_str(),x,r);
	vectInt tab(GetNdofPerElt()*x.size());
	for (int j=0;j<x.size();j++){
		tab[3*j]  = j;
		tab[3*j+1]= j;
		tab[3*j+2]= j;
	}
	toc();
	tic();
	Cluster t(x,r,tab);
	toc();
	
	for(int idepth=1; idepth<4; idepth++){
		VisuPartitionedMesh(t, GetDataPath()+"/"+GetMeshName(), "visu_cluster_"+(split(GetMeshName(),'.')).at(0)+"depth"+NbrToStr(idepth)+".msh", idepth);
	}
	
	
}
