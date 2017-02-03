#include <iostream>
#include <complex>
#include <vector>
#include <cassert>
#include <htool/htool.hpp>


#include <stdio.h>
#include <stdlib.h>
#include <time.h>


using namespace std;

/**************************************************************************//**
* It converts the mesh file from the format given by Ibtihel to gmsh format (.msh)
* and to medit format (.mesh) for visualization of the mesh
* (To be run it requires the input file with the desidered parameters)
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
	ExportGMSH(GetMeshPath(),"visu_"+(split(GetMeshName(),'.')).at(0)+".msh");
	
	ExportMEDIT(GetMeshPath(),"visu_"+(split(GetMeshName(),'.')).at(0)+".mesh");
	
}
