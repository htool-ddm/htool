#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include <cassert>
#include <htool/htool.hpp>


#include <stdio.h>
#include <stdlib.h>
#include <time.h>


using namespace std;

/**************************************************************************//**
* It builds the hierarchical matrix with compressed and dense blocks,
* and produces an output file to visualize the compression of the matrix
* (use graphes_output_local_compression.py in postprocessing folder).
*
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
	LoadParam(inputname);
 
	cout<<"############# Inputs #############"<<endl;
	cout<<"Eta : "+NbrToStr(GetEta())<<endl;
	cout<<"Epsilon : "+NbrToStr(GetEpsilon())<<endl;
	cout<<"Output path : "+GetOutputPath()<<endl;
	cout<<"Mesh path : "+GetMeshName()<<endl;
	cout<<"Matrix path : "+GetMatrixName()<<endl;
	cout<<"##################################"<<endl;
 
	//////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////    Build Hmatrix 	//////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////
	vectReal r;
	vectR3   x;
	Matrix   A;
	tic();
	LoadMatrix((GetMatrixPath()).c_str(),A);
	LoadPoints((GetMeshPath()).c_str(),x,r);
	vectInt tab(nb_rows(A));
	for (int j=0;j<x.size();j++){
		tab[3*j]  = j;
		tab[3*j+1]= j;
		tab[3*j+2]= j;
	}
	toc();
	tic();
	HMatrix B(A,x,r,tab);
	toc();
	tic();
    
	//////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////    Create Output 	//////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////
    
    //Output(B, GetOutputPath()+"/output_local_comp_"+NbrToStr(GetEta())+"_"+NbrToStr(GetEpsilon())+"_"+GetMatrixName()); // to visualize the compression of the matrix
    
    string outputSubfolderpathname = GetOutputPath()+"/output_"+split(GetMatrixName(),'.').at(0);
    system(("mkdir "+outputSubfolderpathname).c_str()); // create the outputh subdirectory
    
    Output(B, outputSubfolderpathname+"/output_local_comp_"+NbrToStr(GetEta())+"_"+NbrToStr(GetEpsilon())+"_"+GetMatrixName()); // to visualize the compression of the matrix
    
	toc();
	
}

