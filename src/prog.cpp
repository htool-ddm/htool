#include <iostream>
#include <complex>
#include <vector>
#include <cassert>
#include <htool/htool.hpp>


#include <stdio.h>
#include <stdlib.h>
#include <time.h>


using namespace std;

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
	LoadParam(inputname);
	LoadParamIO(inputname);
	cout<<"############# Inputs #############"<<endl;
	cout<<"Eta : "+NbrToStr(GetEta())<<endl;
	cout<<"Epsilon : "+NbrToStr(GetEpsilon())<<endl;
	cout<<"Data path : "+GetDataPath()<<endl;
	cout<<"Output path : "+GetOutputPath()<<endl;
	cout<<"Mesh name : "+GetMeshName()<<endl;
	cout<<"Matrix name : "+GetMatrixName()<<endl;
	cout<<"##################################"<<endl;
	
	////////////////========================================================////////////////
	
	
}
