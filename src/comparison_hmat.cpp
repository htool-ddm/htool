//#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include <cassert>
#include <htool/htool.hpp>
#include <htool/loading.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>


using namespace std;
using namespace htool;

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

  // Initialize the MPI environment
	MPI_Init(NULL, NULL);

	// Get the number of processes
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// Get the rank of the process
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Load the inputs
	string inputname = argv[1];
	LoadParamIO(inputname);
  SetNdofPerElt(3);
	SetMinClusterSize(3);
	SetMaxBlockSize(1000000);
	cout<<"############# Inputs #############"<<endl;
	cout<<"Output path : "+GetOutputPath()<<endl;
	cout<<"Mesh path : "+GetMeshPath()<<endl;
	cout<<"Matrix path : "+GetMatrixPath()<<endl;
	cout<<"##################################"<<endl;

  string outputSubfolderpathname = GetOutputPath()+"/output_"+split(GetMatrixName(),'.').at(0);
  system(("mkdir "+outputSubfolderpathname).c_str()); // create the outputh subdirectory



	vector<R3>   x;
	Matrix<double>   A;
  vector<double> r;

	tic();
	LoadPoints(GetMeshPath().c_str(),x,r);
	toc();
	tic();
	// LoadMatrix(GetMatrixPath().c_str(),A);

	A.bytes_to_matrix(GetMatrixPath().c_str());
	toc();
std::cout << "ok" << std::endl;
	// // Vecteur pour le produit matrice vecteur
	// int nr  = nb_rows(A);
	// vector<Cplx> u(nr);
	// int NbSpl = 1000;
	// double du = 5./double(NbSpl);
	// srand (1); // !! pour reproducibilite'
	// for(int j=0; j<nr; j++){
	// 	int n = rand()%(NbSpl+1);
	// 	u[j] = n*du;}
  //
	// vector<Cplx> ua(nr);
	// MvProd(ua,A,u);

	tic();
	double normA = normFrob(A);
	toc();
	cout << "Frobenius norm of the dense matrix: " << normA << endl;

	// Vecteur renvoyant pour chaque dof l'indice de l'entite geometrique correspondante dans x
	vector<int> tab(A.nb_rows());
	for (int j=0;j<x.size();j++){
		tab[3*j]  = j;
		tab[3*j+1]= j;
		tab[3*j+2]= j;
	}

	// Values of eta and epsilon
	const int neta = 2;
	double eta [neta] = {1e1, 1e0};
  const int nepsilon = 6;
  double epsilon[nepsilon] = {-1, 1e0, 9e-1, 5e-1, 1e-1, 1e-2};
    //{-1, 1e0, 7e-1, 5e-1, 3e-1, 1e-1, 7e-2, 5e-2, 3e-2, 1e-2, 9e-1};

	// for output file
	//string filename=GetOutputPath()+"/output_compression_18_08_2016"+GetMatrixName();
  string filename=outputSubfolderpathname+"/output_compression_26_09_2017"+split(GetMatrixName(),'.').at(0)+".txt";
	ofstream output(filename.c_str());
	if (!output){
		cerr<<"Output file cannot be created"<<endl;
		exit(1);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank==0){
  	output<< "#Eta "<<"Epsilon "<<"Compression "<<"Erreur_MvProd "<<"Erreur_Frob "<<"Time_hmatrix "<<"Time_MvProd"<<endl;
	}

//	for(int iepsilon=0; iepsilon<nepsilon; iepsilon++)
//	{
//		cout << "iepsilon: " << iepsilon << endl;
  for(int ieta=0; ieta<neta; ieta++) {
      // cout << "ieta: " << ieta << endl;
      for(int iepsilon=0; iepsilon<nepsilon; iepsilon++){
        // cout << "iepsilon: " << iepsilon << endl;

  			SetEta(eta[ieta]);
  			SetEpsilon(epsilon[iepsilon]);
  			// cout<<"Eta : "+NbrToStr(GetEta())<<endl;
  			// cout<<"Epsilon : "+NbrToStr(GetEpsilon())<<endl;

			  vector<double> times;

  			tic();
  			// Build the hierarchical matrix with compressed and dense blocks
  			HMatrix<partialACA,double> B(A,x,r,tab,GetEpsilon()==-1 ? 0 : -1);
  			// if epsilon=-1 rank 0 blocks, otherwise aca compression with the given precision
  			toc(times);

			// vectCplx ub(nr);
			// tic();
			// MvProd(ub,B,u); // Do the matrix vector product
			// toc(times);

			// Real errH = norm(ua-ub)/norm(ua);
			// cout << "Matrix-vector product relative error with HMatrix: " << errH << endl;

  			double compressionH=B.compression();
  			double froberrH = Frobenius_absolute_error(B,A)/normA;
				if (rank==0){
					cout << "Compression rate with HMatrix:" << compressionH << endl;
  				cout << "Relative error in Frobenius norm with HMatrix: " << froberrH << endl;
				}
			// write in output file
			//output<<GetEta()<<" "<<GetEpsilon()<<" "<<compressionH<<" "<<errH<<" "<<froberrH<<endl;
			if (rank==0)
      	output<<GetEta()<<" "<<GetEpsilon()<<" "<<compressionH<<" "<<1<<" "<<froberrH<<" "<<times[0]<<" "<<times[1]<<endl;

			B.print_stats();
		}
	}
	output.close();

  //Finalize the MPI environment.
	MPI_Finalize();

	return 0;
}
