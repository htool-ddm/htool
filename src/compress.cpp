#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include <cassert>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <mpi.h>

#include <htool/htool.hpp>


using namespace std;
using namespace htool;


/*
class MyMatrix: public VirtualMatrix{

private:

	Cplx (*getcoef)(const int&, const int&);

public:

	MyMatrix(const int& anr, const int& anc,Cplx (*fctptr)(const int&, const int&)): getcoef(fctptr){
		nr = anr;
		nc = anc;
	}

	Cplx operator()(const int& j, const int& k){
		return getcoef(j,k);
	}
	const Cplx operator()(const int& j, const int& k) const{
		return getcoef(j,k);
	}
};
*/

class MyMatrix: public VirtualMatrix{

public:

	const Matrix& A;

	MyMatrix(const Matrix& mat): A(mat){
		nr = nb_rows(mat);
		nc = nb_cols(mat);
	}

	const Cplx get_coef(const int& j, const int& k) const{
		return A(j,k);
	}
};

/**************************************************************************//**
* It builds the hierarchical matrix with compressed and dense blocks,
* computes the consistency error for a matrix vector product and
* the relative error in Frobenius norm with respect to the dense matrix.
*
* (To be run it requires the input file with the desidered parameters)
*****************************************************************************/

int main(int argc, char* argv[]){

    MPI_Init(&argc, &argv);
    /*# Init #*/
    int rankWorld, sizeWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);

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

	if (rankWorld == 0) {
		cout<<"############# Inputs #############"<<endl;
		cout<<"Eta : "+NbrToStr(GetEta())<<endl;
		cout<<"Epsilon : "+NbrToStr(GetEpsilon())<<endl;
		cout<<"MinClusterSize : "+NbrToStr(GetMinClusterSize())<<endl;
		cout<<"Output path : "+GetOutputPath()<<endl;
		cout<<"Mesh path : "+GetMeshPath()<<endl;
		cout<<"Matrix path : "+GetMatrixPath()<<endl;
		cout<<"##################################"<<endl;
	}

	//////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////    Build Hmatrix 	////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////
	//vector<double> times;
	vectReal r;
	vectR3   x;
	Matrix   A;
	//tic();

	// LoadMatrix(GetMatrixPath().c_str(),A);
	bytes_to_matrix(GetMatrixPath().c_str(),A);
	LoadPoints(GetMeshPath().c_str(),x,r);
	vectInt tab(nb_rows(A));
	for (int j=0;j<x.size();j++){
		tab[3*j]  = j;
		tab[3*j+1]= j;
		tab[3*j+2]= j;
	}
	//toc();
	//tic();

	//if (rankWorld == 0)
		// matrix_to_bytes(A, "../data/matrice450Fracs.bin");


	//MyMatrix mA(A);
	//HMatrix B(mA,x,r,tab);

	HMatrix B(A,x,r,tab);

	//toc();


	//////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////     Errors 	//////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////

	// Vecteur
	int nr  = nb_rows(A);
	vectCplx u(nr);
	int NbSpl = 1000;
	double du = 5./double(NbSpl);
	srand (1);
	for(int j=0; j<nr; j++){
		int n = rand()%(NbSpl+1);
		u[j] = n*du;}

	vectCplx ua(nr),ub(nr);
	MvProd(ua,A,u);
	std::pair <double,double > mvp_stats= MvProdMPI(ub,B,u);
	Real normA = NormFrob(A);

	add_stats(B,"MvProd (mean)",get<0>(mvp_stats));
	add_stats(B,"MvProd (max)",get<1>(mvp_stats));
	add_stats(B,"MvProd err",norm(ua-ub)/norm(ua));
	add_stats(B,"Compression",CompressionRate(B));
	add_stats(B,"Nb dense mats",nb_densemats(B));
	add_stats(B,"Nb lr mats",nb_lrmats(B));
	add_stats(B,"Relative Frob error",sqrt(squared_absolute_error(B,A))/normA);

	print_stats(B);

	MPI_Finalize();
	return 0;
}
