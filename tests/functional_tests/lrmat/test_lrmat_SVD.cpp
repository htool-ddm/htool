#include <iostream>
#include <complex>
#include <vector>

#include <htool/clustering/ncluster.hpp>
#include <htool/lrmat/SVD.hpp>
#include "test_lrmat.hpp"


using namespace std;
using namespace htool;


int main(int argc, char *argv[]){

	// Initialize the MPI environment
	MPI_Init(&argc,&argv);

	const int ndistance = 4;
	double distance[ndistance];
	distance[0] = 15; distance[1] = 20; distance[2] = 30; distance[3] = 40;

	SetNdofPerElt(1);
	SetEpsilon(0.0001);

	int nr=500;
	int nc=100;
	std::vector<R3> xt(nr);
	std::vector<R3> xs(nc);
	std::vector<int> tabt(500);
	std::vector<int> tabs(100);
	bool test =0;
	for(int idist=0; idist<ndistance; idist++)
	{
		
		create_geometry(distance[idist],xt,tabt,xs,tabs);

		GeometricClustering t,s; 

		std::vector<int> tabt(xt.size()),tabs(xs.size());
		std::iota(tabt.begin(),tabt.end(),int(0));
		std::iota(tabs.begin(),tabs.end(),int(0));
		t.build(xt,std::vector<double>(xt.size(),0),tabt,std::vector<double>(xt.size(),1));
		s.build(xs,std::vector<double>(xs.size(),0),tabs,std::vector<double>(xs.size(),1));

		MyMatrix A(xt,xs);

		// SVD fixed rank
    	int reqrank_max = 10;
		SVD<double,GeometricClustering> A_SVD_fixed(t.get_perm(),s.get_perm(),reqrank_max);
		A_SVD_fixed.build(A);
		std::vector<double> SVD_fixed_errors;
		std::vector<double> SVD_errors_check(reqrank_max,0);

		for (int k = 0 ; k < reqrank_max ; k++){
			SVD_fixed_errors.push_back(Frobenius_absolute_error(A_SVD_fixed,A,k));
			for (int l=k ; l<min(nr,nc) ; l++){
				SVD_errors_check[k]+=pow(A_SVD_fixed.get_singular_value(l),2);
			}
			SVD_errors_check[k]=sqrt(SVD_errors_check[k]);
		}

    	// Testing with Eckart–Young–Mirsky theorem for Frobenius norm
		cout << "Testing with Eckart–Young–Mirsky theorem" << endl;
		test = test || !(norm2(SVD_fixed_errors-SVD_errors_check)<1e-10);
		cout << "> Errors with Frobenius norm: "<<SVD_fixed_errors<<endl;
		cout << "> Errors computed with the remaining eigenvalues : "<<SVD_errors_check << endl;

		// ACA automatic building
		SVD<double,GeometricClustering> A_SVD(t.get_perm(),s.get_perm());
		A_SVD.build(A);

		std::pair<double,double> fixed_compression_interval(0.87,0.89);
		std::pair<double,double> auto_compression_interval(0.95,0.97);
		test = test_lrmat(A,A_SVD_fixed,A_SVD,t.get_perm(),s.get_perm(),fixed_compression_interval,auto_compression_interval);
	}
	cout << "test : "<<test<<endl;

	// Finalize the MPI environment.
	MPI_Finalize();
	return test;
}
