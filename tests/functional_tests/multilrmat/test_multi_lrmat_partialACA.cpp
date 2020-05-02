#include <iostream>
#include <complex>
#include <vector>

#include <htool/multilrmat/multipartialACA.hpp>
#include <htool/lrmat/partialACA.hpp>
#include "test_multi_lrmat.hpp"


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
	std::vector<int> tabt(nr);
	std::vector<int> tabs(nc);
	bool test =0;

	double test_time;

	for(int idist=0; idist<ndistance; idist++)
	{
		
		create_geometry(distance[idist],xt,tabt,xs,tabs);

		GeometricClustering t,s; 

		std::vector<int> tabt(xt.size()),tabs(xs.size());
		std::iota(tabt.begin(),tabt.end(),int(0));
		std::iota(tabs.begin(),tabs.end(),int(0));
		t.build(xt,std::vector<double>(xt.size(),0),tabt,std::vector<double>(xt.size(),1));
		s.build(xs,std::vector<double>(xs.size(),0),tabs,std::vector<double>(xs.size(),1));

		MyMultiMatrix A(xt,xs);
		int nm = A.nb_matrix();
		MyMatrix A_test(xt,xs);

		// partialACA fixed rank
    	int reqrank_max = 10;
		MultipartialACA<double,GeometricClustering> A_partialACA_fixed(t.get_perm(),s.get_perm(),nm,reqrank_max);
		partialACA<double,GeometricClustering> A_partialACA_fixed_test(t.get_perm(),s.get_perm(),reqrank_max);
		A_partialACA_fixed.build(A,t,xt,tabt,s,xs,tabs);;
		A_partialACA_fixed_test.build(A_test,t,xt,tabt,s,xs,tabs);;

		// ACA automatic building
		MultipartialACA<double,GeometricClustering> A_partialACA(t.get_perm(),s.get_perm(),nm);
		A_partialACA.build(A,t,xt,tabt,s,xs,tabs);
		partialACA<double,GeometricClustering> A_partialACA_test(t.get_perm(),s.get_perm());
		A_partialACA_test.build(A_test,t,xt,tabt,s,xs,tabs);;

		// Comparison with lrmat
		std::vector<double> one(nc,1);
		test = test || !(norm2(A_partialACA_fixed[0]*one-A_partialACA_fixed_test*one)<1e-10);
		cout << "> Errors for fixed rank compared to lrmat: "<<norm2(A_partialACA_fixed[0]*one-A_partialACA_fixed_test*one)<<endl;

		test = test || !(norm2(A_partialACA[0]*one-A_partialACA_test*one)<1e-10);
		cout << "> Errors for auto rank compared to lrmat: "<<norm2(A_partialACA[0]*one-A_partialACA_test*one)<<endl;

		

		// Test multi lrmat
		std::pair<double,double> fixed_compression_interval(0.87,0.89);
		std::pair<double,double> auto_compression_interval(0.93,0.96);
		
		test = test || (test_multi_lrmat(A,A_partialACA_fixed,A_partialACA,t.get_perm(),s.get_perm(),fixed_compression_interval,auto_compression_interval));
	
	}

	cout << "test : "<<test<<endl;

	// Finalize the MPI environment.
	MPI_Finalize();
	
	return test;
}
