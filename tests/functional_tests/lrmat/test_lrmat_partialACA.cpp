#include <iostream>
#include <complex>
#include <vector>


#include <htool/clustering/ncluster.hpp>
#include <htool/lrmat/partialACA.hpp>
#include "test_lrmat.hpp"


using namespace std;
using namespace htool;


int main(int argc, char *argv[]){
	// Initialize the MPI environment
	MPI_Init(&argc,&argv);

	bool verbose=1;
	if (argc>=2){
		verbose=argv[1];
	}

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
		
		create_geometry(distance[idist],xt,tabt,xs,tabs,verbose);

		GeometricClustering t,s; 

		std::vector<int> tabt(xt.size()),tabs(xs.size());
		std::iota(tabt.begin(),tabt.end(),int(0));
		std::iota(tabs.begin(),tabs.end(),int(0));
		t.build(xt,std::vector<double>(xt.size(),0),tabt,std::vector<double>(xt.size(),1));
		s.build(xs,std::vector<double>(xs.size(),0),tabs,std::vector<double>(xs.size(),1));

		MyMatrix A(xt,xs);

		// partialACA fixed rank
    	int reqrank_max = 10;
		partialACA<double,GeometricClustering> A_partialACA_fixed(t.get_perm(),s.get_perm(),reqrank_max);
		A_partialACA_fixed.build(A,t,xt,tabt,s,xs,tabs);;

		
		// ACA automatic building
		partialACA<double,GeometricClustering> A_partialACA(t.get_perm(),s.get_perm());
		A_partialACA.build(A,t,xt,tabt,s,xs,tabs);
		
		std::pair<double,double> fixed_compression_interval(0.87,0.89);
		std::pair<double,double> auto_compression_interval(0.93,0.96);
		test = test || (test_lrmat(A,A_partialACA_fixed,A_partialACA,t.get_perm(),s.get_perm(),fixed_compression_interval,auto_compression_interval,verbose,3));
	}
	
	cout << "test : "<<test<<endl;

	// Finalize the MPI environment.
	MPI_Finalize();
	
	return test;
}
