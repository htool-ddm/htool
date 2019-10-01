#include <iostream>
#include <complex>
#include <vector>

#include <htool/lrmat/sympartialACA.hpp>
#include "test_lrmat.hpp"


using namespace std;
using namespace htool;


int main(){
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

		std::vector<int> permt,perms;
		Cluster t(xt,permt); Cluster s(xs,perms); // We avoid 
		MyMatrix A(xt,xs);

		// sympartialACA fixed rank
    	int reqrank_max = 10;
		sympartialACA<double> A_sympartialACA_fixed(permt,perms,reqrank_max);
		A_sympartialACA_fixed.build(A,t,xt,tabt,s,xs,tabs);;

		// ACA automatic building
		sympartialACA<double> A_sympartialACA(permt,perms);
		A_sympartialACA.build(A,t,xt,tabt,s,xs,tabs);

		std::pair<double,double> fixed_compression_interval(0.87,0.89);
		std::pair<double,double> auto_compression_interval(0.93,0.96);
		test = test || (test_lrmat(A,A_sympartialACA_fixed,A_sympartialACA,permt,perms,fixed_compression_interval,auto_compression_interval));
	}
	cout << "test : "<<test<<endl;
	return test;
}
