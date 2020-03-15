#include <iostream>
#include <complex>
#include <vector>

#include <htool/multilrmat/multipartialACA.hpp>
#include "test_multi_lrmat.hpp"


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
// int nr=5;
// 	int nc=10;
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
		MyMultiMatrix A(xt,xs);
		int nm = A.nb_matrix();
		MyMatrix A_test(xt,xs);

		// partialACA fixed rank
    	int reqrank_max = 10;
		MultipartialACA<double> A_partialACA_fixed(permt,perms,nm,reqrank_max);
		partialACA<double> A_partialACA_fixed_test(permt,perms,reqrank_max);
		A_partialACA_fixed.build(A,t,xt,tabt,s,xs,tabs);;
		A_partialACA_fixed_test.build(A_test,t,xt,tabt,s,xs,tabs);;

		// ACA automatic building
		MultipartialACA<double> A_partialACA(permt,perms,nm);
		A_partialACA.build(A,t,xt,tabt,s,xs,tabs);
		partialACA<double> A_partialACA_test(permt,perms);
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
		for (int l=0;l<nm;l++){
			cout << "Matrix numbered : " << NbrToStr(l) << endl;
			test = test || (test_multi_lrmat(A,A_partialACA_fixed,A_partialACA,permt,perms,fixed_compression_interval,auto_compression_interval,l));
		}
		
	}
	cout << "test : "<<test<<endl;
	return test;
}
