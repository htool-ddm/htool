#include <iostream>
#include <complex>
#include <vector>
#include <random>


#include <htool/clustering/ncluster.hpp>
#include <htool/types/multihmatrix.hpp>
#include <htool/lrmat/SVD.hpp>
#include <htool/lrmat/fullACA.hpp>
#include <htool/lrmat/partialACA.hpp>
#include <htool/lrmat/sympartialACA.hpp>
#include <htool/multilrmat/multipartialACA.hpp>



using namespace std;
using namespace htool;


class MyMultiMatrix: public MultiIMatrix<double>{
	const vector<R3>& p1;
	const vector<R3>& p2;

public:
	MyMultiMatrix(const vector<R3>& p10,const vector<R3>& p20 ):MultiIMatrix(p10.size(),p20.size(),5),p1(p10),p2(p20) {}
	std::vector<double> get_coefs(const int& i, const int& j)const {
		return std::vector<double> {
			1./(4*M_PI*norm2(p1[i]-p2[j])),
			2./(4*M_PI*norm2(p1[i]-p2[j])),
			3./(4*M_PI*norm2(p1[i]-p2[j])),
			4./(4*M_PI*norm2(p1[i]-p2[j])),
			5./(4*M_PI*norm2(p1[i]-p2[j])),
		};
	}
	std::vector<double> mult(std::vector<double>& a, int l) const{
		std::vector<double> result(p1.size(),0);
		for (int i=0;i<p1.size();i++){
			for (int k=0;k<p2.size();k++){
				result[i]+=this->get_coefs(i,k)[l]*a[k];
			}
		}
		return result;
	 }

	 double normFrob(int l) const{
		double norm=0;
		for (int j=0;j<this->nb_rows();j++){
			for (int k=0;k<this->nb_cols();k++){
				norm = norm + std::pow(std::abs((this->get_coefs(j,k))[l]),2);
			}
		}
    	return sqrt(norm);
	}
};

class MyMatrix: public IMatrix<double>{
	const vector<R3>& p1;
	const vector<R3>& p2;

public:
	MyMatrix(const vector<R3>& p10,const vector<R3>& p20 ):IMatrix<double>(p10.size(),p20.size()),p1(p10),p2(p20) {}

	double get_coef(const int& i, const int& j)const {
		return 1./(4*M_PI*norm2(p1[i]-p2[j]));
	}
	
	std::vector<double> mult(std::vector<double>& a) const{
		std::vector<double> result(p1.size(),0);
		for (int i=0;i<p1.size();i++){
			for (int k=0;k<p2.size();k++){
				result[i]+=this->get_coef(i,k)*a[k];
			}
		}
		return result;
	 }
};

void create_geometry(int distance, std::vector<R3>& xt, std::vector<int>& tabt, std::vector<R3>& xs, std::vector<int>& tabs){
	cout << "Distance between the clusters: " << NbrToStr(distance) << endl;
	
	srand (1);
	// we set a constant seed for rand because we want always the same result if we run the check many times
	// (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)

	int nr = xt.size();
	int nc = xs.size();
	vector<int> Ir(nr); // row indices for the lrmatrix
	vector<int> Ic(nc); // column indices for the lrmatrix

	double z1 = 1;
	for(int j=0; j<nr; j++){
		Ir[j] = j;
		double rho = ((double) rand() / (double)(RAND_MAX)); // (double) otherwise integer division!
		double theta = ((double) rand() / (double)(RAND_MAX));
		xt[j][0] = sqrt(rho)*cos(2*M_PI*theta); xt[j][1] = sqrt(rho)*sin(2*M_PI*theta); xt[j][2] = z1;
		// sqrt(rho) otherwise the points would be concentrated in the center of the disk
		tabt[j]=j;
	}
	// p2: points in a unit disk of the plane z=z2
	double z2 = 1+distance;
	vector<int> tab2(nc);
	for(int j=0; j<nc; j++){
		Ic[j] = j;
		double rho = ((double) rand() / (RAND_MAX)); // (double) otherwise integer division!
		double theta = ((double) rand() / (RAND_MAX));
		xs[j][0] = sqrt(rho)*cos(2*M_PI*theta); xs[j][1] = sqrt(rho)*sin(2*M_PI*theta); xs[j][2] = z2;
		tabs[j]=j;
	}
}

template<template<typename,typename> class MultiLowRankMatrix>
int test_multi_hmat_cluster(const MyMultiMatrix& MultiA, const MultiHMatrix<double,MultiLowRankMatrix,GeometricClustering>& MultiHA,int l) {
	bool test =0;
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	std::vector<double> f(MultiA.nb_cols(),1);
	if (rank==0){
		double lower_bound = 0;
		double upper_bound = 10000;
		std::random_device rd;
		std::mt19937 mersenne_engine(rd());
		std::uniform_real_distribution<double> dist(lower_bound,upper_bound);
		auto gen = [&dist, &mersenne_engine](){
					return dist(mersenne_engine);
				};

		generate(begin(f), end(f), gen);
	}
	MPI_Bcast(f.data(),MultiA.nb_cols(),MPI_DOUBLE,0,MPI_COMM_WORLD);

	std::vector<double> result(MultiA.nb_rows(),0);
	MultiHA[l].print_infos();
	result = MultiHA[l]*f;
	double erreur2 = norm2(MultiA.mult(f,l)-result)/norm2(MultiA.mult(f,l));
	double erreurFrob = Frobenius_absolute_error(MultiHA,MultiA,l)/MultiA.normFrob(l);

	test = test || !(erreurFrob<GetEpsilon());
	test = test || !(erreur2<GetEpsilon());

	if (rank==0){
		cout << "Errors with Frobenius norm: "<<erreurFrob<<endl;
		cout << "Errors on a mat vec prod : "<< erreur2<<endl;
		cout << "test: "<<test<<endl;
	}

	return test;
}
