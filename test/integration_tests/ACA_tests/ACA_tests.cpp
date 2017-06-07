#include <iostream>
#include <complex>
#include <vector>

#include <htool/cluster.hpp>
#include <htool/lrmat.hpp>
#include <htool/SVD.hpp>


using namespace std;
using namespace htool;


class MyMatrix: public IMatrix<double>{
	const vector<R3>& p1;
	const vector<R3>& p2;

public:
	MyMatrix(const vector<R3>& p10,const vector<R3>& p20 ):p1(p10),p2(p20) {}
	 double get_coef(const int& i, const int& j)const {return 1./(4*M_PI*norm(p1[i]-p2[j]));}
	 std::vector<double> operator*(std::vector<double> a){
		std::vector<double> result(a.size(),0);
		for (int i=0;i<p1.size();i++){
			for (int k=0;k<p2.size();k++){
				result[i]+=this->get_coef(i,k)*a[k];
			}
		}
		return result;
	 }
};


int main(){
	const int ndistance = 4;
	double distance[ndistance];
	distance[0] = 2; distance[1] = 5; distance[2] = 10; distance[3] = 20;
	SetNdofPerElt(1);

	for(int idist=0; idist<ndistance; idist++)
	{
		cout << "Distance between the clusters: " << distance[idist] << endl;

		srand (1);
		// we set a constant seed for rand because we want always the same result if we run the check many times
		// (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)

		// Build matrix A with property for ACA
		int nr = 100;
		int nc = 100;
		vector<int> Ir(nr); // row indices for the lrmatrix
		vector<int> Ic(nr); // column indices for the lrmatrix
		// p1: points in a unit disk of the plane z=z1
		double z1 = 1;
		vector<R3>     p1(nr);
		vector<double> r1(nr);
		vector<int>  tab1(nr);
		for(int j=0; j<nr; j++){
			Ir[j] = j;
			Ic[j] = j;
			double rho = ((double) rand() / (double)(RAND_MAX)); // (double) otherwise integer division!
			double theta = ((double) rand() / (double)(RAND_MAX));
			p1[j][0] = sqrt(rho)*cos(2*M_PI*theta); p1[j][1] = sqrt(rho)*sin(2*M_PI*theta); p1[j][2] = z1;
			// sqrt(rho) otherwise the points would be concentrated in the center of the disk
			r1[j]=0.;
			tab1[j]=j;
		}
		// p2: points in a unit disk of the plane z=z2
		double z2 = 1+distance[idist];
		vectR3 p2(nc);
		vectReal r2(nc);
		vectInt tab2(nc);
		for(int j=0; j<nc; j++){
			double rho = ((double) rand() / (RAND_MAX)); // (double) otherwise integer division!
			double theta = ((double) rand() / (RAND_MAX));
			p2[j][0] = sqrt(rho)*cos(2*M_PI*theta); p2[j][1] = sqrt(rho)*sin(2*M_PI*theta); p2[j][2] = z2;
			r2[j]=0.;
			tab2[j]=j;
		}

		Cluster t(p1,r1,tab1); Cluster s(p2,r2,tab2);
		MyMatrix A(p1,p2);

		// SVD
		SVD<double> A_SVD(Ir,Ic,t,s,10);
		A_SVD.build(A);
		std::vector<double> SVD_errors;


		// ACA
		ACA<double> A_ACA(Ir,Ic,t,s,10);
		std::vector<double> ACA_errors;

		// Comparaison
		for (int k = 0 ; k < 10 ; k++){
			SVD_errors.push_back(Frobenius_relative_error(A_SVD,A,k));
			ACA_errors.push_back(Frobenius_relative_error(A_ACA,A,k));
		}


		//// ACA
		// Cluster t(p1,r1,tab1); Cluster s(p2,r2,tab2);
		// ACA<double> CA(Ir,Ic,t,s);
		// CA.build(A,10);


		// LowRankMatrix B(Abis,Ir,Ic,t,s); // construct a low rank matrix B applying ACA to matrix A
    //
		// // Vecteur
		// vectCplx u(nr);
		// int NbSpl = 1000;
		// double du = 5./double(NbSpl);
		// for(int j=0; j<nr; j++){
		// 	int n = rand()%(NbSpl+1);
		// 	u[j] = n*du;}
    //
		// vectCplx ua(nr),ub(nr);
		// MvProd(ua,A,u);
		// MvProd(ub,B,u);
		// Real err = norm(ua-ub)/norm(ua);
		// cout << "Erreur: " << err << endl;
    //
		// cout << "Taux de compression: ";
		// cout << CompressionRate(B) << endl;
	}

}
