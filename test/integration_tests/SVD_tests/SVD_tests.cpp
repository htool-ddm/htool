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
	distance[0] = 10; distance[1] = 20; distance[2] = 30; distance[3] = 40;
	SetNdofPerElt(1);

	for(int idist=0; idist<ndistance; idist++)
	{
		cout << "Distance between the clusters: " << distance[idist] << endl;

		srand (1);
		// we set a constant seed for rand because we want always the same result if we run the check many times
		// (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)

		int nr = 500;
		int nc = 100;
		vector<int> Ir(nr); // row indices for the lrmatrix
		vector<int> Ic(nc); // column indices for the lrmatrix

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
		vector<R3> p2(nc);
		vector<double> r2(nc);
		vector<int> tab2(nc);
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
    int reqrank_max = 10;
		SVD<double> A_SVD(Ir,Ic,t,s,reqrank_max);
		A_SVD.build(A);
		std::vector<double> SVD_errors;
		std::vector<double> SVD_errors_check(reqrank_max,0);

		for (int k = 0 ; k < reqrank_max ; k++){
			SVD_errors.push_back(Frobenius_absolute_error(A_SVD,A,k));
			for (int l=k ; l<min(nr,nc) ; l++){
				SVD_errors_check[k]+=pow(A_SVD.get_singular_value(l),2);
			}
			SVD_errors_check[k]=sqrt(SVD_errors_check[k]);
		}

    // Testing with Eckart–Young–Mirsky theorem for Frobenius norm
		cout << SVD_errors<<endl;
		cout << SVD_errors_check << endl;

	}

}
