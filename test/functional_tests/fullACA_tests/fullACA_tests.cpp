#include <iostream>
#include <complex>
#include <vector>

#include <htool/cluster.hpp>
#include <htool/lrmat.hpp>
#include <htool/fullACA.hpp>


using namespace std;
using namespace htool;


class MyMatrix: public IMatrix<double>{
	const vector<R3>& p1;
	const vector<R3>& p2;

public:
	MyMatrix(const vector<R3>& p10,const vector<R3>& p20 ):IMatrix<double>(p10.size(),p20.size()),p1(p10),p2(p20) {}
	 double get_coef(const int& i, const int& j)const {return 1./(4*M_PI*norm(p1[i]-p2[j]));}
	 std::vector<double> operator*(std::vector<double> a){
		std::vector<double> result(p1.size(),0);
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
	bool test = 0;

	for(int idist=0; idist<ndistance; idist++)
	{
		cout << "Distance between the clusters: " << distance[idist] << endl;
		SetEpsilon(0.0001);
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
            Ic[j] = j;
			double rho = ((double) rand() / (RAND_MAX)); // (double) otherwise integer division!
			double theta = ((double) rand() / (RAND_MAX));
			p2[j][0] = sqrt(rho)*cos(2*M_PI*theta); p2[j][1] = sqrt(rho)*sin(2*M_PI*theta); p2[j][2] = z2;
			r2[j]=0.;
			tab2[j]=j;
		}

		MyMatrix A(p1,p2);

		// ACA
		int reqrank_max = 10;
		fullACA<double> A_fullACA(Ir,Ic,reqrank_max);
		A_fullACA.build(A);

		std::vector<double> fullACA_errors;
		for (int k = 0 ; k < A_fullACA.rank_of()+1 ; k++){
			fullACA_errors.push_back(Frobenius_absolute_error(A_fullACA,A,k));
		}


		// Test errors
		for (int i =1 ;i<fullACA_errors.size();i++){
			test = test || !(fullACA_errors[i-1]>fullACA_errors[i]);
		}
		cout << "Errors with Frobenius norm: "<<fullACA_errors<<endl;

		// Test compression
		test = test || !(0.89<A_fullACA.compression() && A_fullACA.compression()<0.9);
		cout << "Compression rate : "<<A_fullACA.compression()<<endl;

		// Test error on mat vec prod
		std::vector<double> f(nc,1);
		double error=norm2(A*f-A_fullACA*f);
		test = test ||!(error<1e-7);
		cout << "Errors on a mat vec prod : "<< error<<endl;


	}

	return test ;
}
