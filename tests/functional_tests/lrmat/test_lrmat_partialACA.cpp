#include <iostream>
#include <complex>
#include <vector>

#include <htool/lrmat/partialACA.hpp>


using namespace std;
using namespace htool;


class MyMatrix: public IMatrix<double>{
	const vector<R3>& p1;
	const vector<R3>& p2;

public:
	MyMatrix(const vector<R3>& p10,const vector<R3>& p20 ):IMatrix<double>(p10.size(),p20.size()),p1(p10),p2(p20) {}
	 double get_coef(const int& i, const int& j)const {return 1./(4*M_PI*norm2(p1[i]-p2[j]));}
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
	distance[0] = 15; distance[1] = 20; distance[2] = 30; distance[3] = 40;
	SetNdofPerElt(1);
	bool test = 0;
	for(int idist=0; idist<ndistance; idist++)
	{
		cout << "Distance between the clusters: " << NbrToStr(distance[idist]) << endl;
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
		vector<int>  tab1(nr);
		for(int j=0; j<nr; j++){
			Ir[j] = j;
			double rho = ((double) rand() / (double)(RAND_MAX)); // (double) otherwise integer division!
			double theta = ((double) rand() / (double)(RAND_MAX));
			p1[j][0] = sqrt(rho)*cos(2*M_PI*theta); p1[j][1] = sqrt(rho)*sin(2*M_PI*theta); p1[j][2] = z1;
			// sqrt(rho) otherwise the points would be concentrated in the center of the disk
			tab1[j]=j;
		}
		// p2: points in a unit disk of the plane z=z2
		double z2 = 1+distance[idist];
		vector<R3> p2(nc);
		vector<int> tab2(nc);
		for(int j=0; j<nc; j++){
            Ic[j] = j;
			double rho = ((double) rand() / (RAND_MAX)); // (double) otherwise integer division!
			double theta = ((double) rand() / (RAND_MAX));
			p2[j][0] = sqrt(rho)*cos(2*M_PI*theta); p2[j][1] = sqrt(rho)*sin(2*M_PI*theta); p2[j][2] = z2;
			tab2[j]=j;
		}

        // Clustering
        std::vector<int> permt,perms;
		Cluster t(p1,permt); Cluster s(p2,perms); // We avoid cluster_tree and MPI here
		MyMatrix A(p1,p2);

		// ACA with fixed rank
		int reqrank_max = 10;
		partialACA<double> A_partialACA_fixed(permt,perms,reqrank_max);
		A_partialACA_fixed.build(A,t,p1,tab1,s,p2,tab2);
		std::vector<double> partialACA_fixed_errors;
		for (int k = 0 ; k < A_partialACA_fixed.rank_of()+1 ; k++){
			partialACA_fixed_errors.push_back(Frobenius_absolute_error(A_partialACA_fixed,A,k));
		}

		cout << "Partial ACA with fixed rank" << endl;
		// Test rank
		cout << "rank : "<<A_partialACA_fixed.rank_of() << endl;
		test = test || !(A_partialACA_fixed.rank_of()==reqrank_max);

		// Test Frobenius errors
		test = test || !(partialACA_fixed_errors.back()<1e-8);
		cout << "Errors with Frobenius norm : "<<partialACA_fixed_errors<<endl;

		// Test compression
		test = test || !(0.87<A_partialACA_fixed.compression() && A_partialACA_fixed.compression()<0.89);
		cout << "Compression rate : "<<A_partialACA_fixed.compression()<<endl;

		// Test mat vec prod
		std::vector<double> f(nc,1),out_perm(nr);
        std::vector<double> out=A_partialACA_fixed*f;
        for (int i = 0; i<permt.size();i++){
            out_perm[permt[i]]=out[i];
        }
		double error=norm2(A*f-out_perm);
		test = test || !(error<1e-7);
		cout << "Errors on a mat vec prod : "<< error<<endl<<endl;

		// ACA automatic building
		partialACA<double> A_partialACA(permt,perms);
		A_partialACA.build(A,t,p1,tab1,s,p2,tab2);
		std::vector<double> partialACA_errors;
		for (int k = 0 ; k < A_partialACA.rank_of()+1 ; k++){
			partialACA_errors.push_back(Frobenius_absolute_error(A_partialACA,A,k));
		}

		cout << "Partial ACA" << endl;
		// Test Frobenius error
		test = test || !(partialACA_errors[A_partialACA.rank_of()]<GetEpsilon());
		cout << "Errors with Frobenius norm: "<<partialACA_errors<<endl;

		// Test compression rate
		test = test || !(0.93<A_partialACA.compression() && A_partialACA.compression()<0.96);
		cout << "Compression rate : "<<A_partialACA.compression()<<endl;

		// Test mat vec prod
        out=A_partialACA*f;
        for (int i = 0; i<permt.size();i++){
            out_perm[permt[i]]=out[i];
        }
		error = norm2(A*f-out_perm);
		test = test || !(error<GetEpsilon());
		cout << "Errors on a mat vec prod : "<< error<<endl<<endl<<endl;
	}
	cout << "test : "<<test<<endl;
	return test;
}
