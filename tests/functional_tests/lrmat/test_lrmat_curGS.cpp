#include <iostream>
#include <complex>
#include <vector>

#include <htool/lrmat/curGS.hpp>


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
		cout << "Distance between the clusters: " << distance[idist] << endl;
		SetEpsilon(0.00001);
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

		// CUR with fixed rank
		int reqrank_max = 10;
		curGS<double> CUR_fixed(permt,perms,reqrank_max);
		CUR_fixed.build(A,t,p1,tab1,s,p2,tab2);
		std::vector<double> cur_fixed_errors;
		for (int k = 0 ; k < CUR_fixed.rank_of()+1 ; k++){
			cur_fixed_errors.push_back(Frobenius_absolute_error(CUR_fixed,A,k));
		}

		cout << "CUR_GCS with fixed rank" << endl;
		// Test rank
		cout << "rank : "<<CUR_fixed.rank_of() << endl;
		test = test || !(CUR_fixed.rank_of()==reqrank_max);

		// Test Frobenius errors
		test = test || !(cur_fixed_errors.back()<1e-8);
		cout << "Errors with Frobenius norm : "<<cur_fixed_errors<<endl;

		// Test compression
		test = test || !(0.87<CUR_fixed.compression() && CUR_fixed.compression()<0.89);
		cout << "Compression rate : "<<CUR_fixed.compression()<<endl;

		// Test mat vec prod
        std::vector<double> f(nc,1),out_perm(nr);
        std::vector<double> out=CUR_fixed*f;
        for (int i = 0; i<permt.size();i++){
            out_perm[permt[i]]=out[i];
        }
		double error=norm2(A*f-out_perm);
		test = test || !(error<1e-7);
		cout << "Errors on a mat vec prod : "<< error<<endl<<endl;

		// CUR automatic building
		curGS<double> CUR(permt,perms);
		CUR.build(A,t,p1,tab1,s,p2,tab2);
		std::vector<double> cur_errors;
		for (int k = 0 ; k < CUR.rank_of()+1 ; k++){
			cur_errors.push_back(Frobenius_absolute_error(CUR,A,k));
		}

		cout << "CUR_GS" << endl;
		// Test Frobenius error
		test = test || !(cur_errors[CUR.rank_of()]<GetEpsilon());
		cout << "Errors with Frobenius norm: "<<cur_errors<<endl;

		// Test compression rate
		test = test || !(0.93<CUR.compression() && CUR.compression()<0.96);
		cout << "Compression rate : "<<CUR.compression()<<endl;

        // Test mat vec prod
        out=CUR*f;
        for (int i = 0; i<permt.size();i++){
            out_perm[permt[i]]=out[i];
        }
		error = norm2(A*f-out_perm);
		test = test || !(error<GetEpsilon()*10);
		cout << "Errors on a mat vec prod : "<< error<<endl<<endl<<endl;


	}
	cout << "test : "<<test<<endl;
	return test;
}
