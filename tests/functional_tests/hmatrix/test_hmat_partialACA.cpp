#include <iostream>
#include <complex>
#include <vector>

#include <htool/cluster.hpp>
#include <htool/lrmat.hpp>
#include <htool/hmatrix.hpp>
#include <htool/partialACA.hpp>
#include <htool/matrix.hpp>
#include <mpi.h>



using namespace std;
using namespace htool;


class MyMatrix: public IMatrix<double>{
	const vector<R3>& p1;
	const vector<R3>& p2;

public:
	MyMatrix(const vector<R3>& p10,const vector<R3>& p20 ):IMatrix(p10.size(),p20.size()),p1(p10),p2(p20) {}
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


int main(int argc, char *argv[]) {

	// Initialize the MPI environment
	MPI_Init(&argc,&argv);

	// Get the number of processes
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// Get the rank of the process
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//
	bool test = 0;
	const int ndistance = 4;
	double distance[ndistance];
	distance[0] = 10; distance[1] = 20; distance[2] = 30; distance[3] = 40;
	SetNdofPerElt(1);
	SetEpsilon(1e-8);
	SetEta(0.1);

	for(int idist=0; idist<ndistance; idist++)
	{
		// cout << "Distance between the clusters: " << distance[idist] << endl;

		srand (1);
		// we set a constant seed for rand because we want always the same result if we run the check many times
		// (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)

		int nr = 500;
		int nc = 400;
		vector<int> Ir(nr); // row indices for the lrmatrix
		vector<int> Ic(nc); // column indices for the lrmatrix

		double z1 = 1;
		vector<R3>     p1(nr);
	  vector<double> r1(nr,0);
	  vector<double> g1(nr,1);
		vector<int>    tab1(nr);
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
		vector<R3> 		 p2(nc);
		vector<double> r2(nc,0);
	  vector<double> g2(nc,1);
		vector<int>    tab2(nc);
		for(int j=0; j<nc; j++){
            Ic[j] = j;
			double rho = ((double) rand() / (RAND_MAX)); // (double) otherwise integer division!
			double theta = ((double) rand() / (RAND_MAX));
			p2[j][0] = sqrt(rho)*cos(2*M_PI*theta); p2[j][1] = sqrt(rho)*sin(2*M_PI*theta); p2[j][2] = z2;
			tab2[j]=j;
		}

		vector<double> rhs(p2.size(),1);
		MyMatrix A(p1,p2);
		HMatrix<partialACA,double> HA(A,p1,r1,tab1,g1,p2,r2,tab2,g2);
		HA.print_stats();

		std::vector<double> f(nc,1);
		double erreur2 = norm2(A*f-HA*f);
		double erreurFrob = Frobenius_absolute_error(HA,A);

		test = test || !(erreurFrob<GetEpsilon());
		test = test || !(erreur2<GetEpsilon()*10);

		if (rank==0){
			cout << "Errors with Frobenius norm: "<<erreurFrob<<endl;
			cout << "Errors on a mat vec prod : "<< erreur2<<endl;
		}

	}
	if (rank==0){
		cout << "test :"<<test << endl;
	}
	// Finalize the MPI environment.
	MPI_Finalize();
}
