#include <htool/types/hmatrix.hpp>
#include <htool/lrmat/fullACA.hpp>
#include <htool/clustering/ncluster.hpp>

using namespace std;
using namespace htool;

double sign(int x){
	if (x > 0) return 1;
	if (x < 0) return -1;
	return 0;
}

class MyMatrix: public IMatrix<complex<double>>{
	const vector<R3>& p1;

public:
	MyMatrix(const vector<R3>& p10 ):IMatrix(p10.size(),p10.size()),p1(p10) {}

	complex<double> get_coef(const int& i, const int& j)const {
		return (1. + sign(i-j)*std::complex<double>(0,1))/(4*M_PI*(1+norm2(p1[i]-p1[j])));
		
	}


  std::vector<complex<double>> operator*(std::vector<complex<double>> a){
		std::vector<complex<double>> result(p1.size(),0);
		for (int i=0;i<p1.size();i++){
			for (int k=0;k<p1.size();k++){
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
  distance[0] = 3; distance[1] = 5; distance[2] = 7; distance[3] = 10;
  SetNdofPerElt(1);
  SetEpsilon(1e-3);
  SetEta(10);

  	for(int idist=0; idist<ndistance; idist++)
  	{

  		srand (1);
  		// we set a constant seed for rand because we want always the same result if we run the check many times
  		// (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)

  		int nr = 2000;
  		int nc = 2000;
  		vector<int> Ir(nr); // row indices for the lrmatrix
  		vector<int> Ic(nc); // column indices for the lrmatrix

  		double z1 = 1;
  		vector<R3>        p1(nr);
			vector<double>  r1(nr,0);
  		vector<int>     tab1(nr);
  		for(int j=0; j<nr; j++){
  			Ir[j] = j;
  			double rho = ((double) rand() / (double)(RAND_MAX)); // (double) otherwise integer division!
  			double theta = ((double) rand() / (double)(RAND_MAX));
  			p1[j][0] = sqrt(rho)*cos(2*M_PI*theta); p1[j][1] = sqrt(rho)*sin(2*M_PI*theta); p1[j][2] = z1;
  			// sqrt(rho) otherwise the points would be concentrated in the center of the disk
  			tab1[j]=j;
  		}

  		MyMatrix A(p1);


  		HMatrix<complex<double>,fullACA,RegularClustering,RjasanowSteinbach> HA_L(A,p1,r1,tab1,'H','L');
  		HA_L.print_infos();

  		HMatrix<complex<double>,fullACA,RegularClustering,RjasanowSteinbach> HA_U(A,p1,r1,tab1,'H','U');
  		HA_U.print_infos();

      	// Global vectors
  		std::vector<complex<double>> x_global(nc,std::complex<double>(2,2)),f_global(nr),f_global_L(nr),f_global_U(nr);
		f_global = A*x_global;

      	// Global product
      	HA_L.mvprod_global(x_global.data(),f_global_L.data());
		HA_U.mvprod_global(x_global.data(),f_global_U.data());

		// Errors
		double global_diff_L = norm2(f_global-f_global_L)/norm2(f_global);
		double global_diff_U = norm2(f_global-f_global_U)/norm2(f_global);
		double global_diff_L_U = norm2(f_global_L-f_global_U)/norm2(f_global);


  		if (rank==0){
  			cout <<"difference on mat vec prod computed globally for lower hermitian matrix: "<<global_diff_L << endl;
  		}
		test = test || !(global_diff_L<GetEpsilon());

		if (rank==0){
  			cout <<"difference on mat vec prod computed globally for upper hermitian matrix: "<<global_diff_U << endl;
  		}
		test = test || !(global_diff_U<GetEpsilon());

		if (rank==0){
  			cout <<"difference on mat vec prod computed globally between lower hermitian matrix and lower hermitian matrix: "<<global_diff_L_U << endl;
  		}
		test = test || !(global_diff_L_U<GetEpsilon());
  	}
		if (rank==0){
			cout <<"test: "<<test << endl;

		}
  	// Finalize the MPI environment.
  	MPI_Finalize();

  	return test;
  }
