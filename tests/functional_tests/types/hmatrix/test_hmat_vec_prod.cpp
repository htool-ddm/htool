#include <htool/types/hmatrix.hpp>
#include <htool/lrmat/fullACA.hpp>
#include <htool/clustering/geometric_splitting.hpp>
#include <htool/clustering/regular_splitting.hpp>

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
  distance[0] = 3; distance[1] = 5; distance[2] = 7; distance[3] = 10;
  SetNdofPerElt(1);
  SetEpsilon(1e-6);
  SetEta(-0.1);

  	for(int idist=0; idist<ndistance; idist++)
  	{

  		srand (1);
  		// we set a constant seed for rand because we want always the same result if we run the check many times
  		// (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)

  		int nr = 2000;
  		int nc = 1000;
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
  		// p2: points in a unit disk of the plane z=z2
  		double z2 = 1+distance[idist];
  		vector<R3>       p2(nc);
			vector<double> r2(nc,0);
  		vector<int>    tab2(nc);
  		for(int j=0; j<nc; j++){
              Ic[j] = j;
  			double rho = ((double) rand() / (RAND_MAX)); // (double) otherwise integer division!
  			double theta = ((double) rand() / (RAND_MAX));
  			p2[j][0] = sqrt(rho)*cos(2*M_PI*theta); p2[j][1] = sqrt(rho)*sin(2*M_PI*theta); p2[j][2] = z2;
  			tab2[j]=j;
  		}

  		MyMatrix A(p1,p2);
		//   vector<double> g2(nc,1);
		//   vector<double> g1(nr,1);
		// std::shared_ptr<RegularClustering> t=make_shared<RegularClustering>();
		// std::shared_ptr<RegularClustering> s=make_shared<RegularClustering>();
		// t->build(p1,r1,tab1,g1,2);
		// s->build(p2,r2,tab2,g2,2);
		// HMatrix<double,fullACA,RegularClustering> HA(A,t,p1,tab1,s,p2,tab2);


  		HMatrix<double,fullACA,RegularClustering> HA(A,p1,r1,tab1,p2,r2,tab2);
  		HA.print_infos();



      // Global vectors
  		std::vector<double> x_global(nc,1),f_global(nr),f_global_test(nr);
			f_global = A*x_global;

      // Global product
      HA.mvprod_global(x_global.data(),f_global_test.data());

      // Errors
      double global_diff = norm2(f_global-f_global_test)/norm2(f_global);


  		if (rank==0){
  			cout <<"difference on mat vec prod computed globally: "<<global_diff << endl;
  		}
			test = test || !(global_diff<GetEpsilon());
  	}
		if (rank==0){
			cout <<"test: "<<test << endl;

		}
  	// Finalize the MPI environment.
  	MPI_Finalize();

  	return test;
  }
