#include <iostream>
#include <complex>
#include <vector>

#include <htool/clustering/cluster.hpp>
#include <htool/types/hmatrix.hpp>
#include <htool/lrmat/fullACA.hpp>



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

		MyMatrix A(p1,p2);
		std::shared_ptr<Cluster_tree> t=make_shared<Cluster_tree>(p1,r1,tab1,g1);
		std::shared_ptr<Cluster_tree> s=make_shared<Cluster_tree>(p2,r2,tab2,g2);
		HMatrix<fullACA,double> HA(A,t,p1,tab1,s,p2,tab2);
		HA.print_infos();

        HA.save_plot("plot_"+NbrToStr(idist));

	}

	// Finalize the MPI environment.
	MPI_Finalize();
	return test;
}