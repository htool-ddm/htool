#include "test_multi_hmat.hpp"

using namespace std;
using namespace htool;

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
	const int ndistance = 4;
	double distance[ndistance];
	distance[0] = 15; distance[1] = 20; distance[2] = 30; distance[3] = 40;
	SetNdofPerElt(1);
	SetEpsilon(1e-8);
	SetEta(0.1);
	int nr=500;
	int nc=400;
	std::vector<R3> xt(nr);
	std::vector<R3> xs(nc);
	std::vector<int> tabt(nr);
	std::vector<int> tabs(nc);

	bool test = 0;

	//
	for(int idist=0; idist<ndistance; idist++){
		
		create_geometry(distance[idist],xt,tabt,xs,tabs);


		vector<double> rhs(xs.size(),1);
		MyMultiMatrix MultiA(xt,xs);
		int nm = MultiA.nb_matrix();
		MyMatrix A(xt,xs);
		MultiHMatrix<double,MultipartialACA,GeometricClustering> MultiHA(MultiA,xt,xs);
		HMatrix<double,partialACA,GeometricClustering> HA(A,xt,xs);

		// Comparison with HMatrix
		std::vector<double> one(nc,1);
		double error=norm2(MultiHA[0]*one-HA*one);
		test = test || !(error<1e-10);
		cout << "> Errors compared to HMatrix: "<<error<<endl;
		for (int l=0;l<nm;l++){
			test = test || (test_multi_hmat_cluster(MultiA,MultiHA,l));
		}
	}

	// Finalize the MPI environment.
	MPI_Finalize();

	return test;

}
