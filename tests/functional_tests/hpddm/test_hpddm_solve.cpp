#include <htool/point.hpp>
#include <htool/wrapper_hpddm.hpp>
#include <htool/fullACA.hpp>
#include <htool/hmatrix.hpp>
#include <htool/geometry.hpp>


using namespace std;
using namespace htool;



int main(int argc, char const *argv[]){

	// Input file
	if ( argc < 5 ){ // argc should be 5 or more for correct execution
    // We print argv[0] assuming it is the program name
    cout<<"usage: "<< argv[0] <<" <matrixfile> <rhsfile> <meshfile> <solutionfile>\n";
		return 1;
	}

	// Initialize the MPI environment
	MPI_Init(NULL, NULL);

	// Get the number of processes
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// Get the rank of the process
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//
	bool test =0;
	SetNdofPerElt(1);
	SetEpsilon(1e-6);
	SetEta(0.1);
	SetMinClusterSize(1);

	// HPDDM verbosity
	HPDDM::Option& opt = *HPDDM::Option::get();
	opt.parse(argc, argv, rank == 0);
	if(rank != 0)
		opt.remove("verbosity");

	// Matrix
	Matrix<complex<double>> A;
	A.bytes_to_matrix(argv[1]);
	int n = A.nb_rows();

	// Right-hand side
	std::vector<complex<double>> f_global(n,1);
	bytes_to_vector(f_global,argv[2]);

	// Mesh
	std::vector<R3> p;
	LoadGMSHMesh(p,argv[3]);

	// Hmatrix
	std::vector<int> tab(n);
	for (int i=0;i<n;i++){
		tab[i]=i;
	}
	HMatrix<fullACA,complex<double>> HA(A,p,tab);

	// Global vectors
	std::vector<complex<double>> x_global(n,0),x_ref(n);
	bytes_to_vector(x_ref,argv[4]);

	// Local vectors
	int n_local= HA.get_local_size_cluster();
	const std::vector<std::vector<int>>& MasterClusters = HA.get_MasterClusters();
	std::vector<complex<double>> x_local(n_local,0),f_local(n_local,1);
	for (int i=0;i<n_local;i++){
		f_local[i]=f_global[MasterClusters[rank][i]];
	}

	// Solve
	Identity<Cplx> P(n_local);
  HPDDMOperator<fullACA,complex<double>> A_HPDDM(HA,P);
  complex<double>* const rhs = &(f_local[0]);
  complex<double>* x = &(x_local[0]);
  HPDDM::IterativeMethod::solve(A_HPDDM, rhs, x, 1,HA.get_comm());


	// Local to Global
	std::vector<complex<double>> rcv(n);

	std::vector<int> recvcounts(size);
	std::vector<int>  displs(size);

	displs[0] = 0;

	for (int i=0; i<size; i++) {
		recvcounts[i] = MasterClusters[i].size();
		if (i > 0)
			displs[i] = displs[i-1] + recvcounts[i-1];
	}


	MPI_Allgatherv(&(x_local.front()), recvcounts[rank], MPI_DOUBLE_COMPLEX, &(rcv.front()), &(recvcounts.front()), &(displs.front()), MPI_DOUBLE_COMPLEX, HA.get_comm());

	for (int i=0; i<size; i++)
	for (int j=0; j< MasterClusters[i].size(); j++)
		x_global[MasterClusters[i][j]] = rcv[displs[i]+j];

	// Error on inversion
	double inv_error2=norm2(f_global-A*x_global)/norm2(f_global);
	double error2 = norm2(x_ref-x_global)/norm2(x_ref);
  if (rank==0){
    cout <<"error on inversion : "<<inv_error2 << endl;
		cout <<"error on solution  : "<<error2 << endl;
  }

	test = test || !(inv_error2<opt.val("tol",1e-6));
	test = test || !(error2<1e-5);

	//Finalize the MPI environment.
	MPI_Finalize();

	return test;
}
