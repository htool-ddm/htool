#include <htool/types/point.hpp>
#include <htool/solvers/ddm.hpp>
#include <htool/lrmat/fullACA.hpp>
#include <htool/types/hmatrix.hpp>
#include <htool/input_output/geometry.hpp>


using namespace std;
using namespace htool;



int main(int argc, char *argv[]){

	// Input file
	if ( argc < 6 ){ // argc should be 5 or more for correct execution
    // We print argv[0] assuming it is the program name
    cout<<"usage: "<< argv[0] <<" <matrixfile> <rhsfile> <meshfile> <solutionfile> <mu>\n";
		return 1;
	}

	// Initialize the MPI environment
	MPI_Init(&argc,&argv);

	// Get the number of processes
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// Get the rank of the process
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//
	bool test =0;
	SetNdofPerElt(1);
	SetEpsilon(1e-3);
	SetEta(0.1);
	SetMinClusterSize(1);
    int mu = StrToNbr<int>(argv[5]);

	// HPDDM verbosity
	HPDDM::Option& opt = *HPDDM::Option::get();
	opt.parse(argc, argv, rank == 0);
    opt["schwarz_method"]=HPDDM_SCHWARZ_METHOD_NONE;
	if(rank != 0)
		opt.remove("verbosity");

	// Matrix
	Matrix<complex<double>> A;
	A.bytes_to_matrix(argv[1]);
	int n = A.nb_rows();

	// Right-hand side
    Matrix<complex<double>> f_global(n,mu);
	std::vector<complex<double>> temp(n);
	bytes_to_vector(temp,argv[2]);
    for (int i=0;i<mu;i++){
        f_global.set_col(i,temp);
    }

	// Mesh
	std::vector<R3> p;
	Load_GMSH_nodes(p,argv[3]);

	// Hmatrix
	std::vector<int> tab(n);
	for (int i=0;i<n;i++){
		tab[i]=i;
	}
	HMatrix<fullACA,complex<double>> HA(A,p);

	// Global vectors
	Matrix<complex<double>> x_global(n,mu),x_ref(n,mu);
	bytes_to_vector(temp,argv[4]);
    for (int i=0;i<mu;i++){
        x_ref.set_col(i,temp);
    }

	// Solve
    DDM<fullACA,complex<double>> hpddm_operator(HA);
    hpddm_operator.facto_one_level();
	hpddm_operator.solve(f_global.data(),x_global.data(),mu);
	HA.print_infos();

	// Error on inversion
	double inv_error2=normFrob(f_global-A*x_global)/normFrob(f_global);
	double error2 = normFrob(x_ref-x_global)/normFrob(x_ref);
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
