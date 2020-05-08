#include <htool/types/point.hpp>
#include <htool/solvers/ddm.hpp>
#include <htool/lrmat/fullACA.hpp>
#include <htool/types/hmatrix.hpp>
#include <htool/input_output/geometry.hpp>
#include <htool/clustering/ncluster.hpp>


using namespace std;
using namespace htool;



int test_solver_ddm(int argc, char *argv[], int mu){

	// Input file
	if ( argc < 2 ){ // argc should be 5 or more for correct execution
    // We print argv[0] assuming it is the program name
    cout<<"usage: "<< argv[0] <<" datapath\n";
		return 1;
	}
	string datapath=argv[1];

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


	// HPDDM verbosity
	HPDDM::Option& opt = *HPDDM::Option::get();
	opt.parse(argc, argv, rank == 0);
	double tol = opt.val("tol",1e-6);
	if(rank != 0)
		opt.remove("verbosity");
	opt.parse("-hpddm_max_it 200");

	// HTOOL
	SetNdofPerElt(1);
	SetEpsilon(tol);
	SetEta(0.1);
	SetMinClusterSize(1);

	// Matrix
	Matrix<complex<double>> A;
	A.bytes_to_matrix(datapath+"matrix.bin");
	int n = A.nb_rows();

	// Right-hand side
    Matrix<complex<double>> f_global(n,mu);
	std::vector<complex<double>> temp(n);
	bytes_to_vector(temp,datapath+"rhs.bin");
    for (int i=0;i<mu;i++){
        f_global.set_col(i,temp);
    }

	// Mesh
	std::vector<R3> p;
	Load_GMSH_nodes(p,datapath+"mesh.msh");

   // Clustering
    if (rank==0)
	   std::cout << "Creating cluster tree" << std::endl;
    std::shared_ptr<htool::GeometricClustering> t=std::make_shared<htool::GeometricClustering>();
	(*t).read_cluster(datapath+"cluster_"+NbrToStr(size)+"_permutation.csv",datapath+"cluster_"+NbrToStr(size)+"_tree.csv");

    // std::vector<int>tab(n);
    // std::iota(tab.begin(),tab.end(),int(0));
    // t->build(p,std::vector<double>(n,0),tab,std::vector<double>(n,1),2);
	// std::vector<int> permutation_test(n);
	// bytes_to_vector(permutation_test,datapath+"permutation.bin");
	// t->save(p,"test_cluster",{0,1,2,3});


	// Hmatrix
	if (rank==0)
	   std::cout << "Creating HMatrix" << std::endl;
	HMatrix<complex<double>,fullACA,GeometricClustering> HA(A,t,p);
	HA.print_infos();
	
	// Global vectors
	Matrix<complex<double>> x_global(n,mu),x_ref(n,mu),test_global(n,mu);
	bytes_to_vector(temp,datapath+"sol.bin");
    for (int i=0;i<mu;i++){
        x_ref.set_col(i,temp);
    }

	// Partition
	std::vector<int> cluster_to_ovr_subdomain;
    std::vector<int> ovr_subdomain_to_global;
    std::vector<int> neighbors;
    std::vector<std::vector<int> > intersections;
	bytes_to_vector(cluster_to_ovr_subdomain,datapath+"cluster_to_ovr_subdomain_"+NbrToStr(size)+"_"+NbrToStr(rank)+".bin");
	bytes_to_vector(ovr_subdomain_to_global,datapath+"ovr_subdomain_to_global_"+NbrToStr(size)+"_"+NbrToStr(rank)+".bin");
	bytes_to_vector(neighbors,datapath+"neighbors_"+NbrToStr(size)+"_"+NbrToStr(rank)+".bin");

	intersections.resize(neighbors.size());
	for (int p=0;p<neighbors.size();p++){
		bytes_to_vector(intersections[p],datapath+"intersections_"+NbrToStr(size)+"_"+NbrToStr(rank)+"_"+NbrToStr(p)+".bin");
		std::cout << intersections[p] << std::endl;
	}

	// Errors
	double error2;

	// Solve
    DDM<complex<double>,fullACA,GeometricClustering> ddm_wo_overlap(HA);
	DDM<complex<double>,fullACA,GeometricClustering> ddm_with_overlap(A,HA,ovr_subdomain_to_global,cluster_to_ovr_subdomain,neighbors,intersections);

	// No precond wo overlap
    if (rank==0)
        std::cout<<"No precond without overlap:"<<std::endl;

    opt.parse("-hpddm_schwarz_method none");
    ddm_wo_overlap.solve(f_global.data(),x_global.data(),mu);
	ddm_wo_overlap.print_infos();
	error2=normFrob(f_global-A*x_global)/normFrob(f_global);
    if (rank==0){
		cout <<"error: "<<error2 << endl;
    }

	test = test || !(error2<tol);

    x_global = 0;


    // DDM one level ASM wo overlap
    if (rank==0)
        std::cout<<"ASM one level without overlap:"<<std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    opt.parse("-hpddm_schwarz_method asm ");
    ddm_wo_overlap.facto_one_level();
    ddm_wo_overlap.solve(f_global.data(),x_global.data(),mu);
	ddm_wo_overlap.print_infos();
	error2=normFrob(f_global-A*x_global)/normFrob(f_global);

    if (rank==0){
		cout <<"error: "<<error2 << endl;
    }

	test = test || !(error2<tol);

    x_global = 0;
	
    // DDM one level RAS wo overlap
    if (rank==0)
        std::cout<<"RAS one level without overlap:"<<std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    opt.parse("-hpddm_schwarz_method ras ");
    ddm_wo_overlap.solve(f_global.data(),x_global.data(),mu);
	ddm_wo_overlap.print_infos();
	HA.mvprod_global(x_global.data(),test_global.data(),mu);
	error2=normFrob(f_global-A*x_global)/normFrob(f_global);

    if (rank==0){
        cout <<"error: "<<error2 << endl;
    }

	test = test || !(error2<tol);

    x_global = 0;

	// No precond with overlap
    if (rank==0)
        std::cout<<"No precond without overlap:"<<std::endl;

    opt.parse("-hpddm_schwarz_method none");
    ddm_with_overlap.solve(f_global.data(),x_global.data(),mu);
	ddm_with_overlap.print_infos();
	error2=normFrob(f_global-A*x_global)/normFrob(f_global);

    if (rank==0){
        cout <<"error: "<<error2 << endl;
    }

	test = test || !(error2<tol);

    x_global = 0;


    // DDM one level ASM with overlap
    if (rank==0)
        std::cout<<"ASM one level without overlap:"<<std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    opt.parse("-hpddm_schwarz_method asm ");
    ddm_with_overlap.facto_one_level();
    ddm_with_overlap.solve(f_global.data(),x_global.data(),mu);
	ddm_with_overlap.print_infos();
	error2=normFrob(f_global-A*x_global)/normFrob(f_global);

    if (rank==0){
        cout <<"error: "<<error2 << endl;
    }

	test = test || !(error2<tol);

    x_global = 0;
	
    // DDM one level RAS with overlap
    if (rank==0)
        std::cout<<"RAS one level without overlap:"<<std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    opt.parse("-hpddm_schwarz_method ras ");
    ddm_with_overlap.solve(f_global.data(),x_global.data(),mu);
	ddm_with_overlap.print_infos();
	error2=normFrob(f_global-A*x_global)/normFrob(f_global);
    if (rank==0){
        cout <<"error: "<<error2 << endl;
    }

	test = test || !(error2<tol);

    x_global = 0;

	//Finalize the MPI environment.
	MPI_Finalize();

	return test;
}
