#include <htool/wrapper_hpddm.hpp>
#include <htool/fullACA.hpp>
#include <htool/hmatrix.hpp>
#include <htool/geometry.hpp>

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


int main(int argc, char const *argv[]){

	// // Input file
	// if ( argc != 5 ){ // argc should be 2 for correct execution
  //   // We print argv[0] assuming it is the program name
  //   cout<<"usage: "<< argv[0] <<" <matrixfile> <rhsfile> <meshfile> <solutionfile>\n";
	// 	return 1;
	// }

	// Initialize the MPI environment
	MPI_Init(NULL, NULL);

	// Get the number of processes
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// Get the rank of the process
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// HPDDM verbosity
	HPDDM::Option& opt = *HPDDM::Option::get();
	opt.val("tol", 0.0001);
	cout << opt["tol"]<< endl;
	// opt.parse(argc, argv, rank == 0);
	// cout << opt["tol"]<< endl;
	// if(rank != 0)
	// 	opt.remove("verbosity");
	//
	// //
	// bool test = 0;
	// const int ndistance = 4;
	// double distance[ndistance];
  // distance[0] = 3; distance[1] = 5; distance[2] = 7; distance[3] = 10;
	// SetNdofPerElt(1);
	// SetEpsilon(1e-6);
	// SetEta(0.1);
	// SetMinClusterSize(1);
	// //
	// // for(int idist=0; idist<ndistance; idist++)
	// // {
	// // 	cout << "Distance between the clusters: " << distance[idist] << endl;
	// //
	// // 	srand (1);
	// // 	// we set a constant seed for rand because we want always the same result if we run the check many times
	// // 	// (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)
	// //
	// // 	int nr = 100;
	// // 	int nc = 100;
	// // 	vector<int> Ir(nr); // row indices for the lrmatrix
	// // 	vector<int> Ic(nc); // column indices for the lrmatrix
	// //
	// // 	double z1 = 1;
	// // 	vector<R3>     p1(nr);
	// // 	vector<int>  tab1(nr);
	// // 	for(int j=0; j<nr; j++){
	// // 		Ir[j] = j;
	// // 		double rho = ((double) rand() / (double)(RAND_MAX)); // (double) otherwise integer division!
	// // 		double theta = ((double) rand() / (double)(RAND_MAX));
	// // 		p1[j][0] = sqrt(rho)*cos(2*M_PI*theta); p1[j][1] = sqrt(rho)*sin(2*M_PI*theta); p1[j][2] = z1;
	// // 		// sqrt(rho) otherwise the points would be concentrated in the center of the disk
	// // 		tab1[j]=j;
	// // 	}
	// // 	// p2: points in a unit disk of the plane z=z2
	// // 	double z2 = 1+distance[idist];
	// // 	vector<R3> p2(nc);
	// // 	vector<int> tab2(nc);
	// // 	for(int j=0; j<nc; j++){
  // //           Ic[j] = j;
	// // 		double rho = ((double) rand() / (RAND_MAX)); // (double) otherwise integer division!
	// // 		double theta = ((double) rand() / (RAND_MAX));
	// // 		p2[j][0] = sqrt(rho)*cos(2*M_PI*theta); p2[j][1] = sqrt(rho)*sin(2*M_PI*theta); p2[j][2] = z2;
	// // 		tab2[j]=j;
	// // 	}
	//
	// // Matrix
	// Matrix<complex<double>> A;
	// A.bytes_to_matrix(argv[1]);
	// int n = A.nb_rows();
	//
	// // Right-hand side
	// std::vector<complex<double>> f_global(n,1);
	// bytes_to_vector(f_global,argv[2]);
	//
	// // Mesh
	// std::vector<R3> p;
	// LoadGMSHMesh(p,argv[3]);
	//
	// // Hmatrix
	// std::vector<int> tab(n);
	// for (int i=0;i<n;i++){
	// 	tab[i]=i;
	// }
	// HMatrix<fullACA,complex<double>> HA(A,p,tab);
	//
	// // Global vectors
	// std::vector<complex<double>> x_global(n,0),x_ref(n);
	// bytes_to_vector(x_ref,argv[4]);
	//
	// // Local vectors
	// int n_local= HA.get_local_size_cluster();
	// const std::vector<std::vector<int>>& MasterClusters = HA.get_MasterClusters();
	// std::vector<complex<double>> x_local(n_local,0),f_local(n_local,1);
	// for (int i=0;i<n_local;i++){
	// 	f_local[i]=f_global[MasterClusters[rank][i]];
	// }
	//
	// // Solve
  // HPDDMOperator<fullACA,complex<double>> A_HPDDM(HA,A);
  // complex<double>* const rhs = &(f_local[0]);
  // complex<double>* x = &(x_local[0]);
  // HPDDM::IterativeMethod::solve(A_HPDDM, rhs, x, 1,HA.get_comm());
	//
	//
	// // Local to Global
	// std::vector<complex<double>> rcv(n);
	//
	// std::vector<int> recvcounts(size);
	// std::vector<int>  displs(size);
	//
	// displs[0] = 0;
	//
	// for (int i=0; i<size; i++) {
	// 	recvcounts[i] = MasterClusters[i].size();
	// 	if (i > 0)
	// 		displs[i] = displs[i-1] + recvcounts[i-1];
	// }
	//
	//
	// MPI_Allgatherv(&(x_local.front()), recvcounts[rank], MPI_DOUBLE_COMPLEX, &(rcv.front()), &(recvcounts.front()), &(displs.front()), MPI_DOUBLE_COMPLEX, HA.get_comm());
	//
	// for (int i=0; i<size; i++)
	// for (int j=0; j< MasterClusters[i].size(); j++)
	// 	x_global[MasterClusters[i][j]] = rcv[displs[i]+j];
	//
	// // Error on inversion
	// double inv_error2=norm2(f_global-A*x_global)/norm2(f_global);
	// double error2 = norm2(x_ref-x_global)/norm2(x_ref);
  // if (rank==0){
  //   cout <<"error on inversion : "<<inv_error2 << endl;
	// 	cout <<"error on solution  : "<<error2 << endl;
  // }
	// cout << test <<" "<<opt["tol"]<< endl;
	// test = test || !(inv_error2<opt["tol"]);
	// cout << test << endl;
	// test = test || !(error2<1e-5);
	//

	// Finalize the MPI environment.
	// cout << test << endl;
	MPI_Finalize();
	return 0;
}
