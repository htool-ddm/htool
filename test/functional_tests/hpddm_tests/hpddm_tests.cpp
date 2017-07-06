#include <htool/wrapper_hpddm.hpp>
#include <htool/fullACA.hpp>
#include <htool/matrix.hpp>
#include <htool/loading.hpp>


using namespace std;
using namespace htool;


class MyMatrix: public IMatrix<double>{
	const vector<R3>& p1;
	const vector<R3>& p2;

public:
	MyMatrix(const vector<R3>& p10,const vector<R3>& p20 ):IMatrix(p10.size(),p20.size()),p1(p10),p2(p20) {}

	double get_coef(const int& i, const int& j)const {return 1./(4*M_PI*norm(p1[i]-p2[j]));}


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

	// Initialize the MPI environment
	MPI_Init(NULL, NULL);

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
	SetNdofPerElt(3);
	SetEpsilon(1e-3);
	SetEta(0.5);
	SetMinClusterSize(100);
	typedef  std::complex<double> scalar_type;
	for(int idist=0; idist<1; idist++)
	{
		// cout << "Distance between the clusters: " << distance[idist] << endl;

		srand (1);
		// we set a constant seed for rand because we want always the same result if we run the check many times
		// (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)

		// int nr = 10000;
		// int nc = 10000;
		// vector<int> Ir(nr); // row indices for the lrmatrix
		// vector<int> Ic(nc); // column indices for the lrmatrix

		// double z1 = 1;
		// vector<R3>     p1(nr);
		// vector<int>  tab1(nr);
		// for(int j=0; j<nr; j++){
		// 	Ir[j] = j;
		// 	double rho = ((double) rand() / (double)(RAND_MAX)); // (double) otherwise integer division!
		// 	double theta = ((double) rand() / (double)(RAND_MAX));
		// 	p1[j][0] = sqrt(rho)*cos(2*M_PI*theta); p1[j][1] = sqrt(rho)*sin(2*M_PI*theta); p1[j][2] = z1;
		// 	// sqrt(rho) otherwise the points would be concentrated in the center of the disk
		// 	tab1[j]=j;
		// }
		// // p2: points in a unit disk of the plane z=z2
		// double z2 = 1+distance[idist];
		// vector<R3> p2(nc);
		// vector<int> tab2(nc);
		// for(int j=0; j<nc; j++){
    //         Ic[j] = j;
		// 	double rho = ((double) rand() / (RAND_MAX)); // (double) otherwise integer division!
		// 	double theta = ((double) rand() / (RAND_MAX));
		// 	p2[j][0] = sqrt(rho)*cos(2*M_PI*theta); p2[j][1] = sqrt(rho)*sin(2*M_PI*theta); p2[j][2] = z2;
		// 	tab2[j]=j;
		// }
		//
		// HPDDM::Option& opt = *HPDDM::Option::get();
    // opt.parse(argc, argv, rank == 0);
		// if(rank != 0)
    //     opt.remove("verbosity");
		// MyMatrix A(p1,p2);

		std::vector<R3> p;
		std::vector<double> r;
		LoadPoints("../data/maillage3600FracsV1DN1.txt",p,r);
		Matrix<scalar_type> A;
		A.bytes_to_matrix("../data/matrice3600FracsV1DN1.bin");

		int nc = A.nb_rows();

		vector<int> tab(A.nb_rows());
		for (int j=0;j<p.size();j++){
			tab[3*j]  = j;
			tab[3*j+1]= j;
			tab[3*j+2]= j;
		}

		HMatrix<fullACA,scalar_type> HA(A,p,tab,p,tab);
		int nbr_dmat = HA.get_ndmat();
		int nbr_lmat = HA.get_nlrmat();
		double comp = HA.compression();

		if (rank==0){
			cout << "nbr_dmat : "<<nbr_dmat<<endl;
			cout << "nbr_lmat : "<<nbr_lmat<<endl;
			cout << "compression : "<<comp<<endl;
		}
		HA.print_stats();
  	double mytime, maxtime, meantime;

		std::vector<scalar_type> x_ref(nc,1),f_ref(nc,1),f(nc,1),x_test_global(nc,1);
		vector<R3> p1_local(nc),p2_local(nc);
		f_ref=A*x_ref;
		nc=HA.get_local_size_cluster();
		const std::vector<std::vector<int>>& MasterClusters = HA.get_MasterClusters();
		std::vector<scalar_type> x_ref_local(nc,0),x_test_local(nc,0),f_local(nc,1);
		for (int i=0;i<nc;i++){
			f_local[i]=f[MasterClusters[rank][i]];
		}
		MPI_Barrier(HA.get_comm());
		mytime = MPI_Wtime();
    HA.mvprod_global(x_ref.data(),f.data());
		MPI_Barrier(HA.get_comm());
		mytime= MPI_Wtime() - mytime;
		MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0,HA.get_comm());
		MPI_Reduce(&mytime, &meantime, 1, MPI_DOUBLE, MPI_SUM, 0,HA.get_comm());
		meantime/=size;
		if (rank==0){
			cout << maxtime<<" "<<meantime<<endl;
		}
		MPI_Barrier(HA.get_comm());
		mytime = MPI_Wtime();
    HA.mvprod_local(x_ref_local.data(),f_local.data());
		MPI_Barrier(HA.get_comm());
		mytime= MPI_Wtime() - mytime;
		MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0,HA.get_comm());
		MPI_Reduce(&mytime, &meantime, 1, MPI_DOUBLE, MPI_SUM, 0,HA.get_comm());
		meantime/=size;
		if (rank==0){
			cout << maxtime<<" "<<meantime<<endl;
		}
double erreurFrob = Frobenius_absolute_error(HA,A);
if (rank==0){
	cout << "Errors with Frobenius norm: "<<erreurFrob/normFrob(A)<<endl;
	cout << "Error on mat vec prod : "<<norm2(f-f_ref)/norm2(f)<<std::endl;
}
		// cout << pow(norm2(fl),2)<<endl;

		// for (int i=0; i< HA.MasterClusters[rank].size(); i++) {
		// 	x_refl[i] -= x_ref[HA.MasterClusters[rank][i]];
		// }
		// cout <<"rank : "<<rank<<" "<< norm2(x_refl)<<endl;
		// double erreur2 = norm2(HA*x_ref-f)/norm2(f);
		// if (rank==0){
		// 	cout <<"error on mat vec prod : "<<erreur2 << endl;
		// }
		//
    // HPDDMOperator<fullACA,double> A_HPDDM(HA,A);
    // double* const rhs = &(f_local[0]);
    // double* x = &(x_test_local[0]);
		// double time = MPI_Wtime();
    // HPDDM::IterativeMethod::solve(A_HPDDM, rhs, x, 1,HA.get_comm());
		// if (rank==0){
		// 	cout << MPI_Wtime() - time<<endl;
		// }
		// std::vector<double> rcv;
		// rcv.resize(nr);
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
		// MPI_Allgatherv(&(x_test_local.front()), recvcounts[rank], MPI_DOUBLE, &(rcv.front()), &(recvcounts.front()), &(displs.front()), MPI_DOUBLE, HA.get_comm());
		//
		// for (int i=0; i<size; i++)
		// for (int j=0; j< MasterClusters[i].size(); j++)
		// 	x_test_global[MasterClusters[i][j]] = rcv[displs[i]+j];
		//
		//
		//
		// double inv_erreur2_global=norm2(f-A*x_test_global)/norm2(f);
		//
    // if (rank==0){
    //   cout <<"error on inversion : "<<inv_erreur2_global << endl;
		//
    // }
		// test = test || !(inv_erreur2<1.e-6); // default tol in hpddm

	}
	// Finalize the MPI environment.
	MPI_Finalize();
	return test;
}
